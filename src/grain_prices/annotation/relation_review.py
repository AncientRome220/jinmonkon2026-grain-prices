"""Concurrency-safe human review storage; proposals never imply confirmation."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import uuid

from grain_prices.canonical.io import read_jsonl, sha256_file, write_jsonl
from grain_prices.canonical.options import scan_options
from grain_prices.canonical.relations import target_id, validate_target, validate_gold
from grain_prices.canonical.text import normalize_with_map
from grain_prices.legacy.stage2 import tokenize_with_positions
from grain_prices.pipeline_shared import GRAIN_RE


def field_changes(before, after, path="relations"):
    if before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        return [change for k in sorted(set(before)|set(after))
                for change in field_changes(before.get(k),after.get(k),path+"."+k)]
    if isinstance(before, list) and isinstance(after, list):
        return [change for i in range(max(len(before),len(after)))
                for change in field_changes(before[i] if i<len(before) else None,
                                            after[i] if i<len(after) else None,path+f"[{i}]")]
    return [{"field":path,"before":before,"after":after}]


def grain_choices(row):
    mapped = normalize_with_map(row["source_block_text"])
    return [list(mapped.original_span(m.start(), m.end())) for m in GRAIN_RE.finditer(mapped.text)]


def reanchor(row, span):
    if span not in grain_choices(row):
        raise ValueError("choose a located grain mention in this block")
    result = deepcopy(row)
    result["grain_span"] = span
    result["grain_form"] = row["source_block_text"][slice(*span)]
    result["target_id"] = target_id(row["doc_id"], row["source_block_id"], row["source_block_sha256"], span)
    # Re-enumerate the same saved block so a distant new anchor does not inherit
    # the former anchor's clipped option inventory. No text preparation changes.
    mapped = normalize_with_map(row["source_block_text"])
    q, m = scan_options(mapped.text, tokenize_with_positions(mapped.text))
    for kind, prefix, options in (("qty_options", "q", q), ("money_options", "m", m)):
        result[kind] = []
        for index, option in enumerate(options):
            o = deepcopy(option)
            for stem in ("char", "denomination_char", "value_char"):
                a, b = stem+"_start", stem+"_end"
                if o.get(a) is not None:
                    o[a], o[b] = mapped.original_span(o[a], o[b])
            o["option_id"] = prefix+str(index)
            result[kind].append(o)
    for relation in result["relations"]:
        relation["grain_span"] = span
    result.update(review_status="", reviewed_by="", coverage="unreviewed")
    return result


def save_review(path: Path, expected_hash: str, previous_id: str, edited: dict, seconds=0.0):
    """CAS under an exclusive lock, immutable backup, then atomic replacement."""
    path = Path(path)
    lock = path.with_suffix(path.suffix+".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError("another save is in progress; reload after it finishes") from None
    temp = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        os.close(descriptor)
        if sha256_file(path) != expected_hash:
            raise ValueError("review changed on disk; reload before saving")
        rows = list(read_jsonl(path))
        old = next(r for r in rows if r["target_id"] == previous_id)
        row = deepcopy(edited)
        validate_target(row)
        for field in ("doc_id", "source_block_id", "source_block_sha256", "source_block_text", "origin_sample_ids"):
            if row[field] != old[field]:
                raise ValueError("save cannot change source identity or historical origins")
        for field in ('origin_annotations', 'relation_proposals', 'prediction_before_review', 'review_phase'):
            if row.get(field) != old.get(field):
                raise ValueError('save cannot change original annotations or prediction provenance')
        if row.get("review_status") == "confirmed":
            validate_gold([row])
        if seconds < 0:
            raise ValueError("negative review time")
        row["review_sessions"] = old.get("review_sessions", []) + [{
            "saved_utc": datetime.now(timezone.utc).isoformat(), "active_seconds": round(seconds, 3),
            "phase": row.get("review_phase", "target_gold_review"), "reviewer": row.get("reviewed_by", ""),
            "changed_fields": sorted(k for k in row if row.get(k) != old.get(k)),
            "relation_field_changes": field_changes(old.get("relations"),row.get("relations")),
            "context_field_changes": field_changes(old.get("reviewed_contexts", old.get("prediction_before_review", {}).get("contexts")), row.get("reviewed_contexts")),
        }]
        others = [r for r in rows if r["target_id"] != previous_id]
        duplicate = next((r for r in others if r["target_id"] == row["target_id"]), None)
        if duplicate:
            others.remove(duplicate)
            for field in ("origin_sample_ids", "origin_annotations", "relation_proposals", "relations", "review_sessions"):
                row[field] = row.get(field, []) + duplicate.get(field, [])
            row["origin_sample_ids"] = sorted(row["origin_sample_ids"])
            row.update(review_status="", reviewed_by="", coverage="unreviewed")
            row["preparation_flags"] = sorted(set(row.get("preparation_flags", []) + ["merged_targets_require_reconciliation"]))
        result = sorted(others+[row], key=lambda r: (r["doc_id"], r["source_block_id"], r["grain_span"]))
        before = sorted(s for r in rows for s in r["origin_sample_ids"])
        after = sorted(s for r in result for s in r["origin_sample_ids"])
        if before != after or len(after) != len(set(after)):
            raise ValueError("historical sample ledger changed")
        backup = path.parent / "backups" / (expected_hash+".jsonl")
        backup.parent.mkdir(exist_ok=True)
        if not backup.exists():
            shutil.copy2(path, backup)
        write_jsonl(temp, result)
        os.replace(temp, path)
        return {"sha256": sha256_file(path), "target_id": row["target_id"], "merged": bool(duplicate)}
    finally:
        if temp.exists():
            temp.unlink()
        lock.unlink()

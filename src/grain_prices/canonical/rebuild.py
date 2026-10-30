"""Offline rebuild audit and explicit human-review gate for a historical sample."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from grain_prices.canonical.annotation import validate_annotations
from grain_prices.canonical.io import read_jsonl, sha256_file, write_jsonl
from grain_prices.canonical.options import currency_name
from grain_prices.canonical.text import normalize_with_map


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def projection(candidate: dict) -> dict:
    return {kind: [{k: o.get(k) for k in ("value", "unit", "currency", "char_start", "char_end")}
                   for o in candidate.get(kind, [])] for kind in ("qty_options", "money_options")}


def review_display_fields(candidate: dict | None) -> dict:
    """Refresh machine displays only; never replace human readings or selections."""
    candidate = candidate or {}
    quantities = candidate.get("qty_options", [])
    money = candidate.get("money_options", [])
    return {
        "quantity_option_choices": ";".join(o["option_id"] for o in quantities),
        "money_option_choices": ";".join(o["option_id"] for o in money),
        "option_details_json": json.dumps({"quantity": quantities, "money": money},
                                          ensure_ascii=False, sort_keys=True),
        "n_qty_options": str(len(quantities)), "n_money_options": str(len(money)),
        "evidence_text": candidate.get("evidence_text", ""),
        "window_text": candidate.get("window_text", ""),
    }


def map_option(old: dict, new: dict, kind: str, selected: str) -> tuple[str, str]:
    if not selected:
        return "", "original_null_requires_review"
    option = next((o for o in old.get(kind, []) if o["option_id"] == selected), None)
    if option is None:
        return "", "old_option_absent"
    matches = []
    for other in new.get(kind, []):
        # Position must identify the same printed denomination, not merely the
        # same numerical value elsewhere in an account. Old loose anchors are
        # word spans; old strict anchors include the amount.
        anchor = other.get("denomination_char_start")
        if anchor is None or not (option["char_start"] <= anchor < option["char_end"]):
            continue
        name = "currency" if kind == "money_options" else "unit"
        normalize = currency_name if name == "currency" else lambda v: normalize_with_map(v or "").text
        if normalize(option.get(name)) != normalize(other.get(name)):
            continue
        old_value, new_value = option.get("value"), other.get("value")
        try:
            equal_value = abs(float(old_value) - float(new_value)) < 1e-9
        except (ValueError, TypeError):
            equal_value = old_value is None and new_value is None
        if equal_value and not other.get("ownership_ambiguous"):
            matches.append(other["option_id"])
    return (matches[0], "same_position_and_value") if len(matches) == 1 else ("", "changed_or_ambiguous")


def prepare(old_path: Path, new_path: Path, gold_path: Path, out: Path) -> dict:
    old = {c["window_id"]: c for c in read_jsonl(old_path)}
    new = {c["window_id"]: c for c in read_jsonl(new_path)}
    gold = read_csv(gold_path)
    out.mkdir(parents=True, exist_ok=True)
    common = old.keys() & new.keys()
    counts = {"old_candidates": len(old), "new_candidates": len(new),
              "added_windows": len(new.keys() - old.keys()), "removed_windows": len(old.keys() - new.keys()),
              "shared_windows": len(common),
              "shared_windows_with_changed_options": sum(projection(old[k]) != projection(new[k]) for k in common)}
    changes = [{"window_id": key, "doc_id": (new.get(key) or old[key])["ddb_id"],
                "change": "added" if key not in old else "removed" if key not in new
                else "changed" if projection(old[key]) != projection(new[key]) else "unchanged"}
               for key in sorted(old.keys() | new.keys())]
    write_csv(out / "candidate_changes.csv", changes)
    reviews, details = [], []
    for original in gold:
        key = original["window_id"]
        before, after = old[key], new.get(key)
        proposed = dict(original)
        proposed.update(review_display_fields(after))
        proposed.update(review_status="", reviewed_by="", review_note="",
                        candidate_fingerprint=after.get("candidate_fingerprint", "") if after else "",
                        extraction_status="present" if after else "missing")
        for kind, column in (("qty_options", "quantity_option"), ("money_options", "money_option")):
            selected, status = map_option(before, after, kind, original.get(column, "")) if after else ("", "window_missing")
            proposed["original_" + column] = original.get(column, "")
            proposed[column] = selected
            proposed[column + "_mapping"] = status
        reviews.append(proposed)
        details.append({"sample_id": original["sample_id"], "original_annotation": original,
                        "old_candidate": before, "new_candidate": after,
                        "proposed_mapping": {c: proposed[c] for c in ("quantity_option", "money_option")}})
    write_csv(out / "gold_review.csv", reviews)
    write_jsonl(out / "gold_review_evidence.jsonl", details)
    subset = [new[r["window_id"]] for r in gold if r["window_id"] in new]
    write_jsonl(out / "gold_candidates_unreviewed.jsonl", subset)
    counts.update(gold_windows=len(gold), gold_windows_present=len(subset),
                  gold_windows_missing=len(gold)-len(subset), human_review_complete=False)
    all_options = [o for c in new.values() for o in c["qty_options"] + c["money_options"]]
    counts["option_occurrences"] = len(all_options)
    counts["inference_option_occurrences"] = dict(Counter(o["inference_rule"] for o in all_options if o.get("inference_rule")))
    counts["ambiguous_ownership_option_occurrences"] = sum(bool(o.get("ownership_ambiguous")) for o in all_options)
    counts["money_option_occurrences_by_currency"] = dict(sorted(Counter(currency_name(o.get("currency")) or "unstated"
        for c in new.values() for o in c["money_options"]).items()))
    collisions = 0
    for candidate in new.values():
        spans = [(o["value_char_start"], o["value_char_end"]) for o in candidate["qty_options"] + candidate["money_options"]
                 if o.get("value_char_start") is not None]
        collisions += len(spans) - len(set(spans))
    counts["reused_numeral_spans_within_windows"] = collisions
    counts["observation_deduplication"] = "awaiting_new_judgments; candidate counts are not observation counts"
    manifest = {"status": "awaiting_human_review", "result": counts,
                "inputs": {"old_candidates": {"path": old_path.as_posix(), "sha256": sha256_file(old_path)},
                           "new_candidates": {"path": new_path.as_posix(), "sha256": sha256_file(new_path)},
                           "original_gold": {"path": gold_path.as_posix(), "sha256": sha256_file(gold_path)}}}
    write_json(out / "manifest_rebuild_audit.json", manifest)
    return counts


def freeze(review_path: Path, candidates_path: Path, audit_path: Path, out: Path) -> dict:
    """Only human-confirmed rows become a scoreable replacement gold artifact."""
    rows = read_csv(review_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if sha256_file(candidates_path) != audit["inputs"]["new_candidates"]["sha256"]:
        raise ValueError("candidates changed since review preparation")
    original = read_csv(Path(audit["inputs"]["original_gold"]["path"]))
    original_hash = audit["inputs"]["original_gold"].get("sha256")
    if original_hash and sha256_file(Path(audit["inputs"]["original_gold"]["path"])) != original_hash:
        raise ValueError("historical gold changed since review preparation")
    if len(rows) != len(original) or {r["sample_id"] for r in rows} != {r["sample_id"] for r in original}:
        raise ValueError("review must retain every historical sample row exactly once")
    original_windows = {r["sample_id"]: r["window_id"] for r in original}
    candidates = {c["window_id"]: c for c in read_jsonl(candidates_path)}
    for row in rows:
        if row["review_status"] != "confirmed" or not row["reviewed_by"].strip():
            raise ValueError("human review incomplete: " + row["sample_id"])
        if row.get("canonical_label") not in {"PRICE", "NO_PRICE"}:
            raise ValueError("every historical window requires a label, including extraction failures")
        if row["window_id"] != original_windows[row["sample_id"]]:
            raise ValueError("historical sample identity changed")
        candidate = candidates.get(row["window_id"])
        if candidate and row["candidate_fingerprint"] != candidate["candidate_fingerprint"]:
            raise ValueError("reviewed candidate fingerprint differs")
        if row["extraction_status"] != ("present" if candidate else "missing"):
            raise ValueError("extraction status differs from candidate membership")
    out.mkdir(parents=True, exist_ok=True)
    present = [r for r in rows if r["window_id"] in candidates]
    validation_path = out / "reviewed_present_gold.csv"
    write_csv(validation_path, present)
    errors = validate_annotations(validation_path, candidates_path)
    if errors:
        raise ValueError("invalid reviewed annotations: " + "; ".join(errors[:10]))
    write_csv(out / "reviewed_gold150.csv", rows)
    subset = [candidates[r["window_id"]] for r in present]
    write_jsonl(out / "reviewed_gold_candidates.jsonl", subset)
    selected = random.Random(20261023).sample(sorted(subset, key=lambda c: c["window_id"]), min(30, len(subset)))
    write_jsonl(out / "stability30_candidates.jsonl", selected)
    selected_ids = {r["window_id"] for r in selected}
    write_csv(out / "stability30_gold.csv", [r for r in rows if r["window_id"] in selected_ids])
    result = {"status": "reviewed_awaiting_run_authorization", "historical_denominator": len(rows),
              "scoreable_windows": len(subset), "extraction_failures": len(rows)-len(subset),
              "stability_windows": len(selected), "seed": 20261023,
              "review_sha256": sha256_file(review_path), "candidates_sha256": sha256_file(candidates_path)}
    write_json(out / "manifest_review_freeze.json", result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare")
    for name in ("old-candidates", "new-candidates", "gold", "output"):
        prepare_parser.add_argument("--" + name, required=True, type=Path)
    freeze_parser = sub.add_parser("freeze")
    for name in ("review", "candidates", "audit", "output"):
        freeze_parser.add_argument("--" + name, required=True, type=Path)
    args = p.parse_args()
    result = (prepare(args.old_candidates, args.new_candidates, args.gold, args.output)
              if args.action == "prepare" else freeze(args.review, args.candidates, args.audit, args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import re
import json
import logging
from collections import Counter
from pathlib import Path

from grain_prices.canonical import BUILD_VERSION, DEFINITION_VERSION, NORM_VERSION
from grain_prices.canonical.text import normalize_with_map, stable_window_id
from grain_prices.pipeline_shared import GRAIN_FALSE_STEMS, GRAIN_RE, MONEY_RE, PRICEWORD_RE, UNIT_RE
from grain_prices.canonical.options import MONEY_RE
from grain_prices.preprocessing.parse_xml import HgvMeta, iter_xml_files, load_hgv_index, parse_ddb_document

KRINO_BLOCKLIST = re.compile(r"^κριθ(εν|εντ\w*|εις|εισ\w+|ηναι|ητω|ωσ\w*|ησαν)$")
log = logging.getLogger(__name__)


def _known_false(norm_text: str, start: int, end: int, form: str) -> tuple[bool, str]:
    if KRINO_BLOCKLIST.match(form):
        return True, "krino_form"
    for match in GRAIN_FALSE_STEMS.finditer(norm_text):
        if match.start() <= start < match.end():
            return True, "known_false_stem"
    return False, ""


def build_spine(
    ddb_dir: Path,
    out_dir: Path,
    hgv_dir: Path | None = None,
    hgv_cache: Path | None = None,
    limit: int = 0,
    window_chars: int = 200,
) -> dict:
    hgv = {}
    if hgv_dir is not None:
        hgv = load_hgv_index(hgv_dir, cache_path=hgv_cache)
    files = sorted(
        iter_xml_files(ddb_dir),
        key=lambda path: path.relative_to(ddb_dir).as_posix(),
    )
    if limit:
        files = files[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    lexical_count = candidate_count = 0
    candidate_docs: set[str] = set()
    seen_ids: Counter[str] = Counter()
    with (out_dir / "lexical_hits.jsonl").open("w", encoding="utf-8", newline="\n") as lexical_out, \
            (out_dir / "candidate_mentions.jsonl").open("w", encoding="utf-8", newline="\n") as candidate_out:
        for file_index, xml_path in enumerate(files, 1):
            if file_index % 5000 == 0:
                log.info("spine %d/%d XML; lexical=%d candidates=%d",
                         file_index, len(files), lexical_count, candidate_count)
            doc = parse_ddb_document(xml_path, hgv_index=hgv or None)
            if doc is None:
                continue
            logical_xml_path = xml_path.relative_to(ddb_dir).as_posix()
            meta = doc.metadata or HgvMeta()
            full_surface = doc.full_text
            block_offset = 0
            for block_index, (_tag, block_surface) in enumerate(doc.blocks):
                mapped = normalize_with_map(block_surface)
                for match in GRAIN_RE.finditer(mapped.text):
                    local_os, local_oe = mapped.original_span(match.start(), match.end())
                    os, oe = block_offset + local_os, block_offset + local_oe
                    half = window_chars // 2
                    ws, we = max(0, os - half), min(len(full_surface), oe + half)
                    display_window = full_surface[ws:we].strip()
                    norm_window = normalize_with_map(display_window).text
                    identity = block_surface[
                        max(0, local_os - 100):min(len(block_surface), local_oe + 100)
                    ].strip()
                    false, false_reason = _known_false(
                        mapped.text, match.start(), match.end(), match.group(0)
                    )
                    grain_surface = block_surface[local_os:local_oe]
                    base_id = stable_window_id(doc.ddb_id, grain_surface, identity)
                    seen_ids[base_id] += 1
                    window_id = base_id if seen_ids[base_id] == 1 else f"{base_id}~{seen_ids[base_id]}"
                    row = {
                        "window_id": window_id,
                        "doc_id": doc.ddb_id,
                        "xml_path": logical_xml_path,
                        "block_index_debug": block_index,
                        "grain_surface": grain_surface,
                        "grain_normalized": match.group(0),
                        "char_offset_debug": os,
                        "window_text": display_window,
                        "norm_window": norm_window,
                        "has_number": bool(re.search(r"\d", display_window)),
                        "has_unit": bool(UNIT_RE.search(norm_window)),
                        "has_money": bool(MONEY_RE.search(norm_window)),
                        "has_priceword": bool(PRICEWORD_RE.search(norm_window)),
                        "known_false": false,
                        "false_reason": false_reason,
                        "place": meta.place,
                        "date_not_before": meta.date_notbefore,
                        "date_not_after": meta.date_notafter,
                        "definition_version": DEFINITION_VERSION,
                        "build_version": BUILD_VERSION,
                        "norm_version": NORM_VERSION,
                    }
                    lexical_out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    lexical_count += 1
                    if not false:
                        candidate_out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                        candidate_count += 1
                        candidate_docs.add(doc.ddb_id)
                block_offset += len(block_surface) + len(" ||| ")
    return {
        "documents_scanned": len(files),
        "lexical_hits": lexical_count,
        "candidate_mentions": candidate_count,
        "known_false_hits": lexical_count - candidate_count,
        "candidate_documents": len(candidate_docs),
    }

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from grain_prices.canonical import BUILD_VERSION, DEFINITION_VERSION
from grain_prices.canonical.text import stable_window_id
from grain_prices.legacy import stage2c
from grain_prices.canonical.enumeration import enumerate_for_doc
from grain_prices.canonical.provenance import candidate_fingerprint
from grain_prices.canonical import CANDIDATE_SCHEMA_VERSION
from grain_prices.preprocessing.parse_xml import HgvMeta, iter_xml_files, load_hgv_index

log = logging.getLogger(__name__)


def generate_candidates(
    ddb_dir: Path,
    out_path: Path,
    hgv_dir: Path | None = None,
    hgv_cache: Path | None = None,
    limit: int = 0,
    window_tokens: int = 80,
) -> dict:
    pats = stage2c.S2.compile_patterns()
    hgv = {}
    if hgv_dir is not None:
        hgv = load_hgv_index(hgv_dir, cache_path=hgv_cache)
    files = sorted(
        iter_xml_files(ddb_dir),
        key=lambda path: path.relative_to(ddb_dir).as_posix(),
    )
    if limit:
        files = files[:limit]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_count = raw_count = false_count = 0
    candidate_docs: set[str] = set()
    seen: Counter[str] = Counter()
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for file_index, xml_path in enumerate(files, 1):
            if file_index % 5000 == 0:
                log.info("candidates %d/%d XML; raw=%d kept=%d",
                         file_index, len(files), raw_count, output_count)
            rows = enumerate_for_doc(
                xml_path, pats, HgvMeta(), window_tokens, evidence_chars=60,
                max_evidence=1200,
            )
            for row in rows:
                raw_count += 1
                if row.get("flag_krino") or row.get("flag_admin_compound") or row.get("flag_false_stem"):
                    false_count += 1
                    continue
                meta = hgv.get(row["ddb_id"])
                if meta:
                    row.update({
                        "place": meta.place,
                        "date_not_before": meta.date_notbefore,
                        "date_not_after": meta.date_notafter,
                    })
                # Use the same 200-character document window as build-spine.
                # The full block is retained by Stage 2C solely for this stable
                # identity projection; offsets remain debug attributes.
                block_surface = row["identity_source_text"]
                grain_surface_start = row["identity_grain_start_debug"]
                grain_surface_end = row["identity_grain_end_debug"]
                identity = block_surface[
                    max(0, grain_surface_start - 100):min(len(block_surface), grain_surface_end + 100)
                ].strip()
                base_id = stable_window_id(row["ddb_id"], row["grain_form"], identity)
                seen[base_id] += 1
                row["window_id"] = base_id if seen[base_id] == 1 else f"{base_id}~{seen[base_id]}"
                row["legacy_mention_id"] = row.pop("mention_id", "")
                row.pop("identity_source_text", None)
                row["definition_version"] = DEFINITION_VERSION
                row["build_version"] = BUILD_VERSION
                row["doc_id"] = row["ddb_id"]
                row["grain_surface"] = row["grain_form"]
                row["window_text"] = row["evidence_text"]
                row["candidate_schema_version"] = CANDIDATE_SCHEMA_VERSION
                row["candidate_fingerprint"] = candidate_fingerprint(row)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                output_count += 1
                candidate_docs.add(row["ddb_id"])
    return {
        "documents_scanned": len(files),
        "raw_candidate_mentions": raw_count,
        "candidate_mentions": output_count,
        "known_false_removed": false_count,
        "candidate_documents": len(candidate_docs),
    }

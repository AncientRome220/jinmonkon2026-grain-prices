"""Conservative source grouping without discarding window judgments (#036)."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from grain_prices.canonical.text import normalize_with_map


def locate_evidence(candidate: dict, quote: str) -> list[int] | None:
    mapped = normalize_with_map(candidate.get("source_block_text", ""))
    needle = normalize_with_map(quote).text
    if not needle:
        return None
    start = mapped.text.find(needle)
    if start < 0 or mapped.text.find(needle, start + 1) >= 0:
        return None
    return list(mapped.original_span(start, start + len(needle)))


def source_key(row: dict) -> tuple | None:
    if not row.get("source_block_id") or not row.get("source_block_sha256"):
        return None
    qty = row.get("quantity_value_span") or row.get("quantity_char_span")
    money = row.get("price_value_span") or row.get("price_char_span")
    quote = row.get("evidence_char_span")
    if (not qty or not money) and not quote:
        return None
    return (row.get("method"), row.get("repeat_index"), row.get("doc_id"),
            row["source_block_id"], row["source_block_sha256"], row.get("coordinate_system"),
            tuple(qty) if qty else None, tuple(money) if money else None,
            tuple(quote) if (not qty or not money) else None)


INTERPRETATION_FIELDS = (
    "commodity", "commodity_class", "quantity_value", "quantity_unit_canonical",
    "price_value_numeric", "price_currency_canonical", "rate_basis", "context_tags",
    "normalization_status", "price_per_artaba_drachma", "in_scope", "analysis_eligible",
    "series_strict", "series_broad", "series_exploratory", "linkage_strength",
)


def deduplicate_observations(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        key = source_key(row)
        groups[key if key is not None else ("unresolved", index)].append(row)
    result = []
    for key, members in groups.items():
        fingerprints = set()
        for row in members:
            values = {field: row.get(field) for field in INTERPRETATION_FIELDS}
            values["context_tags"] = sorted(set(values["context_tags"] or []))
            fingerprints.add(json.dumps(values, sort_keys=True, ensure_ascii=False))
        unresolved = key[0] == "unresolved"
        conflict = len(fingerprints) > 1
        row = dict(members[0])
        row.update(
            observation_group_id=hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest(),
            grouping_status="unresolved_location" if unresolved else "interpretation_conflict" if conflict else "agreed",
            contributing_window_ids=sorted({m["window_id"] for m in members}),
            contributing_observation_count=len(members),
            source_observations=members,
        )
        if conflict or unresolved:
            row["analysis_eligible"] = False
            for series in ("strict", "broad", "exploratory"):
                row[f"series_{series}"] = False
            if conflict:
                # No first-row interpretation is presented as the group result.
                for field in INTERPRETATION_FIELDS:
                    if field not in {"analysis_eligible", "series_strict", "series_broad", "series_exploratory"}:
                        row[field] = None
        result.append(row)
    return result

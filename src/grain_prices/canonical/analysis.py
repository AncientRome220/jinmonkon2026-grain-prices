from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from grain_prices.canonical.io import read_jsonl
from grain_prices.canonical.observations import deduplicate_observations

# Coarse bins by the midpoint of the HGV date range; the boundaries follow the
# currency periods #041 found (Ptolemaic obols and chalkous, post-250 denarii).
PERIOD_BINS = [
    (None, -30, "before_-30"),
    (-30, 100, "-30_to_99"),
    (100, 200, "100_to_199"),
    (200, 250, "200_to_249"),
    (250, 300, "250_to_299"),
    (300, None, "300_and_later"),
]


def _year(value: str) -> int | None:
    """Signed year of an HGV date (``-0255``, ``0155-02-04``); None if absent."""
    value = (value or "").strip()
    if not value:
        return None
    sign, digits = (-1, value[1:]) if value.startswith("-") else (1, value)
    try:
        return sign * int(digits.split("-")[0])
    except ValueError:
        return None


def _period(row: dict) -> str:
    years = [y for y in (_year(row.get("date_not_before", "")),
                         _year(row.get("date_not_after", ""))) if y is not None]
    if not years:
        return "undated"
    midpoint = sum(years) / len(years)
    for low, high, label in PERIOD_BINS:
        if (low is None or midpoint >= low) and (high is None or midpoint < high):
            return label
    raise AssertionError(midpoint)


def _coverage(rows: list[dict]) -> dict:
    with_rate = sum(r.get("normalization_status") == "direct_drachma_per_artaba" for r in rows)
    return {"observations": len(rows), "with_rate": with_rate,
            "without_rate": len(rows) - with_rate,
            "share_with_rate": round(with_rate / len(rows), 4) if rows else None}


def conversion_coverage(observations: list[dict]) -> dict:
    """How much of each series carries a per-artaba drachma rate, and what is left out (#041).

    A series built from these observations silently drops every row without a
    rate, and the missing part is concentrated by currency and period, so any
    figure drawn from it should state this coverage alongside its n.
    """
    def grouped(key) -> dict:
        groups: dict[str, list[dict]] = {}
        for row in observations:
            groups.setdefault(key(row), []).append(row)
        return {name: _coverage(rows) for name, rows in sorted(groups.items())}

    in_scope = [r for r in observations if r.get("in_scope")]
    return {
        "overall": _coverage(observations),
        "by_series": {name: _coverage([r for r in in_scope if r.get(f"series_{name}")])
                      for name in ("strict", "broad", "exploratory")},
        "by_currency": grouped(lambda r: r.get("price_currency_canonical") or "none"),
        "by_period": grouped(_period),
        "by_normalization_status": dict(sorted(
            Counter(r.get("normalization_status") or "none" for r in observations).items())),
    }


def duplicate_observations(observations: list[dict], series: dict[str, list[dict]]) -> dict:
    """Rows recording the same economic fact twice, via two overlapping windows (#036).

    A fact is one method's selection of the same quantity and money option spans in
    one document. Rows without a money span (no option selected, or a file written
    before the spans existed) cannot be matched and are counted separately.
    The rows are reported, not collapsed: collapsing moves every series count.
    """
    def key(row: dict):
        qty = row.get("quantity_char_span")
        return (row.get("method"), row.get("repeat_index"), row.get("doc_id"),
                row.get("source_block_id"), row.get("source_block_sha256"),
                tuple(qty) if qty else None, tuple(row["price_char_span"]))

    def excess(rows: list[dict]) -> dict:
        groups = Counter(key(r) for r in rows if r.get("price_char_span"))
        repeated = [n for n in groups.values() if n > 1]
        return {"facts_recorded_more_than_once": len(repeated),
                "rows_involved": sum(repeated),
                "excess_rows": sum(n - 1 for n in repeated)}

    return {
        **excess(observations),
        "rows_without_price_span": sum(not r.get("price_char_span") for r in observations),
        "by_series": {name: excess(rows) for name, rows in series.items()},
    }


def analyze(observations_path: Path, out_dir: Path) -> dict:
    observations = list(read_jsonl(observations_path))
    emitted_count = len(observations)
    is_v2 = any(r.get("observation_schema_version") for r in observations)
    if is_v2 and not all(r.get("grouping_status") for r in observations):
        observations = deduplicate_observations(observations)
    out_dir.mkdir(parents=True, exist_ok=True)
    series = {
        "strict": [r for r in observations if r.get("series_strict") and r.get("analysis_eligible")],
        "broad": [r for r in observations if r.get("series_broad") and r.get("analysis_eligible")],
        "exploratory": [r for r in observations if r.get("series_exploratory") and r.get("analysis_eligible")],
    }
    fields = sorted({key for row in observations for key in row})
    if not fields:
        fields = [
            "window_id", "observation_index", "doc_id", "commodity",
            "commodity_class", "quantity_value", "quantity_unit", "price_value",
            "price_currency", "linkage_strength", "context_tags", "in_scope",
            "price_per_artaba_drachma", "normalization_status",
        ]
    for name, rows in series.items():
        with (out_dir / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "context_tags": ",".join(row.get("context_tags") or [])})
    excluded = [row for row in observations if not row.get("analysis_eligible")]
    with (out_dir / "excluded_from_public_analysis.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in excluded:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metrics = {
        "canonical_results_status": "measured" if observations else "awaiting_external_llm_run",
        "final_observations": len(observations),
        "counting_unit": "source_groups_including_unresolved_rows" if is_v2 else "historical_window_observations",
        "emitted_window_observations": (sum(r.get("contributing_observation_count", 1) for r in observations)
                                         if is_v2 else emitted_count),
        "interpretation_conflicts": sum(r.get("grouping_status") == "interpretation_conflict" for r in observations),
        "unresolved_location_rows": sum(r.get("grouping_status") == "unresolved_location" for r in observations),
        "series_counts": {key: len(value) for key, value in series.items()},
        "direct_drachma_artaba": sum(
            r.get("normalization_status") == "direct_drachma_per_artaba" for r in observations
        ),
        "unapproved_normalization": sum(
            r.get("normalization_status") == "unapproved_factor" for r in observations
        ),
        "excluded_from_public_analysis": len(excluded),
        "linkage_counts": dict(Counter(r.get("linkage_strength") or "unresolved" for r in observations)),
        "commodity_class_counts": dict(Counter(r.get("commodity_class") or "unresolved" for r in observations)),
        "conversion_coverage": conversion_coverage(observations),
        "duplicate_observations": duplicate_observations(observations, series),
    }
    (out_dir / "paper_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    return metrics

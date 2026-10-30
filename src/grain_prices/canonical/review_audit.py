"""Mechanical triage of historical gold mappings, without human sign-off or LLMs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re

from grain_prices.canonical.annotation import validate_annotations
from grain_prices.canonical.io import read_jsonl, sha256_file
from grain_prices.canonical.observations import locate_evidence
from grain_prices.canonical.provenance import candidate_fingerprint
from grain_prices.canonical.rebuild import read_csv, review_display_fields, write_csv, write_json
from grain_prices.canonical.text import normalize_with_map


def norm(value) -> str:
    return normalize_with_map(str(value or "")).text


def rounding_only(old, new) -> bool:
    """A diagnostic, not permission to change the selected option or reading."""
    try:
        before, after = Decimal(str(old)), Decimal(str(new))
        return before.is_finite() and after.is_finite() and "." in str(old) and (
            before != after and abs(before - after) <= Decimal(5).scaleb(before.as_tuple().exponent - 1))
    except InvalidOperation:
        return False


def inside(option: dict, quote_span: list[int] | None) -> bool:
    if not quote_span:
        return False
    start = option.get("denomination_char_start")
    if start is None:
        start = option.get("value_char_start")
    return start is not None and quote_span[0] <= start < quote_span[1]


def describe(option: dict, candidate: dict) -> str:
    start, end = option.get("char_start", 0), option.get("char_end", 0)
    text = candidate.get("source_block_text", "")[start:end]
    denomination = option.get("unit") or option.get("currency") or "unstated denomination"
    return (f"{option['option_id']}: value={option.get('value')}, {denomination}; text={text!r}; "
            f"rule={option.get('inference_rule') or 'explicit denomination'}, "
            f"multiplier={option.get('multiplier', 1)}, ambiguous={bool(option.get('ownership_ambiguous'))}")


def audit_review(review: Path, evidence: Path, candidates_path: Path,
                 output: Path, refresh_display: bool = False) -> dict:
    rows = read_csv(review)
    details_list = list(read_jsonl(evidence))
    details = {d["sample_id"]: d for d in details_list}
    candidates_list = list(read_jsonl(candidates_path))
    candidates = {c["window_id"]: c for c in candidates_list}
    if len(candidates) != len(candidates_list):
        raise ValueError("duplicate candidate window IDs")
    if (len(details) != len(details_list) or len(rows) != len(details)
            or len({r["sample_id"] for r in rows}) != len(rows)
            or {r["sample_id"] for r in rows} != set(details)):
        raise ValueError("review/evidence sample membership differs")
    # Check bindings before any repair. Evidence snapshots are not silently updated.
    for row in rows:
        detail = details[row["sample_id"]]
        candidate = candidates.get(row["window_id"])
        if row["window_id"] != detail["original_annotation"]["window_id"]:
            raise ValueError("historical window identity changed")
        if candidate != detail["new_candidate"]:
            raise ValueError("candidate differs from review evidence snapshot")
        if candidate and (candidate_fingerprint(candidate) != candidate.get("candidate_fingerprint")
                          or row["candidate_fingerprint"] != candidate["candidate_fingerprint"]):
            raise ValueError("candidate fingerprint mismatch")

    before_hash = sha256_file(review)
    reports, source_groups = [], defaultdict(list)
    for row in rows:
        candidate = candidates.get(row["window_id"])
        detail = details[row["sample_id"]]
        display = review_display_fields(candidate)
        stale = [k for k, v in display.items() if str(row.get(k, "")) != str(v)]
        report = {"sample_id": row["sample_id"], "doc_id": row["doc_id"],
                  "canonical_label": row["canonical_label"], "flags": [], "notes": [],
                  "stale_display_fields": stale, "option_diagnostics": [],
                  "evidence_span": row.get("evidence_span", ""),
                  "human_notes": row.get("Human_Notes", ""),
                  "context_note": row.get("context_note", ""),
                  "exclusion_reason": row.get("exclusion_reason", ""),
                  "candidate_excerpt": candidate.get("evidence_text", "") if candidate else "",
                  "human_reading": {k: row.get(k, "") for k in (
                      "Human_Qty", "Human_Unit", "Human_Price", "Human_Currency", "rate_basis")}}
        reports.append(report)
        if not candidate:
            report["flags"].append("missing_candidate_retained_in_denominator")
            continue
        quote = locate_evidence(candidate, row.get("evidence_span", "")) if row.get("evidence_span") else None
        if row["canonical_label"] == "PRICE" and not quote:
            report["flags"].append("quote_missing_or_not_uniquely_located")
        if norm(candidate.get("evidence_text")) != norm(detail["old_candidate"].get("evidence_text")):
            report["flags"].append("window_text_changed")
        if quote:
            source_groups[(row["doc_id"], candidate["source_block_id"],
                           candidate["source_block_sha256"], tuple(quote))].append((row, report))
        owned = []
        for kind, column in (("qty_options", "quantity_option"), ("money_options", "money_option")):
            options = candidate[kind]
            selected = next((o for o in options if o["option_id"] == row[column]), None)
            if row[column] and selected is None:
                report["flags"].append(column + "_invalid_id")
            old_id = detail["original_annotation"].get(column)
            old = next((o for o in detail["old_candidate"][kind] if o["option_id"] == old_id), None)
            if row.get(column + "_mapping") == "changed_or_ambiguous":
                matches = [o for o in options if old and o.get("denomination_char_start") is not None
                           and old["char_start"] <= o["denomination_char_start"] < old["char_end"]]
                rounded = len(matches) == 1 and rounding_only(old.get("value"), matches[0].get("value"))
                report["flags"].append(column + ("_rounding_difference" if rounded else "_changed_reading"))
                report["option_diagnostics"].append(f"Old {column}: {old_id}, value={old.get('value') if old else None}")
                report["option_diagnostics"].extend(describe(o, candidate) for o in matches)
            if row["canonical_label"] == "PRICE" and not row[column]:
                local = [o for o in options if inside(o, quote)]
                if local:
                    report["flags"].append(column + "_null_with_options_in_quote")
                    report["option_diagnostics"].extend(describe(o, candidate) for o in local)
                else:
                    report["notes"].append(column + ": no option anchor in the quoted passage; null remains unchallenged mechanically")
            if selected:
                if selected.get("ownership_ambiguous"):
                    report["flags"].append(column + "_ambiguous_numeral")
                if quote and not inside(selected, quote):
                    report["flags"].append(column + "_outside_quote")
                    report["option_diagnostics"].append(describe(selected, candidate))
            for o in options:
                if o.get("value_char_start") is not None:
                    owned.append((o["value_char_start"], o["value_char_end"]))
                    tail = norm(candidate["source_block_text"][o["value_char_end"]:])
                    if re.match(r"^ετου[σς]\b", tail):
                        report["flags"].append("option_number_immediately_before_year_word")
                        report["option_diagnostics"].append(describe(o, candidate))
                if row["canonical_label"] == "NO_PRICE" and o.get("inference_rule"):
                    report["flags"].append("negative_window_has_inferred_option")
                    report["option_diagnostics"].append(describe(o, candidate))
            if row["canonical_label"] == "PRICE" and selected and selected.get("inference_rule"):
                report["notes"].append(column + ": inferred option selected; denomination must remain unstated")
        if len(owned) != len(set(owned)):
            report["flags"].append("reused_numeral_span")
        if refresh_display:
            row.update(display)

    compare_fields = ("canonical_label", "Human_Commodity", "Human_Qty", "Human_Unit",
                      "Human_Price", "Human_Currency", "rate_basis", "final_context_tags")
    for group in source_groups.values():
        if len(group) < 2:
            continue
        differences = [k for k in compare_fields if len({r.get(k, "") for r, _ in group}) > 1]
        if differences:
            message = "Same located quote: " + ", ".join(r["sample_id"] for r, _ in group) + "; differing fields: " + ", ".join(differences)
            for _, report in group:
                report["flags"].append("same_source_quote_different_annotations")
                report["notes"].append(message)

    output.mkdir(parents=True, exist_ok=True)
    if refresh_display and any(r["stale_display_fields"] for r in reports):
        backup = output / f"gold_review_before_display_refresh_{before_hash[:12]}.csv"
        if backup.exists() and backup.read_bytes() != review.read_bytes():
            raise ValueError("backup path already contains different data")
        backup.write_bytes(review.read_bytes())
        write_csv(review, rows)
    errors = validate_annotations(review, candidates_path)
    for error in errors:
        match = re.match(r"row (\d+):", error)
        if match and 0 <= int(match[1]) - 2 < len(reports):
            report = reports[int(match[1]) - 2]
            report["flags"].append("annotation_validation_error")
            report["notes"].append(error)
    for report in reports:
        report["flags"] = sorted(set(report["flags"]))
        report["option_diagnostics"] = list(dict.fromkeys(report["option_diagnostics"]))
        report["status"] = "needs_human_attention" if report["flags"] else "no_mechanical_anomaly"
        report["priority"] = ("reading_check" if set(report["flags"]) - {
            "window_text_changed", "negative_window_has_inferred_option"} else
            "context_screen" if report["flags"] else "none")
    flagged = [r for r in reports if r["flags"]]
    result = {"review_sha256_before": before_hash, "review_sha256_after": sha256_file(review),
              "evidence_sha256": sha256_file(evidence), "candidates_sha256": sha256_file(candidates_path),
              "rows": len(rows), "flagged_rows": len(flagged), "unflagged_rows": len(rows)-len(flagged),
              "stale_display_rows": sum(bool(r["stale_display_fields"]) for r in reports),
              "priority_counts": dict(sorted(Counter(r["priority"] for r in reports).items())),
              "display_refreshed": refresh_display, "validation_errors": errors,
              "flag_counts": dict(sorted(Counter(f for r in reports for f in r["flags"]).items())),
              "human_confirmation_written": False,
              "limitations": "Mechanical consistency only. Does not establish Greek interpretation, PRICE/NO_PRICE correctness, primary-pair choice or historical conversions.",
              "reports": reports}
    write_json(output / "mechanical_audit.json", result)
    write_csv(output / "mechanical_audit_rows.csv", [{
        "sample_id": r["sample_id"], "doc_id": r["doc_id"], "status": r["status"],
        "priority": r["priority"],
        "flags": ";".join(r["flags"]), "notes": " | ".join(r["notes"]),
        "options": " | ".join(r["option_diagnostics"]), "evidence_span": r["evidence_span"],
    } for r in reports])
    lines = ["# Gold mapping: mechanical review", "",
             f"Checked {len(rows)} historical windows; {len(flagged)} flagged, {len(rows)-len(flagged)} without a detected mechanical anomaly.",
             "No labels, selected IDs, human readings or review confirmations were changed. Unflagged does not mean human-verified.",
             "Inferred options in NO_PRICE windows are screening flags, not demonstrated false negatives. IDs below are NEW candidate IDs.", ""]
    lines += ["## Review order", "", "| Priority | Sample IDs |", "|---|---|"]
    for priority in ("reading_check", "context_screen"):
        lines.append("| " + priority + " | " + ", ".join(r["sample_id"] for r in flagged if r["priority"] == priority) + " |")
    lines += ["", "Context-screen rows have changed excerpt boundaries and/or inferred options in a NO_PRICE window; these are not demonstrated label errors.", ""]
    for r in sorted(flagged, key=lambda r: (r["priority"] != "reading_check", r["sample_id"])):
        lines += [f"## {r['sample_id']} — {r['doc_id']} ({r['canonical_label']})", "",
                  "Flags: " + "; ".join(r["flags"]), "",
                  "Recorded reading: " + json.dumps(r["human_reading"], ensure_ascii=False), "",
                  "> " + (r["evidence_span"] or "No primary-observation quote recorded for this window."), ""]
        if r["human_notes"] or r["context_note"]:
            lines += ["Existing annotation notes: " + " / ".join(
                x for x in (r["human_notes"], r["context_note"]) if x), ""]
        if r["canonical_label"] == "NO_PRICE":
            lines += ["Existing exclusion: " + r["exclusion_reason"], "",
                      "> " + r["candidate_excerpt"], ""]
        lines += ["- " + line for line in r["notes"] + r["option_diagnostics"]] + [""]
    (output / "mechanical_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("review", "evidence", "candidates", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--refresh-display", action="store_true")
    args = parser.parse_args()
    result = audit_review(args.review, args.evidence, args.candidates, args.output, args.refresh_display)
    print(json.dumps({k: v for k, v in result.items() if k != "reports"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

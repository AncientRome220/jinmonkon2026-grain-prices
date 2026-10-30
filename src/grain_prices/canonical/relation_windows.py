"""Prepare source-anchored review targets and search token radii without LLM calls."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from grain_prices.canonical.io import read_jsonl, write_jsonl, sha256_file
from grain_prices.canonical.provenance import candidate_fingerprint
from grain_prices.canonical.rebuild import read_csv, write_csv, write_json
from grain_prices.canonical.relations import (
    COORDINATES, GOLD_VERSION, OBJECTIVE_VERSION, target_id, validate_gold, validate_target,
)
from grain_prices.canonical.text import normalize_with_map
from grain_prices.legacy.stage2 import tokenize_with_positions

STEMS = {
    "wheat": ("πυρ", "σιτ"), "barley": ("κριθ",), "emmer": ("ολυρ",),
    "zea": ("ζε",), "flour": ("αλευρ",),
    "grain_unspecified": ("πυρ", "σιτ", "κριθ", "ολυρ", "ζε", "σταχυ"),
}


def identity(candidate: dict) -> tuple:
    return candidate["ddb_id"], candidate["source_block_id"], candidate["source_block_sha256"]


def grain_span(candidate: dict) -> list[int]:
    return [candidate["grain_surface_char_start_debug"], candidate["grain_surface_char_end_debug"]]


def occurrences(text: str, quote: str) -> list[list[int]]:
    mapped = normalize_with_map(text)
    needle = normalize_with_map(quote).text
    if not needle:
        return []
    result, start = [], 0
    while (pos := mapped.text.find(needle, start)) >= 0:
        result.append(list(mapped.original_span(pos, pos+len(needle))))
        start = pos+1
    return result


def matched_option(row: dict, detail: dict, kind: str, column: str) -> tuple[dict | None, str]:
    candidate = detail["new_candidate"]
    selected = row.get(column)
    if selected:
        matches = [o for o in candidate[kind] if o["option_id"] == selected]
        if len(matches) != 1:
            raise ValueError("review names an unoffered option")
        return matches[0], "current_selection"
    old_id = detail["original_annotation"].get(column)
    old = next((o for o in detail["old_candidate"][kind] if o["option_id"] == old_id), None)
    if not old:
        return None, "unselected"
    matches = [o for o in candidate[kind] if o.get("denomination_char_start") is not None
               and old["char_start"] <= o["denomination_char_start"] < old["char_end"]]
    if len(matches) == 1:
        return matches[0], "historical_position_only_not_confirmed"
    return None, "unresolved_historical_position"


def target_from_candidate(c: dict) -> dict:
    span = grain_span(c)
    return {
        "schema_version": GOLD_VERSION, "objective_version": OBJECTIVE_VERSION,
        "target_id": target_id(c["ddb_id"], c["source_block_id"], c["source_block_sha256"], span),
        "doc_id": c["ddb_id"], "source_block_id": c["source_block_id"],
        "source_block_sha256": c["source_block_sha256"], "source_block_text": c["source_block_text"],
        "coordinate_system": COORDINATES, "grain_span": span, "grain_form": c["grain_form"],
        **{k: c.get(k) for k in ("date_not_before", "date_not_after", "place")},
        "qty_options": deepcopy(c["qty_options"]), "money_options": deepcopy(c["money_options"]),
        "review_status": "", "reviewed_by": "", "review_note": "", "coverage": "unreviewed",
        "relations": [], "relation_proposals": [], "origin_sample_ids": [],
        "origin_annotations": [], "preparation_flags": [],
    }


def prepare_relations(review_path: Path, evidence_path: Path, candidates_path: Path, out: Path) -> dict:
    """Proposals only: no positive/negative target judgment is human-confirmed here."""
    if (out / "relation_gold_review.jsonl").exists():
        raise ValueError("relation review already exists; use a new directory to preserve human edits")
    review_hash = sha256_file(review_path)
    rows = read_csv(review_path)
    details_list = list(read_jsonl(evidence_path))
    details = {d["sample_id"]: d for d in details_list}
    if len(details) != len(details_list) or len(rows) != len(details) or {r["sample_id"] for r in rows} != set(details):
        raise ValueError("review sample identities differ")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate sample rows")
    wanted = {identity(d["new_candidate"]) for d in details_list if d["new_candidate"]}
    by_block = defaultdict(list)
    for c in read_jsonl(candidates_path):
        if identity(c) in wanted:
            if candidate_fingerprint(c) != c.get("candidate_fingerprint"):
                raise ValueError("candidate fingerprint invalid")
            by_block[identity(c)].append(c)
    targets, ledger = {}, []
    for row in rows:
        detail = details[row["sample_id"]]
        original = detail["new_candidate"]
        if original is None:
            ledger.append({"sample_id": row["sample_id"], "target_id": "", "resolution": "missing_candidate",
                           "historical_label": row["canonical_label"], "proposed_grain_spans": ""})
            continue
        if (row["window_id"] != original["window_id"]
                or row["candidate_fingerprint"] != original["candidate_fingerprint"]
                or original not in by_block[identity(original)]):
            raise ValueError("review/candidate binding differs")
        selected = {side: matched_option(row, detail, kind, col) for side, kind, col in (
            ("quantity", "qty_options", "quantity_option"), ("money", "money_options", "money_option"))}
        quotes = occurrences(original["source_block_text"], row.get("evidence_span", ""))
        anchors = [o["denomination_char_start"] if o.get("denomination_char_start") is not None
                   else o["char_start"] for o, _ in selected.values() if o]
        compatible_quotes = [q for q in quotes if all(q[0] <= a < q[1] for a in anchors)]
        if compatible_quotes:
            quotes = compatible_quotes
        compatible_grains = []
        for c in by_block[identity(original)]:
            normal = normalize_with_map(c["grain_form"]).text
            if any(normal.startswith(s) for s in STEMS.get(row.get("Human_Commodity", ""), ())):
                for q in quotes:
                    if q[0] <= grain_span(c)[0] < grain_span(c)[1] <= q[1]:
                        compatible_grains.append(c)
                        break
        unique_grains = {tuple(grain_span(c)): c for c in compatible_grains}
        positive = row["canonical_label"] == "PRICE"
        if positive and len(unique_grains) == 1:
            chosen = next(iter(unique_grains.values()))
            resolution = "unique_commodity_compatible_grain_in_quote"
        else:
            chosen = original
            resolution = ("historical_negative_needs_target_review" if not positive else
                          "ambiguous_grain_anchor" if unique_grains else "grain_reference_outside_quote")
        target = target_from_candidate(chosen)
        target = targets.setdefault(target["target_id"], target)
        target["origin_sample_ids"].append(row["sample_id"])
        target["origin_annotations"].append(deepcopy(row))
        target["preparation_flags"].append(resolution)
        ledger.append({
            "sample_id": row["sample_id"], "target_id": target["target_id"], "resolution": resolution,
            "historical_label": row["canonical_label"],
            "proposed_grain_spans": str(sorted(unique_grains)),
        })
        if not positive:
            continue
        # Build independent source-position proposals. Missing IDs are unresolved,
        # not silently interpreted as genuinely unstated quantities or money.
        sides = {}
        for side, (o, mapping) in selected.items():
            value_field, denom_field = (("Human_Qty", "Human_Unit") if side == "quantity"
                                         else ("Human_Price", "Human_Currency"))
            sides[side] = {
                "status": "present" if o else "unresolved",
                "components": [{"span": [o["char_start"], o["char_end"]],
                                "value": row.get(value_field) or None,
                                "denomination": row.get(denom_field) or None}] if o else [],
            }
            if mapping != "current_selection":
                target["preparation_flags"].append(side + ":" + mapping)
        quote = quotes[0] if len(quotes) == 1 else None
        relation = {"grain_span": target["grain_span"], "commodity": row.get("Human_Commodity", ""),
                    **sides, "rate_basis": row.get("rate_basis", "unknown"), "evidence_span": quote}
        proposal = {"origin_sample_id": row["sample_id"], "anchor_resolution": resolution,
                    "quote_occurrences": quotes, "candidate_grain_spans": [list(s) for s in sorted(unique_grains)],
                    "relation": relation}
        target["relation_proposals"].append(proposal)
        # The editable draft retains every proposal, including conflicts. Freeze
        # will require a reconciled exhaustive list, not accept these duplicates.
        target["relations"].append(deepcopy(relation))
    if sha256_file(review_path) != review_hash:
        raise ValueError("review changed during preparation; retry from a consistent snapshot")
    ordered = sorted(targets.values(), key=lambda t: (t["doc_id"], t["source_block_id"], t["grain_span"]))
    for target in ordered:
        validate_target(target)
        target["preparation_flags"] = sorted(set(target["preparation_flags"]))
        target["origin_sample_ids"].sort()
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "relation_gold_review.jsonl", ordered)
    write_csv(out / "historical_sample_ledger.csv", ledger)
    result = {
        "objective_version": OBJECTIVE_VERSION, "historical_sample_rows": len(rows),
        "target_review_rows": len(ordered), "historical_positive_rows": sum(r["canonical_label"] == "PRICE" for r in rows),
        "relation_proposals": sum(len(t["relation_proposals"]) for t in ordered),
        "resolution_counts": dict(sorted(Counter(r["resolution"] for r in ledger).items())),
        "confirmed_targets": 0, "status": "awaiting_target_relation_review",
        "sample_sources": {sample: {k: t[k] for k in
                            ("doc_id", "source_block_id", "source_block_sha256")}
                           for t in ordered for sample in t["origin_sample_ids"]},
        "inputs": {str(p): sha256_file(p) for p in (review_path, evidence_path, candidates_path)},
        "artifacts": {str(p): sha256_file(p) for p in (out / "relation_gold_review.jsonl", out / "historical_sample_ledger.csv")},
    }
    write_json(out / "manifest_prepare_relations.json", result)
    return result


def token_window(target: dict, radius: int) -> tuple[list[int], list[dict]]:
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    mapped = normalize_with_map(target["source_block_text"])
    toks = tokenize_with_positions(mapped.text)
    grain_start = target["grain_span"][0]
    normalized_start = next(i for i, source in enumerate(mapped.original_indexes) if source >= grain_start)
    index = toks.tok_at_char(normalized_start)
    lo, hi = max(0, index-radius), min(len(toks.tokens), index+radius+1)
    bounds = list(mapped.original_span(toks.starts[lo], toks.starts[hi-1]+len(toks.tokens[hi-1])))
    inventory = [{"span": list(mapped.original_span(toks.starts[i], toks.starts[i]+len(toks.tokens[i]))),
                  "text": target["source_block_text"][slice(*mapped.original_span(toks.starts[i], toks.starts[i]+len(toks.tokens[i])))]}
                 for i in range(lo, hi)]
    return bounds, inventory


def within(span, bounds: list[int]) -> bool:
    return bool(span and bounds[0] <= span[0] < span[1] <= bounds[1])


def window_candidate(target: dict, radius: int) -> dict:
    bounds, inventory = token_window(target, radius)
    candidate = {k: target[k] for k in ("target_id", "doc_id", "source_block_id",
        "source_block_sha256", "source_block_text", "coordinate_system", "grain_span", "grain_form")}
    candidate.update(objective_version=OBJECTIVE_VERSION, window_id=target["target_id"],
                     ddb_id=target["doc_id"], radius_tokens=radius,
                     evidence_char_start=bounds[0], evidence_char_end=bounds[1],
                     evidence_text=target["source_block_text"][slice(*bounds)], token_inventory=inventory)
    candidate.update({k: target.get(k) for k in ("date_not_before", "date_not_after", "place")})
    for kind in ("qty_options", "money_options"):
        # Entire source expression, not just denomination anchor, must be visible.
        candidate[kind] = [deepcopy(o) for o in target[kind] if within([o["char_start"], o["char_end"]], bounds)]
    candidate["candidate_fingerprint"] = candidate_fingerprint(candidate)
    return candidate


def search_windows(gold_review: Path, out: Path, radii: list[int]) -> dict:
    targets = list(read_jsonl(gold_review))
    for t in targets:
        validate_target(t)
    radii = sorted(set(radii))
    if not radii or min(radii) < 1 or max(radii) > 80:
        raise ValueError("provide radii from 1 to 80; saved options came from radius 80")
    all_proposals = [(t, p) for t in targets for p in t.get("relation_proposals", [])]
    eligible = [(t, p) for t, p in all_proposals if
                p["anchor_resolution"] == "unique_commodity_compatible_grain_in_quote"
                and p["relation"]["evidence_span"] is not None]
    details, summaries = [], []
    for t, p in all_proposals:
        r = p["relation"]
        spans = [c["span"] for s in ("quantity", "money") for c in r[s]["components"]]
        complete = all(r[s]["status"] != "unresolved" for s in ("quantity", "money"))
        tested = [(radius, token_window(t, radius)[0]) for radius in range(1, 81)]
        details.append({
            "sample_id": p["origin_sample_id"], "target_id": t["target_id"], "doc_id": t["doc_id"],
            "anchor_resolution": p["anchor_resolution"], "quantity_and_money_located": complete,
            "evidence_span": r["evidence_span"], "grain_span": t["grain_span"],
            "component_spans": spans,
            "minimum_radius_for_located_components": next((radius for radius, bounds in tested
                if spans and all(within(s, bounds) for s in spans)), None),
            "minimum_radius_for_quote_and_components": next((radius for radius, bounds in tested
                if within(r["evidence_span"], bounds) and all(within(s, bounds) for s in spans)), None),
        })
    eligible_ids = {p["origin_sample_id"] for _, p in eligible}
    out.mkdir(parents=True, exist_ok=True)
    for radius in radii:
        variants = [window_candidate(t, radius) for t in targets]
        path = out / f"relation_candidates_r{radius}.jsonl"
        write_jsonl(path, variants)
        by_id = {v["target_id"]: v for v in variants}
        quote_retained = component_retained = both_located = both_retained = 0
        for t, p in eligible:
            c = by_id[t["target_id"]]
            bounds = [c["evidence_char_start"], c["evidence_char_end"]]
            r = p["relation"]
            spans = [part["span"] for s in ("quantity", "money") for part in r[s]["components"]]
            component_retained += bool(spans) and all(within(s, bounds) for s in spans)
            quote_retained += within(r["evidence_span"], bounds) and all(within(s, bounds) for s in spans)
            complete = all(r[s]["status"] != "unresolved" for s in ("quantity", "money"))
            both_located += complete
            both_retained += complete and all(within(s, bounds) for s in spans)
        summaries.append({
            "radius_tokens_each_side": radius, "targets_retained": len(targets),
            "option_occurrences": sum(len(c["qty_options"])+len(c["money_options"]) for c in variants),
            "targets_without_money_options_retained": sum(not c["money_options"] for c in variants),
            "eligible_primary_quote_and_components_retained": quote_retained,
            "eligible_any_located_components_retained": component_retained,
            "eligible_both_sides_located": both_located, "eligible_both_sides_retained": both_retained,
            "input_sha256": sha256_file(path), "input_path": str(path),
            "confirmed_gold_relation_retention": [
                {"target_id": t['target_id'], "relation_index": i,
                 "evidence_retained": within(r['evidence_span'], [c['evidence_char_start'], c['evidence_char_end']]),
                 "components_retained": all(within(part['span'], [c['evidence_char_start'], c['evidence_char_end']])
                                            for side in ('quantity', 'money') for part in r[side]['components'])}
                for t, c in zip(targets, variants) if t.get('review_status') == 'confirmed'
                for i, r in enumerate(t['relations'])],
        })
    result = {
        "objective_version": OBJECTIVE_VERSION, "status": "development_geometry_not_accuracy",
        "target_review_sha256": sha256_file(gold_review), "target_review_rows": len(targets),
        "geometry_radii_tested": list(range(1, 81)),
        "historical_positive_proposals": len(all_proposals),
        "eligible_unique_anchor_and_quote_proposals": len(eligible),
        "ineligible_sample_ids": [d["sample_id"] for d in details if d["sample_id"] not in eligible_ids],
        "confirmed_targets": sum(t["review_status"] == "confirmed" for t in targets),
        "summaries": summaries, "proposal_details": details,
        "limitations": [
            "Gold mappings are proposals, not new human judgments.",
            "Unique commodity-compatible grain in a quote is a mechanical anchor proposal, not verified linkage.",
            "Missing or ambiguous mappings remain in the historical ledger and are reported outside conditional coverage.",
            "Whole numeric expressions must fit; offered IDs may be empty without dropping a target.",
            "No provider calls or historical conversions; no model accuracy or optimal radius is claimed.",
            "Saved option inventories originated within radius 80 of the proposed target; no larger radius is supported.",
            "Reanchoring uses historical positive labels and is development-set selection, not a new random or blinded sample.",
        ],
    }
    write_json(out / "manifest_window_search.json", result)
    return result


def freeze_relations(review_path: Path, output: Path) -> dict:
    rows = list(read_jsonl(review_path))
    validate_gold(rows)
    prepare_path = review_path.parent / "manifest_prepare_relations.json"
    prepared = json.loads(prepare_path.read_text(encoding="utf-8"))
    origins = [sample for r in rows for sample in r["origin_sample_ids"]]
    if (len(origins) != prepared["historical_sample_rows"] or len(origins) != len(set(origins))
            or set(origins) != set(prepared["sample_sources"])):
        raise ValueError("all historical sample rows must remain accounted for exactly once")
    for r in rows:
        for sample in r["origin_sample_ids"]:
            if any(r[k] != v for k, v in prepared["sample_sources"][sample].items()):
                raise ValueError("review changed the historical source identity")
    if output.exists():
        raise ValueError("refusing to replace frozen relation gold")
    write_jsonl(output, rows)
    result = {"objective_version": OBJECTIVE_VERSION, "confirmed_targets": len(rows),
              "relations": sum(len(r["relations"]) for r in rows), "review_sha256": sha256_file(review_path),
              "historical_sample_rows": len(origins), "prepare_manifest_sha256": sha256_file(prepare_path),
              "frozen_gold_sha256": sha256_file(output), "status": "reviewed_awaiting_run_authorization"}
    write_json(output.with_suffix(".manifest.json"), result)
    return result

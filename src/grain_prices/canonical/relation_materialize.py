"""Materialize relation judgments without converting compound/unknown amounts."""
from collections import defaultdict, Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from grain_prices.canonical.io import read_jsonl, write_jsonl, sha256_file
from grain_prices.canonical.provenance import validate_candidate_binding
from grain_prices.canonical.relations import PREDICTION_VERSION, linkage_key, reading_key
from grain_prices.canonical.rebuild import write_json
from grain_prices.llm.relation_runner import validate_input, validate_answer


def relation_source_key(row):
    r = row["relation"]
    # Grain references may repeat for one amount: group on actual amount spans.
    parts = linkage_key(r)[1:]
    missing = any(r[s]["status"] == "unstated" for s in ("quantity", "money"))
    return (row["provider"], row["model_id"], row["repeat_index"], row["doc_id"],
            row["source_block_id"], row["source_block_sha256"], row["coordinate_system"],
            parts, tuple(r["evidence_span"]) if missing else None)


def group_relations(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[relation_source_key(r)].append(r)
    output = []
    for key, members in groups.items():
        meanings = {repr((reading_key(r["relation"]), r["relation"]["rate_basis"],
                          sorted(r["context_tags"]), r["linkage_strength"])) for r in members}
        conflict = len(meanings) > 1
        group = dict(members[0])
        group.update(observation_group_id=hashlib.sha256(repr(key).encode()).hexdigest(),
                     grouping_status="interpretation_conflict" if conflict else "agreed",
                     source_observations=members, contributing_target_ids=sorted({r["target_id"] for r in members}),
                     contributing_observation_count=len(members))
        if conflict:
            for field in ("relation", "context_tags", "linkage_strength", "price_per_artaba_drachma"):
                group[field] = None
            group["analysis_eligible"] = False
        output.append(group)
    return output


def materialize_relations(candidates_path: Path, predictions_path: Path, output: Path, repeat_index=0):
    inputs = list(read_jsonl(candidates_path))
    candidates = {c["target_id"]: c for c in inputs}
    if len(inputs) != len(candidates):
        raise ValueError("duplicate candidates")
    for c in inputs:
        validate_input(c)
    preds = [p for p in read_jsonl(predictions_path) if p.get("repeat_index", 0) == repeat_index]
    if len({p["target_id"] for p in preds}) != len(preds) or len({(p["provider"], p["model_id"]) for p in preds}) > 1:
        raise ValueError("duplicate targets or mixed models")
    rows, failures = [], []
    for p in preds:
        c = candidates.get(p["target_id"])
        if not c or p.get("schema_version") != PREDICTION_VERSION:
            raise ValueError("wrong prediction/input contract")
        validate_candidate_binding(p, c)
        answer = p.get("answer")
        if p.get("error") or p.get("validation_problems") or not isinstance(answer, dict):
            failures.append(p["target_id"])
            continue
        problems = validate_answer({**answer, "contexts": p.get("contexts", [])}, c)
        if problems:
            failures.append(p["target_id"])
            continue
        contexts = {x["relation_index"]: x for x in p["contexts"]}
        for index, relation in enumerate(answer["relations"]):
            context = contexts[index]
            row = {k: c[k] for k in ("target_id", "doc_id", "source_block_id", "source_block_sha256", "coordinate_system")}
            row.update({k: p[k] for k in ("provider", "model_id", "repeat_index", "candidate_fingerprint")})
            row.update(observation_schema_version="target-relation-observation-v1", observation_index=index,
                       relation=relation, context_tags=context["context_tags"], linkage_strength=context["linkage_strength"],
                       human_verified=False, normalization_status="unapproved_or_incomplete",
                       price_per_artaba_drachma=None, analysis_eligible=False,
                       source_components={s: [{**x, "written_text": c["source_block_text"][slice(*x["span"])]}
                                             for x in relation[s]["components"]] for s in ("quantity", "money")},
                       **{k: c.get(k) for k in ("date_not_before", "date_not_after", "place")})
            q, m = relation["quantity"]["components"], relation["money"]["components"]
            # Only the already-supported direct denomination; never convert or
            # collapse a compound sum. Eligibility still requires human/context review.
            if len(q) == len(m) == 1 and q[0]["denomination"] == "artaba" and m[0]["denomination"] == "drachma" and m[0]["value"]:
                amount = Fraction(m[0]["value"])
                if relation["rate_basis"] == "stated_rate":
                    row["price_per_artaba_drachma"] = float(amount)
                elif relation["rate_basis"] == "derived_from_total" and q[0]["value"] and Fraction(q[0]["value"]) > 0:
                    row["price_per_artaba_drachma"] = float(amount/Fraction(q[0]["value"]))
                if row["price_per_artaba_drachma"] is not None:
                    row["normalization_status"] = "direct_rate_awaiting_historical_review"
            rows.append(row)
    dedup = group_relations(rows)
    write_jsonl(output, rows)
    write_jsonl(output.with_name(output.stem+"_deduplicated.jsonl"), dedup)
    result = {"emitted_relations": len(rows), "deduplicated_source_groups": len(dedup),
              "group_statuses": dict(Counter(r["grouping_status"] for r in dedup)),
              "interpretation_conflicts": sum(r['grouping_status'] == 'interpretation_conflict' for r in dedup),
              "unresolved_locations": 0,
              "observations_eligible_for_historical_analysis": sum(r["analysis_eligible"] for r in dedup),
              "invalid_or_errored_targets": failures,
              "missing_prediction_targets": sorted(set(candidates)-{p["target_id"] for p in preds}),
              "inputs": {str(x): sha256_file(x) for x in (candidates_path, predictions_path)},
              "note": "Predicted relations are not human-verified historical records; direct rates remain provisional."}
    write_json(output.with_suffix(".manifest.json"), result)
    return result

"""Target-grain relation contracts and order-independent evaluation (#048).

Historical window-level files are intentionally not accepted by this evaluator.
Source linkage, numeric readings and price basis are separate scoring axes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from grain_prices.canonical.contracts import RATE_BASIS_VALUES
from grain_prices.canonical.io import read_jsonl, sha256_file
from grain_prices.canonical.options import currency_name
from grain_prices.canonical.provenance import validate_candidate_binding
from grain_prices.canonical.rebuild import write_json
from grain_prices.pipeline_shared import canon_unit

OBJECTIVE_VERSION = "target-grain-relation-v1"
GOLD_VERSION = "target-grain-gold-v1"
PREDICTION_VERSION = "target-grain-prediction-v1"
COORDINATES = "expanded-block-surface-v1"

COMPONENT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["span", "value", "denomination"],
    "properties": {
        "span": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
        "value": {"type": ["string", "null"]},
        "denomination": {"type": ["string", "null"]},
    },
}
SIDE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["status", "components"],
    "properties": {
        "status": {"type": "string", "enum": ["present", "unreadable", "unstated"]},
        "components": {"type": "array", "items": COMPONENT_SCHEMA},
    },
}
RELATION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["grain_span", "commodity", "quantity", "money", "rate_basis", "evidence_span"],
    "properties": {
        "grain_span": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
        "commodity": {"type": "string"},
        "quantity": SIDE_SCHEMA, "money": SIDE_SCHEMA,
        "rate_basis": {"type": "string", "enum": sorted(RATE_BASIS_VALUES)},
        "evidence_span": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
    },
}
PREDICTION_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["relations"],
    "properties": {"relations": {"type": "array", "items": RELATION_SCHEMA}},
}


def target_id(doc_id: str, block_id: str, block_sha256: str, grain_span: list[int]) -> str:
    key = [doc_id, block_id, block_sha256, COORDINATES, grain_span]
    return "grain-" + hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()[:24]


def valid_span(value, length: int) -> bool:
    return (isinstance(value, list) and len(value) == 2
            and all(type(v) is int for v in value) and 0 <= value[0] < value[1] <= length)


def validate_target(target: dict) -> None:
    text = target.get("source_block_text", "")
    if target.get("coordinate_system") != COORDINATES:
        raise ValueError("unsupported relation coordinate system")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != target.get("source_block_sha256"):
        raise ValueError("source block hash mismatch")
    if not valid_span(target.get("grain_span"), len(text)):
        raise ValueError("invalid target grain span")
    expected = target_id(target["doc_id"], target["source_block_id"],
                         target["source_block_sha256"], target["grain_span"])
    if target.get("target_id") != expected:
        raise ValueError("target identity mismatch")


def validate_relation(relation: dict, target: dict, excerpt: list[int] | None = None) -> list[str]:
    problems = []
    length = len(target["source_block_text"])
    if not isinstance(relation, dict):
        return ["relation must be an object"]
    required = set(RELATION_SCHEMA["required"])
    if set(relation) != required:
        problems.append("relation fields differ from contract")
    if relation.get("grain_span") != target["grain_span"]:
        problems.append("relation belongs to another grain mention")
    if not isinstance(relation.get("commodity"), str) or not relation.get("commodity", "").strip():
        problems.append("commodity is required")
    if relation.get("rate_basis") not in RATE_BASIS_VALUES:
        problems.append("invalid rate basis")
    evidence = relation.get("evidence_span")
    if not valid_span(evidence, length):
        problems.append("invalid relation evidence span")
    for side in ("quantity", "money"):
        obj = relation.get(side)
        if not isinstance(obj, dict) or set(obj) != {"status", "components"}:
            problems.append(side + " fields differ from contract")
            continue
        status, components = obj["status"], obj["components"]
        if status not in {"present", "unreadable", "unstated"} or not isinstance(components, list):
            problems.append(side + " status/components invalid")
            continue
        if (status == "unstated") != (not components):
            problems.append(side + " unstated must have no components; other statuses need evidence")
        spans = []
        for c in components:
            if not isinstance(c, dict) or set(c) != {"span", "value", "denomination"}:
                problems.append(side + " component fields differ from contract")
                continue
            span = c["span"]
            if not valid_span(span, length):
                problems.append(side + " component span invalid")
            else:
                spans.append(tuple(span))
                if valid_span(evidence, length) and not (evidence[0] <= span[0] < span[1] <= evidence[1]):
                    problems.append(side + " component outside relation evidence")
                if excerpt and not (excerpt[0] <= span[0] < span[1] <= excerpt[1]):
                    problems.append(side + " component outside model input")
            if c["value"] is not None:
                try:
                    if not isinstance(c["value"], str):
                        raise ValueError()
                    Fraction(c["value"])
                except (ValueError, ZeroDivisionError):
                    problems.append(side + " value must be a numeric string or null")
            if c["denomination"] is not None and (not isinstance(c["denomination"], str) or not c["denomination"].strip()):
                problems.append(side + " denomination must be a nonempty string or null")
        if len(set(spans)) != len(spans):
            problems.append(side + " repeated component span")
    if excerpt and valid_span(evidence, length) and not (excerpt[0] <= evidence[0] < evidence[1] <= excerpt[1]):
        problems.append("relation evidence outside model input")
    return problems


def linkage_key(relation: dict) -> tuple:
    """Numerical reading is deliberately absent; an unstated side is not a wildcard."""
    return (tuple(relation["grain_span"]),
            *[("unstated" if relation[s]["status"] == "unstated" else "located",
               tuple(sorted(tuple(c["span"]) for c in relation[s]["components"])))
              for s in ("quantity", "money")])


def reading_key(relation: dict) -> tuple:
    values = [relation["commodity"]]
    for side in ("quantity", "money"):
        canon = canon_unit if side == "quantity" else currency_name
        components = []
        for c in relation[side]["components"]:
            value = str(Fraction(c["value"])) if c["value"] is not None else None
            denomination = canon(c["denomination"]) if c["denomination"] else None
            components.append((tuple(c["span"]), value, denomination))
        values.append((relation[side]["status"], tuple(sorted(components))))
    return tuple(values)


def prf(tp: int, fp: int, fn: int) -> dict:
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp+fp) if tp+fp else 0.0,
            "recall": tp / (tp+fn) if tp+fn else 0.0,
            "f1": 2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else 0.0}


def validate_gold(rows: list[dict]) -> None:
    if any(r.get("provisional_reference") for r in rows):
        raise ValueError("provisional references cannot be used as formal gold")
    if len({r["target_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate gold targets; retain origin sample IDs within the target")
    if not rows:
        raise ValueError("empty relation gold")
    for r in rows:
        validate_target(r)
        if r.get("schema_version") != GOLD_VERSION:
            raise ValueError("historical window labels cannot be silently upgraded to relation gold")
        if r.get("review_status") != "confirmed" or not str(r.get("reviewed_by", "")).strip():
            raise ValueError("relation gold requires human confirmation")
        if r.get("coverage") != "exhaustive":
            raise ValueError("relation precision requires exhaustive target annotations")
        keys = []
        for relation in r["relations"]:
            errors = validate_relation(relation, r)
            if errors:
                raise ValueError("; ".join(errors))
            keys.append(linkage_key(relation))
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate gold relations for one target")


def evaluate_relations(predictions_path: Path, gold_path: Path, candidates_path: Path,
                       output: Path, repeat_index: int = 0) -> dict:
    gold = list(read_jsonl(gold_path))
    validate_gold(gold)
    return _evaluate_relation_rows(predictions_path, gold, candidates_path, output,
                                   repeat_index, gold_path)


def _evaluate_relation_rows(predictions_path: Path, gold: list[dict], candidates_path: Path,
                            output: Path, repeat_index: int, gold_path: Path) -> dict:
    """Shared arithmetic only; callers own formal/provisional validation."""
    by_gold = {r["target_id"]: r for r in gold}
    inputs = list(read_jsonl(candidates_path))
    candidates = {r["target_id"]: r for r in inputs}
    if len(candidates) != len(inputs) or set(candidates)-set(by_gold):
        raise ValueError("duplicate or out-of-sample relation inputs")
    for c in inputs:
        validate_target(c)
        if c.get("objective_version") != OBJECTIVE_VERSION:
            raise ValueError("wrong candidate evaluation objective")
        if c.get("candidate_fingerprint") is None:
            raise ValueError("relation inputs require fingerprints")
        validate_candidate_binding(c, c)
        g = by_gold[c["target_id"]]
        if any(c[k] != g[k] for k in ("doc_id", "source_block_id", "source_block_sha256", "grain_span")):
            raise ValueError("gold/input source identity differs")
    preds = [p for p in read_jsonl(predictions_path) if p.get("repeat_index", 0) == repeat_index] if predictions_path.exists() else []
    if len({p["target_id"] for p in preds}) != len(preds):
        raise ValueError("duplicate predictions in repetition")
    if len({(p.get("provider"), p.get("model_id")) for p in preds}) > 1:
        raise ValueError("do not combine model runs")
    by_pred = {}
    invalid, failures = [], []
    for p in preds:
        c = candidates.get(p["target_id"])
        if c is None:
            raise ValueError("prediction target absent from inputs")
        if p.get("schema_version") != PREDICTION_VERSION:
            raise ValueError("wrong relation prediction contract")
        validate_candidate_binding(p, c)
        if p.get("error"):
            failures.append(p["target_id"])
            continue
        answer = p.get("answer")
        if not isinstance(answer, dict) or set(answer) != {"relations"} or not isinstance(answer["relations"], list):
            invalid.append(p["target_id"])
            continue
        problems = [problem for r in answer["relations"] for problem in
                    validate_relation(r, c, [c["evidence_char_start"], c["evidence_char_end"]])]
        if p.get("validation_problems") or problems:
            invalid.append(p["target_id"])
            continue
        by_pred[p["target_id"]] = answer["relations"]
    tp = fp = fn = readings = basis = complete = exact_targets = expected_count = emitted_count = 0
    detection = Counter()
    diagnostics = []
    for target in gold:
        expected, emitted = target["relations"], by_pred.get(target["target_id"], [])
        expected_count += len(expected)
        emitted_count += len(emitted)
        expected_keys = {linkage_key(r): r for r in expected}
        emitted_groups = defaultdict(list)
        for r in emitted:
            emitted_groups[linkage_key(r)].append(r)
        matched = expected_keys.keys() & emitted_groups.keys()
        c = candidates.get(target['target_id'])
        bounds = [c['evidence_char_start'], c['evidence_char_end']] if c else None
        cropped = [i for i, r in enumerate(expected) if bounds is None or not
                   (bounds[0] <= r['evidence_span'][0] < r['evidence_span'][1] <= bounds[1])]
        diagnostics.append({'target_id': target['target_id'], 'origin_sample_ids': target.get('origin_sample_ids', []),
                            'valid_prediction': target['target_id'] in by_pred,
                            'gold_relations': len(expected), 'emitted_relations': len(emitted),
                            'relationship_tp': len(matched), 'relationship_fp': len(emitted)-len(matched),
                            'relationship_fn': len(expected)-len(matched),
                            'gold_evidence_outside_excerpt_relation_indexes': cropped,
                            'unmatched_gold': [r for k, r in expected_keys.items() if k not in matched],
                            'unmatched_predictions': [r for r in emitted if linkage_key(r) not in matched]})
        tp += len(matched)
        fp += len(emitted)-len(matched)
        fn += len(expected)-len(matched)
        exact_targets += (target["target_id"] in by_pred
                          and Counter(expected_keys.keys()) == Counter(linkage_key(r) for r in emitted))
        detection["tp" if expected and emitted else "fn" if expected else "fp" if emitted else "tn"] += 1
        for key in matched:
            wanted = expected_keys[key]
            matches = emitted_groups[key]
            read_ok = any(reading_key(r) == reading_key(wanted) for r in matches)
            basis_ok = any(r["rate_basis"] == wanted["rate_basis"] for r in matches)
            readings += read_ok
            basis += basis_ok
            complete += any(reading_key(r) == reading_key(wanted) and r["rate_basis"] == wanted["rate_basis"] for r in matches)
    missing = sorted(set(by_gold)-{p["target_id"] for p in preds})
    result = {
        "objective_version": OBJECTIVE_VERSION, "repeat_index": repeat_index,
        "gold_targets": len(gold), "gold_relations": expected_count, "emitted_relations": emitted_count,
        "relationship": prf(tp, fp, fn),
        "correct_readings_on_matched_relationships": readings,
        "correct_rate_basis_on_matched_relationships": basis,
        "conditional_reading_and_basis_denominator": tp,
        "complete_relation": prf(complete, emitted_count-complete, expected_count-complete),
        "exact_target_relation_sets": exact_targets,
        "target_has_valuation": {**dict(detection), **prf(detection["tp"], detection["fp"], detection["fn"])},
        "missing_prediction_targets": missing, "errored_targets": failures, "invalid_targets": invalid,
        "targets_without_candidate": sorted(set(by_gold)-set(candidates)),
        "target_diagnostics": diagnostics,
        "inputs": {str(p): sha256_file(p) if p.exists() else None for p in (gold_path, predictions_path, candidates_path)},
        "note": "All confirmed targets and relations remain in denominators. Historical window-level scores are separate.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result

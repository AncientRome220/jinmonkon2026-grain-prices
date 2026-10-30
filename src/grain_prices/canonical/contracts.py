from __future__ import annotations

from typing import Any

from grain_prices.pipeline_shared import normalize_for_search

LINKAGE_VALUES = {"explicit", "proximate", "weak"}
CONTEXT_VALUES = {
    "sale", "valuation", "tax_commutation", "requisition", "loan",
    "penalty", "rate_other", "unknown",
}
COMMODITY_CLASSES = {"raw_grain", "processed", "other"}
CONFIDENCE_VALUES = {"high", "medium", "low"}
RATE_BASIS_VALUES = {"stated_rate", "derived_from_total", "unknown"}

CANONICAL_PREDICTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["any_price", "observations"],
    "properties": {
        "any_price": {"type": "boolean"},
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "quantity_option", "money_option", "commodity",
                    "commodity_class", "linkage_strength", "context_tags",
                    "rate_basis", "evidence_span", "confidence", "reasoning",
                ],
                "properties": {
                    "quantity_option": {"type": ["string", "null"]},
                    # Nullable for the same reason quantity is: an accounting
                    # hand omits the word the option scanner keys on, so a window
                    # can state a price that no offered amount expresses (issue
                    # #034). Without null the model's only ways out were to name
                    # an amount it judged wrong or to deny a price it had read —
                    # the bind the gold file was released from on 2026-08-19.
                    "money_option": {"type": ["string", "null"]},
                    "commodity": {"type": "string"},
                    "commodity_class": {"enum": sorted(COMMODITY_CLASSES)},
                    "linkage_strength": {"enum": sorted(LINKAGE_VALUES)},
                    "context_tags": {
                        "type": "array", "minItems": 1, "uniqueItems": True,
                        "items": {"enum": sorted(CONTEXT_VALUES)},
                    },
                    "rate_basis": {"enum": sorted(RATE_BASIS_VALUES)},
                    "evidence_span": {"type": "string", "minLength": 1},
                    "confidence": {"enum": sorted(CONFIDENCE_VALUES)},
                    "reasoning": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}


def validate_prediction(answer: Any, candidate: dict) -> list[str]:
    problems: list[str] = []
    if not isinstance(answer, dict):
        return ["answer must be an object"]
    unknown_top = set(answer) - {"any_price", "observations"}
    if unknown_top:
        problems.append(f"unknown top-level fields: {','.join(sorted(unknown_top))}")
    if not isinstance(answer.get("any_price"), bool):
        problems.append("any_price must be boolean")
    observations = answer.get("observations")
    if not isinstance(observations, list):
        problems.append("observations must be an array")
        return problems
    if bool(answer.get("any_price")) != bool(observations):
        problems.append("any_price must equal bool(observations)")

    qids = {o["option_id"] for o in candidate.get("qty_options", [])}
    mids = {o["option_id"] for o in candidate.get("money_options", [])}
    evidence = normalize_for_search(candidate.get("evidence_text", ""))
    for index, obs in enumerate(observations):
        prefix = f"observations[{index}]"
        if not isinstance(obs, dict):
            problems.append(f"{prefix} must be an object")
            continue
        allowed = {
            "quantity_option", "money_option", "commodity", "commodity_class",
            "linkage_strength", "context_tags", "rate_basis", "evidence_span",
            "confidence", "reasoning",
        }
        missing = allowed - set(obs)
        unknown = set(obs) - allowed
        if missing:
            problems.append(f"{prefix} missing fields: {','.join(sorted(missing))}")
        if unknown:
            problems.append(f"{prefix} unknown fields: {','.join(sorted(unknown))}")
        if obs.get("quantity_option") is not None and obs.get("quantity_option") not in qids:
            problems.append(f"{prefix}.quantity_option was not offered")
        if obs.get("money_option") is not None and obs.get("money_option") not in mids:
            problems.append(f"{prefix}.money_option was not offered")
        if obs.get("linkage_strength") not in LINKAGE_VALUES:
            problems.append(f"{prefix}.linkage_strength is invalid")
        tags = obs.get("context_tags")
        if not isinstance(tags, list) or not tags or any(t not in CONTEXT_VALUES for t in tags):
            problems.append(f"{prefix}.context_tags is invalid")
        if obs.get("commodity_class") not in COMMODITY_CLASSES:
            problems.append(f"{prefix}.commodity_class is invalid")
        if not isinstance(obs.get("commodity"), str) or not obs.get("commodity", "").strip():
            problems.append(f"{prefix}.commodity is required")
        if obs.get("confidence") not in CONFIDENCE_VALUES:
            problems.append(f"{prefix}.confidence is invalid")
        if obs.get("rate_basis") not in RATE_BASIS_VALUES:
            problems.append(f"{prefix}.rate_basis is invalid")
        span = normalize_for_search(obs.get("evidence_span", ""))
        if not span or span not in evidence:
            problems.append(f"{prefix}.evidence_span is absent from evidence_text")
        if not isinstance(obs.get("reasoning"), str) or not obs.get("reasoning", "").strip():
            problems.append(f"{prefix}.reasoning is required")
    return problems


def option_by_id(candidate: dict, kind: str, option_id: str | None) -> dict | None:
    if option_id is None:
        return None
    key = "qty_options" if kind == "quantity" else "money_options"
    return next((o for o in candidate.get(key, []) if o.get("option_id") == option_id), None)

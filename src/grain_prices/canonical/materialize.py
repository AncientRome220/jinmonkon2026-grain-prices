from __future__ import annotations

import csv
from pathlib import Path

from grain_prices.canonical import DEFINITION_VERSION
from grain_prices.canonical import OBSERVATION_SCHEMA_VERSION
from grain_prices.canonical.options import currency_name
from grain_prices.canonical.provenance import validate_candidate_binding
from grain_prices.canonical.observations import deduplicate_observations, locate_evidence
from grain_prices.canonical.annotation import POSITIVE_LABELS, validate_annotations
from grain_prices.canonical.contracts import option_by_id, validate_prediction
from grain_prices.canonical.io import read_jsonl, write_jsonl
from grain_prices.canonical.relation import relation_linkage
from grain_prices.pipeline_shared import canon_currency, canon_grain, canon_unit, to_float


# Written into `context_note` by the annotator when the option's own figure is
# known to be wrong at source. Machine-readable on purpose: the surrounding note
# says which figure and why, in prose, and `materialize` must not try to read
# that. It only needs to know to keep the row out of the analysable series.
EXCLUDE_MARK = "exclude_from_series="


def _json_number(value):
    number = to_float(value)
    return number if number == number else None


def commodity_class(commodity: str) -> str:
    value = (commodity or "").lower()
    if value == "flour" or "flour" in value:
        return "processed"
    if value in {"wheat", "barley", "emmer", "zea", "grain_unspecified"}:
        return "raw_grain"
    return "other"


def _char_span(option: dict | None) -> list[int] | None:
    if not option or option.get("char_start") is None or option.get("char_end") is None:
        return None
    return [option["char_start"], option["char_end"]]


def _final_row(candidate: dict, qty: dict | None, money: dict | None, commodity: str,
               method: str, observation_index: int, context_tags: list[str],
               rate_basis: str, evidence_span: str, confidence: str,
               reasoning: str, definition_version: str, unit_read: str = "",
               note: str = "") -> dict:
    # `money` is None when the window states a price the scanner could not turn
    # into an option — an account writes the sum as a bare numeral and the
    # edition does not restore the currency word (issue #034). The observation is
    # kept, because §5 keeps incomplete observations with the rate left
    # uncomputable. Linkage, though, is a machine-verifiable grade of how tightly
    # the money sits to the grain, and with no money position there is nothing to
    # grade: it takes the weakest value rather than a fabricated token index,
    # which also keeps the row out of series_strict and series_broad.
    if money is None:
        linkage, relation_cues = "weak", []
    else:
        linkage, relation_cues = relation_linkage(
            grain_token=int(candidate.get("grain_token_index", 0)),
            money_token=int(money.get("token_index", 0)),
            quantity_token=(int(qty["token_index"]) if qty else None),
            explicit_cues=candidate.get("explicit_cue_positions", []),
            crosses_boundary=bool(money.get("crosses_relation_boundary", False)),
        )
    unit = canon_unit((qty.get("unit") or "") if qty else "")
    currency = currency_name(money.get("currency") if money else None)
    qvalue = _json_number(qty.get("value")) if qty else None
    pvalue = _json_number(money.get("value")) if money else None
    if money and money.get("intrinsic_value") is not None:
        pvalue = float(money["intrinsic_value"])
    normalization_status = "unapproved_factor"
    price_per_artaba = None
    # A stated rate names its own denominator — "τιμὴν τῆς ἀρτάβης ἑκάστης …
    # δραχμὰς τέσσαρας" is four drachmas the artaba whether or not the scanner
    # produced an artaba option to point at. Taking the unit from the quantity
    # option alone made normalisation an accident of extraction: `c128` reached
    # the series because a value-less ἀρτάβη option happened to exist, while
    # `c007` says the same thing and was dropped for having none.
    #
    # The unit still has to come from evidence rather than from an assumption —
    # a stated rate can be per metron or per keramion, which is why an unknown
    # unit stays out. `unit_read` is the annotator's own reading of it, and is
    # consulted only where the scanner offered no quantity option to read. The
    # model has no equivalent field, so the LLM path is unaffected.
    if not unit and unit_read:
        unit = canon_unit(unit_read)
    if EXCLUDE_MARK in (note or ""):
        # The annotator has recorded that the figure in the option is not what
        # the papyrus says — `c046` selects a money option carrying 1 because
        # the edition encodes `<num value="1"/>` on δέκα (issue #031). Nothing
        # in the row is wrong: it faithfully names the option it was given, and
        # the gold's contract is that it matches. What is wrong is upstream, so
        # the observation is kept and published to
        # `excluded_from_public_analysis.jsonl` with the reason on it, rather
        # than entering a series ten times low.
        normalization_status = "disputed_source_value"
    elif unit == "artaba" and currency == "drachma" and pvalue:
        # Definition §5: a rate is either **stated** ("ἀνὰ δραχμὰς N per artaba")
        # or **derived** from total ÷ quantity. A stated rate is already the
        # per-artaba figure, so dividing it by anything is wrong — and demanding
        # a divisor is why every stated rate used to fall out of the analysable
        # series while weaker derived observations stayed in.
        if rate_basis == "stated_rate":
            price_per_artaba = pvalue
            normalization_status = "direct_drachma_per_artaba"
        elif qvalue:
            price_per_artaba = pvalue / qvalue
            normalization_status = "direct_drachma_per_artaba"
    klass = commodity_class(commodity)
    in_scope = klass == "raw_grain"
    return {
        "window_id": candidate["window_id"],
        "observation_index": observation_index,
        "doc_id": candidate["ddb_id"],
        "method": method,
        "definition_version": definition_version,
        **({"observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "candidate_fingerprint": candidate["candidate_fingerprint"],
            "source_block_id": candidate["source_block_id"],
            "source_block_sha256": candidate["source_block_sha256"],
            "coordinate_system": candidate["coordinate_system"],
            "evidence_char_span": locate_evidence(candidate, evidence_span),
            "quantity_value_span": ([qty["value_char_start"], qty["value_char_end"]]
                                    if qty and qty.get("value_char_start") is not None else None),
            "price_value_span": ([money["value_char_start"], money["value_char_end"]]
                                 if money and money.get("value_char_start") is not None else None),
            "quantity_inference_rule": qty.get("inference_rule") if qty else None,
            "price_inference_rule": money.get("inference_rule") if money else None,
        } if candidate.get("candidate_schema_version") else {}),
        "commodity_surface": candidate.get("grain_form"),
        "commodity": commodity or canon_grain(candidate.get("grain_form", "")),
        "commodity_class": klass,
        "quantity_option": qty.get("option_id") if qty else None,
        "quantity_value": qty.get("value") if qty else None,
        "quantity_unit": qty.get("unit") if qty else None,
        "quantity_unit_canonical": unit or None,
        "price_option": money.get("option_id") if money else None,
        "price_value": money.get("value") if money else None,
        "price_currency": money.get("currency") if money else None,
        "price_currency_canonical": currency or None,
        # Block-local displayed character spans of the selected options.
        # windows can select the same pair; these let analyze find it (#036).
        "quantity_char_span": _char_span(qty),
        "price_char_span": _char_span(money),
        "price_value_numeric": pvalue,
        "linkage_strength": linkage,
        "relation_cues": relation_cues,
        "context_tags": context_tags,
        "rate_basis": rate_basis,
        "evidence_span": evidence_span,
        "confidence": confidence,
        "reasoning": reasoning,
        "in_scope": in_scope,
        "series_strict": in_scope and linkage == "explicit",
        "series_broad": in_scope and linkage in {"explicit", "proximate"},
        "series_exploratory": in_scope,
        "analysis_eligible": in_scope and normalization_status == "direct_drachma_per_artaba",
        "price_per_artaba_drachma": price_per_artaba,
        "normalization_status": normalization_status,
        "place": candidate.get("place", ""),
        "date_not_before": candidate.get("date_not_before", ""),
        "date_not_after": candidate.get("date_not_after", ""),
    }


def materialize(input_candidates: Path, predictions: Path | None, output: Path,
                annotations: Path | None = None, repeat_index: int = 0) -> dict:
    candidates = {r["window_id"]: r for r in read_jsonl(input_candidates)}
    final: list[dict] = []
    invalid = errors = unjudged = 0
    judged = set()
    prediction_windows = set()
    for prediction in read_jsonl(predictions) if predictions else []:
        if int(prediction.get("repeat_index", 0)) != repeat_index:
            continue
        window_id = prediction.get("window_id")
        candidate = candidates.get(window_id)
        if candidate is not None:
            validate_candidate_binding(prediction, candidate)
        if candidate is None or prediction.get("error"):
            errors += 1
            continue
        answer = prediction.get("answer")
        problems = list(prediction.get("validation_problems") or [])
        problems += validate_prediction(answer, candidate)
        if problems:
            invalid += 1
            continue
        if window_id in prediction_windows:
            raise ValueError(
                f"duplicate valid method prediction for window {window_id}; do not mix experiments"
            )
        prediction_windows.add(window_id)
        judged.add(window_id)
        for observation_index, obs in enumerate(answer["observations"], 1):
            qty = option_by_id(candidate, "quantity", obs.get("quantity_option"))
            money = option_by_id(candidate, "money", obs.get("money_option"))
            final.append(_final_row(
                candidate, qty, money, obs.get("commodity", ""),
                f"llm__{prediction.get('provider')}__{prediction.get('model_id')}",
                observation_index, obs.get("context_tags") or [], obs.get("rate_basis", "unknown"),
                obs.get("evidence_span", ""), obs.get("confidence", ""),
                obs.get("reasoning", ""), prediction.get("definition_version") or DEFINITION_VERSION,
            ))
    if annotations:
        annotation_errors = validate_annotations(annotations, input_candidates)
        if annotation_errors:
            raise ValueError("invalid human annotations: " + "; ".join(annotation_errors[:10]))
        with annotations.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                # POSITIVE_LABELS carries the canonical PRICE alongside the legacy
                # TP/ME, so a gold file written under definition canonical_v13_1
                # materialises instead of being silently skipped.
                label = row.get("canonical_label", "").strip()
                if not label:
                    continue
                candidate = candidates.get(row["window_id"].strip())
                if candidate is None:
                    continue
                # A negative judgment is still a judgment: it takes the window out
                # of the unadjudicated pool without yielding an observation. The
                # prediction path above already counts any_price=false this way.
                judged.add(candidate["window_id"])
                if label not in POSITIVE_LABELS:
                    continue
                qty_id = row.get("quantity_option", "").strip() or None
                money_id = row.get("money_option", "").strip() or None
                qty = option_by_id(candidate, "quantity", qty_id)
                money = option_by_id(candidate, "money", money_id)
                tags = [tag.strip() for tag in row.get("final_context_tags", "").replace(";", ",").split(",") if tag.strip()]
                final.append(_final_row(
                    candidate, qty, money, row.get("Human_Commodity", ""), "human",
                    1, tags or ["unknown"], row.get("rate_basis", "unknown") or "unknown",
                    row.get("evidence_span", ""), "human", row.get("Human_Notes", ""),
                    DEFINITION_VERSION, unit_read=row.get("Human_Unit", ""),
                    note=row.get("context_note", ""),
                ))
    unjudged = len(candidates) - len(judged)
    for row in final:
        row["repeat_index"] = None if annotations else repeat_index
    write_jsonl(output, final)
    grouped = deduplicate_observations(final)
    grouped_path = output.with_name(output.stem + "_deduplicated.jsonl")
    write_jsonl(grouped_path, grouped)
    return {
        "candidate_mentions": len(candidates), "judged_mentions": len(judged),
        "unjudged_mentions": unjudged, "final_observations": len(final),
        "invalid_predictions": invalid, "prediction_errors": errors,
        "judgment_source": "human" if annotations else "llm",
        "repeat_index": None if annotations else repeat_index,
        "deduplicated_artifact": grouped_path.name,
        "source_groups": sum(r["grouping_status"] != "unresolved_location" for r in grouped),
        "interpretation_conflicts": sum(r["grouping_status"] == "interpretation_conflict" for r in grouped),
        "unresolved_location_rows": sum(r["grouping_status"] == "unresolved_location" for r in grouped),
    }

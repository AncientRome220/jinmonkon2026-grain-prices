from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from grain_prices.canonical.annotation import validate_annotations
from grain_prices.canonical.io import read_jsonl


LEGACY_EXPECTED = {
    "rb_scored_subset_optimal_f1": 0.593,
    "gemini_all_nonborder_f1": 0.6436781609195403,
}


def _legacy_rb_optimal_f1(gold_csv: Path) -> float:
    gold = pd.read_csv(gold_csv, encoding="utf-8-sig", keep_default_na=False)
    gold = gold.loc[gold.Human_Label.ne("BORDER")].copy()
    gold["score"] = pd.to_numeric(gold.RB_Score, errors="coerce")
    scored = gold.loc[gold.score.notna()].copy()
    truth = scored.Human_Label.isin(["TP", "ME"])
    values = []
    for threshold in scored.score.unique():
        pred = scored.score.ge(threshold)
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum())
        values.append(2 * tp / (2 * tp + fp + fn))
    return max(values, default=0.0)


def evaluate_legacy(report_csv: Path, output: Path,
                    gold_csv: Path = Path("data/annotations/gold_standard_annotation.csv")) -> dict:
    """Validate the frozen legacy harness without relabeling its gold."""
    df = pd.read_csv(report_csv)
    llm = df.loc[df.method.eq("LLM (Gemini Flash)")]
    if len(llm) != 1:
        raise ValueError("legacy Gemini row is missing or duplicated")
    actual = float(llm.iloc[0].f1)
    rb_actual = _legacy_rb_optimal_f1(gold_csv)
    result = {
        "definition_version": "legacy_v12",
        "purpose": "frozen evaluator regression only",
        "gemini_all_nonborder_f1": actual,
        "gemini_regression_passed": abs(actual - LEGACY_EXPECTED["gemini_all_nonborder_f1"]) < 1e-12,
        "rb_scored_subset_optimal_f1": rb_actual,
        "rb_regression_passed": abs(rb_actual - LEGACY_EXPECTED["rb_scored_subset_optimal_f1"]) < 0.0005,
        "warning": "Legacy metrics are not canonical_v13_1 performance.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                      newline="\n")
    return result


def evaluate_canonical(annotation_csv: Path, output: Path,
                       candidates_path: Path | None = None) -> dict:
    errors = validate_annotations(annotation_csv, candidates_path, require_complete_mapping=True)
    df = pd.read_csv(annotation_csv, encoding="utf-8-sig", keep_default_na=False)
    result = {
        "definition_version": "canonical_v13_1",
        "gold_rows": len(df),
        "mapping_complete": not any("window_id" in error for error in errors),
        "annotation_complete": not errors,
        "validation_errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                      newline="\n")
    return result


def _binary_metrics(truth: list[bool], predicted: list[bool]) -> dict:
    tp = sum(1 for t, p in zip(truth, predicted) if t and p)
    fp = sum(1 for t, p in zip(truth, predicted) if not t and p)
    fn = sum(1 for t, p in zip(truth, predicted) if t and not p)
    tn = sum(1 for t, p in zip(truth, predicted) if not t and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(truth), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / len(truth), 4) if truth else 0.0,
    }


def _tag_set(value) -> set[str]:
    return {tag.strip() for tag in str(value or "").replace(";", ",").split(",") if tag.strip()}


def evaluate_llm_window(predictions_path: Path, annotation_csv: Path, output: Path,
                        repeat_index: int = 0, candidates_path: Path | None = None) -> dict:
    """Score one model run against the canonical gold at window level.

    The headline number answers "does this window contain a grain valuation" -
    the `any_price` axis of definition section 7. Rows the model errored on stay
    in the denominator as predicted-negative, and are also reported separately so
    a run with many failures cannot look like a confident negative.
    """
    gold = pd.read_csv(annotation_csv, encoding="utf-8-sig", keep_default_na=False)
    labelled = gold.loc[gold.canonical_label.astype(str).str.strip().ne("")].copy()
    if labelled.empty:
        raise ValueError(f"no canonical labels in {annotation_csv}")

    predictions = [row for row in read_jsonl(predictions_path)
                   if int(row.get("repeat_index", 0)) == repeat_index]
    from grain_prices.canonical.provenance import validate_candidate_binding
    if candidates_path:
        candidates = {c["window_id"]: c for c in read_jsonl(candidates_path)}
        for prediction in predictions:
            candidate = candidates.get(prediction["window_id"])
            if candidate is None:
                raise ValueError("prediction window absent from evaluation candidates")
            validate_candidate_binding(prediction, candidate)
    elif any(p.get("candidate_fingerprint") for p in predictions):
        raise ValueError("fingerprinted predictions require evaluation candidates")
    if len({p["window_id"] for p in predictions}) != len(predictions):
        raise ValueError("duplicate prediction windows in evaluation repetition")
    by_window = {row["window_id"]: row for row in predictions}
    models = sorted({row.get("model_id", "") for row in predictions})
    providers = sorted({row.get("provider", "") for row in predictions})

    truth, predicted, missing, errored, invalid = [], [], [], [], []
    low_confidence = []
    for _, row in labelled.iterrows():
        window_id = str(row.window_id).strip()
        prediction = by_window.get(window_id)
        truth.append(str(row.canonical_label).strip() == "PRICE")
        if prediction is None:
            missing.append(window_id)
            predicted.append(False)
        else:
            if prediction.get("error"):
                errored.append(window_id)
            if prediction.get("validation_problems"):
                invalid.append(window_id)
            answer = prediction.get("answer") or {}
            predicted.append(bool(answer.get("any_price")))
        low_confidence.append(str(row.get("annotator_confidence", "")).strip() == "low")

    result = {
        "definition_version": "canonical_v13_1",
        "annotation": str(annotation_csv),
        "predictions": str(predictions_path),
        "repeat_index": repeat_index,
        "providers": providers,
        "model_ids": models,
        "any_price": _binary_metrics(truth, predicted),
        "predictions_missing": len(missing),
        "predictions_errored": len(errored),
        "predictions_schema_invalid": len(invalid),
    }

    confident = [index for index, low in enumerate(low_confidence) if not low]
    if len(confident) != len(truth):
        result["any_price_excluding_low_confidence"] = _binary_metrics(
            [truth[i] for i in confident], [predicted[i] for i in confident])

    observation = _score_observations(labelled, by_window)
    if observation:
        result["observation"] = observation

    stability = _score_stability(predictions_path)
    if stability:
        result["stability"] = stability

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8", newline="\n")
    return result


def _option(observation: dict, field: str) -> str:
    """One option id, with null and empty string collapsed onto each other.

    A stated rate has no quantity and the contract answers `null` for it (§5);
    the gold file spells the same absence as an empty cell. They have to compare
    equal or every stated rate scores as a quantity miss.
    """
    return str(observation.get(field) or "").strip()


def _score_observations(labelled: pd.DataFrame, by_window: dict) -> dict:
    """Pass 2, over the PRICE rows whose fields are filled in.

    Two readings of the same answers, because they measure different things.

    **Recall** asks whether the gold pairing appears anywhere in the model's
    observation array. That is what the gold row actually asserts — this pairing
    is a real observation in this window — and 90 of the 150 sampled windows hold
    more than one grain mention, so demanding it arrive in a particular slot
    scores ordering as if it were judgement.

    **At-first** keeps the positional reading: definition §7 fixes which pairing
    is primary and the prompt now asks for the array in that order, so element 0
    is a claim the model is making and worth scoring on its own. The gap between
    the two numbers is the size of the ordering artifact; both are reported so it
    stays visible rather than being absorbed into one figure.

    Recall alone is gameable — emit every candidate pairing and it reaches 1.0 —
    so the emitted-count figures below it are not decoration. Read them together.
    """
    # A finished PRICE row is one that carries either half of pass 2. Keying on
    # money_option alone would drop the rows that issue #034 released: a window
    # can state a price the scanner never turned into a candidate, so the gold
    # names no money option and quotes the text instead. Those rows carry a real
    # claim the model can agree or disagree with, since the contract lets it
    # answer null too, and they belong in the denominator.
    def filled(name: str):
        if name not in labelled.columns:
            return pd.Series(False, index=labelled.index)
        return labelled[name].astype(str).str.strip().ne("")

    rows = labelled.loc[labelled.canonical_label.astype(str).str.strip().eq("PRICE")
                        & (filled("evidence_span") | filled("money_option"))]
    if rows.empty:
        return {}
    money = quantity = pair = tags = counts = 0
    money_first = pair_first = 0
    emitted = expected_total = 0
    gold_null = predicted_null = 0
    scored = 0
    for _, row in rows.iterrows():
        prediction = by_window.get(str(row.window_id).strip())
        answer = (prediction or {}).get("answer") or {}
        observations = answer.get("observations") or []
        scored += 1
        emitted += len(observations)
        expected = str(row.n_observations).strip()
        if expected.isdigit():
            expected_total += int(expected)
        if not observations:
            continue

        gold_money = str(row.money_option).strip()
        gold_quantity = str(row.quantity_option).strip()
        gold_tags = _tag_set(row.final_context_tags)
        gold_null += not gold_money
        predicted_null += sum(1 for o in observations if not _option(o, "money_option"))

        matched = [o for o in observations if _option(o, "money_option") == gold_money]
        money += bool(matched)
        quantity += any(_option(o, "quantity_option") == gold_quantity for o in observations)
        paired = [o for o in matched if _option(o, "quantity_option") == gold_quantity]
        pair += bool(paired)
        # Tags are credited only on an observation that got the pairing right.
        # Overlapping tags on some other pairing in the window say nothing about
        # whether the model read *this* transaction correctly.
        tags += any(gold_tags & set(o.get("context_tags") or []) for o in paired)

        first = observations[0]
        money_first += _option(first, "money_option") == gold_money
        pair_first += (_option(first, "money_option") == gold_money
                       and _option(first, "quantity_option") == gold_quantity)

        counts += expected.isdigit() and int(expected) == len(observations)

    def share(value):
        return round(value / scored, 4) if scored else 0.0

    return {
        "n": scored,
        # found the gold pairing somewhere in the answer
        "money_option_recall": share(money),
        "quantity_option_recall": share(quantity),
        "pair_recall": share(pair),
        "context_tag_recall": share(tags),
        # ... and ranked it first, which definition §7 says it should be
        "money_option_at_first": share(money_first),
        "pair_at_first": share(pair_first),
        # the guard on recall: an answer that lists everything scores well above
        # on recall and badly here
        "observation_count_agreement": share(counts),
        "observations_predicted": emitted,
        "observations_expected": expected_total,
        # "no offered amount is the price" (issue #034). It is a real answer, and
        # also the cheapest way to claim a price without committing to a
        # candidate, so both sides are counted and read against each other: a
        # model well above the gold is dodging the choice, not finding ellipsis.
        "money_option_null_gold": gold_null,
        "money_option_null_predicted": predicted_null,
    }


def _score_stability(predictions_path: Path) -> dict:
    """Agreement of `any_price` across repeats of the same window."""
    seen: dict[str, dict[int, bool]] = {}
    for row in read_jsonl(predictions_path):
        answer = row.get("answer") or {}
        seen.setdefault(row["window_id"], {})[int(row.get("repeat_index", 0))] = \
            bool(answer.get("any_price"))
    repeated = {window: runs for window, runs in seen.items() if len(runs) > 1}
    if not repeated:
        return {}
    agreed = sum(1 for runs in repeated.values() if len(set(runs.values())) == 1)
    return {
        "windows_rerun": len(repeated),
        "repeats": max(len(runs) for runs in repeated.values()),
        "unanimous_windows": agreed,
        "agreement": round(agreed / len(repeated), 4),
    }

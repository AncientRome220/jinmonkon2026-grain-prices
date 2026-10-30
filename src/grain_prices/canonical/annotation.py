from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from grain_prices.canonical.io import read_jsonl
from grain_prices.canonical.contracts import (
    COMMODITY_CLASSES, CONFIDENCE_VALUES, CONTEXT_VALUES, LINKAGE_VALUES,
    RATE_BASIS_VALUES, option_by_id,
)
from grain_prices.pipeline_shared import canon_currency, canon_unit, normalize_for_search, to_float

# `PRICE`/`NO_PRICE` are the canonical window-level labels of
# docs/grain_price_definition.md §7; TP/FP/ME/BORDER are the superseded legacy
# set, kept readable so the frozen legacy harness still validates (§11).
CANONICAL_LABELS = {"PRICE", "NO_PRICE"}
LEGACY_LABELS = {"TP", "FP", "ME", "BORDER"}
LABELS = CANONICAL_LABELS | LEGACY_LABELS | {""}
POSITIVE_LABELS = {"PRICE", "TP", "ME"}

# §6 hard-exclusion set, recorded on NO_PRICE rows so the residual is auditable.
# Three kinds of reason live here and they answer different questions:
#   * nothing here values this grain in money -> non_grain, no_money
#   * the transaction really is in kind       -> in_kind_ration, seed_loan,
#                                                pure_tax_in_kind, freight, milling
#   * the papyrus does not survive well enough to judge -> fragmentary
# `no_money` covers more than a mis-matched money word. It is the answer whenever
# no money in the window values *this* grain, which includes the commonest false
# positive of all: a register listing grain and cash side by side with no
# relation asserted between them (`πυροῦ ἀρτάβας 20 καὶ δραχμὰς 32`). Nine of the
# first seventeen `no_money` rows contain a real currency word, so reading the
# name literally would push that whole class into `other`.
# `fragmentary` is not in §6 (nor is `no_money`); both are code-level additions.
# It is kept separate from `other` because it is neither a fixable defect nor an
# economic finding: it is the ceiling on what any method could reach, which is
# what makes it worth counting on its own. `other` is the residual for windows
# none of the rest describes — putting the largest identifiable subclass in it
# would make the residual unreadable.
EXCLUSION_REASONS = {
    "freight", "milling", "in_kind_ration", "seed_loan",
    "pure_tax_in_kind", "non_grain", "no_money", "fragmentary", "other",
}

HUMAN_FIELDS = [
    "Human_Commodity", "Human_Qty", "Human_Unit", "Human_Price", "Human_Currency"
]


def _sample_key(value) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def _context_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    compact_a = a.replace(" ", "")
    compact_b = b.replace(" ", "")
    return compact_a in compact_b or compact_b in compact_a


def _norm(value) -> str:
    return normalize_for_search("" if pd.isna(value) else str(value))


def _proportional_allocation(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder allocation so the parts sum to `total` exactly."""
    pool = sum(sizes.values())
    if pool == 0:
        return {key: 0 for key in sizes}
    exact = {key: value * total / pool for key, value in sizes.items()}
    alloc = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(alloc.values())
    ranked = sorted(sizes, key=lambda key: (exact[key] - alloc[key], sizes[key]), reverse=True)
    for key in ranked[:remainder]:
        alloc[key] += 1
    return alloc


def _doc_id(row: dict) -> str:
    return str(row.get("doc_id", row.get("ddb_id", "")))


def _option_summary(option: dict, kind: str) -> dict:
    """The fields an annotator needs to tell two option ids apart."""
    common = {
        "option_id": option.get("option_id", ""),
        "value": option.get("value", ""),
        "dist_from_grain": option.get("dist_from_grain", ""),
        "crosses_block_boundary": option.get("crosses_block_boundary", ""),
    }
    if kind == "money":
        common["currency"] = option.get("currency", "")
        common["linkage_with_nearest_quantity"] = option.get("linkage_with_nearest_quantity", "")
        common["crosses_relation_boundary"] = option.get("crosses_relation_boundary", "")
    else:
        common["unit"] = option.get("unit", "")
    return common


def sample_canonical(candidates_path: Path, out_path: Path, manifest_path: Path,
                     size: int = 150, seed: int = 20260818, per_doc_cap: int = 2,
                     stratify_by: str = "linkage_strength",
                     subset_path: Path | None = None) -> dict:
    """Draw a fresh canonical gold sample straight from the candidate options.

    Sampling from `candidate_options.jsonl` rather than re-mapping the legacy
    gold means every row already carries a `window_id` and every row is
    scoreable by `score-llm` (it has at least one money option). Strata keep the
    draw representative; the per-document cap keeps a handful of large ledgers
    from dominating it.
    """
    rows = list(read_jsonl(candidates_path))
    if not rows:
        raise ValueError(f"no candidate options in {candidates_path}")

    strata: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        strata[str(row.get(stratify_by, "") or "unknown")].append(row)
    alloc = _proportional_allocation({key: len(value) for key, value in strata.items()}, size)

    rng = random.Random(seed)
    per_doc: Counter[str] = Counter()
    picked: list[tuple[str, dict]] = []
    shortfall: dict[str, int] = {}
    for stratum in sorted(strata):
        pool = strata[stratum][:]
        rng.shuffle(pool)
        taken = 0
        for row in pool:
            if taken >= alloc[stratum]:
                break
            if per_doc[_doc_id(row)] >= per_doc_cap:
                continue
            per_doc[_doc_id(row)] += 1
            picked.append((stratum, row))
            taken += 1
        if taken < alloc[stratum]:
            shortfall[stratum] = alloc[stratum] - taken

    # A stratum can run dry under the cap; top up from the rest so the sample
    # still reaches `size`, and record that the draw was adjusted.
    if len(picked) < size:
        chosen = {row["window_id"] for _, row in picked}
        spare = [(stratum, row) for stratum in sorted(strata) for row in strata[stratum]
                 if row["window_id"] not in chosen]
        rng.shuffle(spare)
        for stratum, row in spare:
            if len(picked) >= size:
                break
            if per_doc[_doc_id(row)] >= per_doc_cap:
                continue
            per_doc[_doc_id(row)] += 1
            picked.append((stratum, row))

    picked.sort(key=lambda item: (_doc_id(item[1]), item[1]["window_id"]))
    output = []
    for index, (stratum, row) in enumerate(picked, start=1):
        qty_options = row.get("qty_options", [])
        money_options = row.get("money_options", [])
        output.append({
            "sample_id": f"c{index:03d}",
            "window_id": row["window_id"],
            "doc_id": _doc_id(row),
            "legacy_label": "",
            "canonical_label": "",
            "annotator_confidence": "",
            "exclusion_reason": "",
            "n_observations": "",
            "grain_surface": row.get("grain_surface", ""),
            "evidence_text": row.get("evidence_text", ""),
            "window_text": row.get("window_text", ""),
            "quantity_option": "",
            "money_option": "",
            "quantity_option_choices": ";".join(o["option_id"] for o in qty_options),
            "money_option_choices": ";".join(o["option_id"] for o in money_options),
            "option_details_json": json.dumps({
                "quantity": [_option_summary(o, "quantity") for o in qty_options],
                "money": [_option_summary(o, "money") for o in money_options],
            }, ensure_ascii=False, sort_keys=True),
            "commodity": "",
            "commodity_class": "",
            "linkage_strength": "",
            "final_context_tags": "",
            "context_note": "",
            "rate_basis": "unknown",
            "evidence_span": "",
            **{field: "" for field in HUMAN_FIELDS},
            "Human_Notes": "",
            "Annotation_Seconds": "",
            "Corrected_Fields": "",
            # Rule-based reference values, kept for error analysis. They are the
            # pipeline's guesses, not ground truth - do not copy them across.
            "sampling_stratum": stratum,
            "rb_linkage_strength": row.get("linkage_strength", ""),
            "rb_context_tags": ";".join(row.get("context_tags", []) or []),
            "rb_priceword_near": row.get("priceword_near", ""),
            "n_qty_options": len(qty_options),
            "n_money_options": len(money_options),
            "place": row.get("place", ""),
            "date_not_before": row.get("date_not_before", ""),
            "date_not_after": row.get("date_not_after", ""),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_csv(out_path, index=False, encoding="utf-8-sig")

    # Emit the matching score-llm input so the run and the gold cannot drift
    # apart; --limit only slices the head of a file, it does not filter.
    if subset_path is not None:
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        with subset_path.open("w", encoding="utf-8", newline="\n") as handle:
            for _, row in picked:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "definition_version": "canonical_v13_1",
        "sample_version": "canonical-gold-sample-v1",
        "source": str(candidates_path),
        "source_sha256": hashlib.sha256(candidates_path.read_bytes()).hexdigest(),
        "source_rows": len(rows),
        "seed": seed,
        "size": size,
        "per_doc_cap": per_doc_cap,
        "stratify_by": stratify_by,
        "stratum_pool": {key: len(value) for key, value in sorted(strata.items())},
        "stratum_allocated": dict(sorted(alloc.items())),
        "stratum_drawn": dict(sorted(Counter(stratum for stratum, _ in picked).items())),
        "stratum_shortfall": shortfall,
        "subset_path": str(subset_path) if subset_path else None,
        "documents": len(per_doc),
        "max_per_document": max(per_doc.values()) if per_doc else 0,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
                             encoding="utf-8", newline="\n")
    return manifest


def prepare_gold(gold_path: Path, spine_path: Path, out_path: Path,
                 unmatched_path: Path, relabel_source: Path | None = None) -> dict:
    gold = pd.read_csv(gold_path, encoding="utf-8-sig", keep_default_na=False)
    spine_by_doc: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(spine_path):
        spine_by_doc[row.get("doc_id", row.get("ddb_id", ""))].append(row)
    output: list[dict] = []
    unmatched: list[dict] = []
    relabel_docs: set[str] = set()
    if relabel_source and relabel_source.exists():
        relabel_df = pd.read_csv(relabel_source, encoding="utf-8-sig", keep_default_na=False)
        relabel_docs = set(relabel_df.DDB_ID.astype(str))
    for _, source in gold.iterrows():
        rec = source.to_dict()
        doc_id = str(rec.get("DDB_ID", ""))
        context = _norm(rec.get("Context_Window", ""))
        grain = _norm(rec.get("Grain_Form", ""))
        candidates = []
        for window in spine_by_doc.get(doc_id, []):
            wtext = _norm(window.get("window_text", window.get("evidence_text", "")))
            wgrain = _norm(window.get("grain_surface", window.get("grain_form", "")))
            if grain and grain != wgrain:
                continue
            exact = _context_match(context, wtext)
            ratio = SequenceMatcher(None, context, wtext).ratio() if context and wtext else 0.0
            candidates.append((exact, ratio, window))
        exact_matches = [item for item in candidates if item[0]]
        selected = None
        if len(exact_matches) == 1:
            selected = exact_matches[0][2]
        elif len(exact_matches) > 1:
            best_ratio = max(item[1] for item in exact_matches)
            best = [item for item in exact_matches if item[1] == best_ratio]
            if len(best) == 1:
                selected = best[0][2]
        mapped = {
            "sample_id": _sample_key(rec.get("Sample_ID", "")),
            "window_id": selected["window_id"] if selected else "",
            "doc_id": doc_id,
            "legacy_label": str(rec.get("Human_Label", "")),
            "canonical_label": "" if doc_id in relabel_docs else str(rec.get("Human_Label", "")),
            "grain_surface": rec.get("Grain_Form", ""),
            "window_text": rec.get("Context_Window", ""),
            "quantity_option": "",
            "money_option": "",
            "quantity_option_choices": ";".join(
                option["option_id"] for option in (selected or {}).get("qty_options", [])
            ),
            "money_option_choices": ";".join(
                option["option_id"] for option in (selected or {}).get("money_options", [])
            ),
            "final_context_tags": "",
            "rate_basis": "unknown",
            "evidence_span": "",
            **{field: rec.get(field, "") for field in HUMAN_FIELDS},
            "Human_Notes": rec.get("Human_Notes", ""),
            "Annotation_Seconds": "",
            "Corrected_Fields": "",
        }
        output.append(mapped)
        if selected is None:
            ranked = sorted(candidates, key=lambda item: item[1], reverse=True)[:5]
            unmatched.append({
                "sample_id": mapped["sample_id"], "doc_id": doc_id,
                "grain_surface": mapped["grain_surface"],
                "candidate_window_ids": [item[2]["window_id"] for item in ranked],
                "candidate_similarity": [round(item[1], 4) for item in ranked],
                "reason": "ambiguous_exact_match" if len(exact_matches) > 1 else "no_exact_match",
            })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_csv(out_path, index=False, encoding="utf-8-sig")
    unmatched_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in unmatched),
        encoding="utf-8", newline="\n",
    )
    return {"gold_rows": len(output), "mapped": len(output) - len(unmatched), "unmatched": len(unmatched)}


def prepare_relabel_worklist(gold_csv: Path, legacy_worklist: Path,
                             candidates_path: Path, output: Path) -> dict:
    gold = pd.read_csv(gold_csv, encoding="utf-8-sig", keep_default_na=False)
    old = pd.read_csv(legacy_worklist, encoding="utf-8-sig", keep_default_na=False)
    candidates_by_doc: dict[str, list[dict]] = defaultdict(list)
    for candidate in read_jsonl(candidates_path):
        candidates_by_doc[candidate["ddb_id"]].append(candidate)
    rows = []
    for _, source in old.iterrows():
        doc_id = str(source["DDB_ID"])
        context = _norm(source.get("Context_Window", ""))
        sample_choices = gold[gold.doc_id.astype(str).eq(doc_id)].sample_id.astype(str).tolist()
        candidate_choices = sorted(
            candidates_by_doc.get(doc_id, []),
            key=lambda c: SequenceMatcher(None, context, _norm(c.get("evidence_text", ""))).ratio(),
            reverse=True,
        )[:5]
        if len(sample_choices) != 1:
            sample_reason = "no_gold_row_for_doc" if not sample_choices else "multiple_gold_rows_for_doc"
        else:
            sample_reason = "requires_human_confirmation"
        rows.append({
            "work_item_id": f"relabel-{len(rows)+1:03d}",
            "sample_id": "",
            "window_id": "",
            "doc_id": doc_id,
            "legacy_label": source.get("Current_Label", ""),
            "canonical_label": "",
            "suggested_context_tags": source.get("New_Context_Tags", ""),
            "final_context_tags": "",
            "sample_id_choices": ";".join(sample_choices),
            "mapping_status": sample_reason,
            "candidate_window_choices": ";".join(c["window_id"] for c in candidate_choices),
            "candidate_options_json": json.dumps({
                candidate["window_id"]: {
                    "quantity": [option["option_id"] for option in candidate.get("qty_options", [])],
                    "money": [option["option_id"] for option in candidate.get("money_options", [])],
                } for candidate in candidate_choices
            }, ensure_ascii=False, sort_keys=True),
            "quantity_option": "",
            "money_option": "",
            "quantity_option_choices": ";".join(
                option["option_id"] for option in candidate_choices[0].get("qty_options", [])
            ) if candidate_choices else "",
            "money_option_choices": ";".join(
                option["option_id"] for option in candidate_choices[0].get("money_options", [])
            ) if candidate_choices else "",
            "rate_basis": "unknown",
            "evidence_span": "",
            **{field: "" for field in HUMAN_FIELDS},
            "Annotation_Seconds": "",
            "Human_Notes": "",
            "Context_Window": source.get("Context_Window", ""),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    return {"relabel_rows": len(rows), "completed": 0}


def _has_value(option: dict) -> bool:
    """Whether an option carries an amount at all.

    `str(None)` is "None" and therefore truthy, which is why this is a function
    and not an inline check: a line break can cut the numeral out of the
    candidate (#027) and the option then holds a null value.
    """
    return str(option.get("value") or "").strip() != ""


def _selected_options(row, candidates) -> dict:
    """The option objects a gold row points at, or None where it points at none.

    Returns None for both when no candidate file was supplied, so validation of
    a bare CSV keeps working — it simply cannot check what it cannot see.
    """
    candidate = candidates.get(str(row.window_id).strip()) if candidates else None
    if candidate is None:
        return {"quantity": None, "money": None}
    return {
        kind: option_by_id(candidate, kind, str(row.get(f"{key}_option", "")).strip() or None)
        for kind, key in (("quantity", "quantity"), ("money", "money"))
    }


def _validate_canonical_window(index, row) -> list[str]:
    """Window-level checks for the canonical label set (definition §6/§7)."""
    errors: list[str] = []
    where = f"row {index+2}"
    label = str(row.canonical_label).strip()

    confidence = str(row.get("annotator_confidence", "")).strip()
    if not confidence:
        errors.append(f"{where}: annotator_confidence is required")
    elif confidence not in CONFIDENCE_VALUES:
        errors.append(f"{where}: unknown annotator_confidence {confidence!r}")

    reason = str(row.get("exclusion_reason", "")).strip()
    if label == "NO_PRICE":
        if not reason:
            errors.append(f"{where}: NO_PRICE needs an exclusion_reason")
        elif reason not in EXCLUSION_REASONS:
            errors.append(f"{where}: unknown exclusion_reason {reason!r}")
    elif reason:
        errors.append(f"{where}: exclusion_reason set on a PRICE row")

    observations = str(row.get("n_observations", "")).strip()
    if not observations:
        errors.append(f"{where}: n_observations is required")
    elif not observations.isdigit():
        errors.append(f"{where}: n_observations must be a non-negative integer")
    elif label == "PRICE" and int(observations) < 1:
        errors.append(f"{where}: PRICE needs n_observations >= 1")
    elif label == "NO_PRICE" and int(observations) != 0:
        errors.append(f"{where}: NO_PRICE needs n_observations == 0")

    for column, domain in (("linkage_strength", LINKAGE_VALUES),
                           ("commodity_class", COMMODITY_CLASSES),
                           ("rate_basis", RATE_BASIS_VALUES)):
        value = str(row.get(column, "")).strip()
        if value and value not in domain:
            errors.append(f"{where}: unknown {column} {value!r}")
    return errors


def validate_annotations(path: Path, candidates_path: Path | None = None,
                         require_complete_mapping: bool = True,
                         require_observation_fields: bool = True) -> list[str]:
    df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    errors: list[str] = []
    for column in ("sample_id", "window_id", "legacy_label", "canonical_label"):
        if column not in df:
            errors.append(f"missing column: {column}")
    if errors:
        return errors
    duplicates = df[df.sample_id.astype(str).duplicated(keep=False)]
    if len(duplicates):
        errors.append(f"duplicate sample_id: {len(duplicates)} rows")
    if require_complete_mapping and (df.window_id.astype(str).str.strip() == "").any():
        errors.append("not every gold row has a window_id")
    # A file that uses the canonical label set is expected to be fully labelled;
    # the legacy set deliberately leaves the relabel rows blank.
    labels = df.canonical_label.astype(str).str.strip()
    if labels.isin(CANONICAL_LABELS).any():
        unlabelled = int((labels == "").sum())
        if unlabelled:
            errors.append(f"{unlabelled} rows have no canonical_label")
    candidates = {}
    if candidates_path:
        candidates = {row["window_id"]: row for row in read_jsonl(candidates_path)}
    for index, row in df.iterrows():
        label = str(row.canonical_label).strip()
        if label not in LABELS:
            errors.append(f"row {index+2}: unknown canonical_label {label!r}")
        if label in CANONICAL_LABELS:
            errors.extend(_validate_canonical_window(index, row))
        # Candidate membership is a property of the window id alone, so it is
        # checked for every row that names one — not just a pass-2 PRICE row.
        # A NO_PRICE row and a pass-1 PRICE row carry the same stale-id risk,
        # and downstream the LLM evaluation reads a gold row whose window is
        # absent from the candidate set as a missing prediction: a mistyped id
        # would quietly shrink the denominator instead of failing this gate.
        window_id = str(row.window_id).strip()
        candidate = candidates.get(window_id) if (candidates and window_id) else None
        if candidates and window_id and candidate is None:
            errors.append(f"row {index+2}: unknown window_id")
        if label in POSITIVE_LABELS and not require_observation_fields:
            continue
        if label in POSITIVE_LABELS:
            # The reading columns are required only where the row has something
            # to read. Human_Commodity always does. The quantity and money pairs
            # follow whichever option was selected: once #027 and #034 made both
            # option ids optional, demanding the columns anyway told an annotator
            # that a correct empty cell — a stated rate with no quantity, a price
            # written as a bare numeral with no candidate — was an error.
            # An option always names its unit or currency, but not always its
            # amount: a line break can cut the numeral out of the candidate
            # (#027), and the number can be misattributed or absent (#033). The
            # amount columns therefore follow the option's *value*, since there
            # is nothing to agree with when the option has none.
            selected = _selected_options(row, candidates)
            required = ["Human_Commodity"]
            if selected["quantity"] is not None:
                if selected["quantity"].get("unit"):
                    required.append("Human_Unit")
                if _has_value(selected["quantity"]):
                    required.append("Human_Qty")
            if selected["money"] is not None:
                if selected["money"].get("currency"):
                    required.append("Human_Currency")
                if _has_value(selected["money"]):
                    required.append("Human_Price")
            missing = [field for field in required if not str(row.get(field, "")).strip()]
            # Neither option id is required, for one reason with two faces: an
            # accounting hand omits whichever word the reader can supply, and the
            # scanner needs that word to build an option (issue #034).
            #   * no quantity option — a stated rate has nothing to point at
            #     ("τιμὴν τῆς ἀρτάβης ἑκάστης … δραχμὰς δέκα"), or the artaba is
            #     written bare ("αἳ πυρῶν 185 1/2"), or a line break split
            #     ἀρ|τάβης out of the candidate set (#027). The prediction
            #     contract already answers null here.
            #   * no money option — a Ptolemaic account writes the sum as a bare
            #     numeral and the edition does not restore δραχμαί, so
            #     "τιμῆς πυροῦ ἀρταβῶν 5 ἀνὰ 1000" yields no money candidate at
            #     all, though it plainly states a price.
            # Requiring either would force the annotator to select some other
            # amount to satisfy the gate — the one thing the gold must not do —
            # or to record NO_PRICE on a window that states a price, which would
            # score the model's correct any_price as a false positive and corrupt
            # the §4-4 headline. evidence_span stays required: the text can always
            # be quoted even when no option covers it.
            missing += [] if str(row.get("evidence_span", "")).strip() else ["evidence_span"]
            if missing:
                errors.append(f"row {index+2}: canonical {label} missing {','.join(missing)}")
            tags = [tag.strip() for tag in str(row.get("final_context_tags", "")).replace(";", ",").split(",") if tag.strip()]
            if tags and any(tag not in CONTEXT_VALUES for tag in tags):
                errors.append(f"row {index+2}: unknown context tag")
            # The option ids and the reading columns only exist after pass 2,
            # so they stay here; the window id itself was checked above.
            if candidate is not None:
                qid = str(row.get("quantity_option", "")).strip() or None
                mid = str(row.get("money_option", "")).strip() or None
                if qid and option_by_id(candidate, "quantity", qid) is None:
                    errors.append(f"row {index+2}: unknown quantity_option")
                if mid and option_by_id(candidate, "money", mid) is None:
                    errors.append(f"row {index+2}: unknown money_option")
                qty = option_by_id(candidate, "quantity", qid)
                money = option_by_id(candidate, "money", mid)
                if qty:
                    human_qty, option_qty = to_float(row.get("Human_Qty")), to_float(qty.get("value"))
                    if human_qty == human_qty and option_qty == option_qty and abs(human_qty - option_qty) > 1e-9:
                        errors.append(f"row {index+2}: Human_Qty differs from selected option")
                    if canon_unit(row.get("Human_Unit", "")) != canon_unit(qty.get("unit") or ""):
                        errors.append(f"row {index+2}: Human_Unit differs from selected option")
                if money:
                    human_price, option_price = to_float(row.get("Human_Price")), to_float(money.get("value"))
                    if money.get("intrinsic_value") is not None:
                        option_price = float(money["intrinsic_value"])
                    if human_price == human_price and option_price == option_price and abs(human_price - option_price) > 1e-9:
                        errors.append(f"row {index+2}: Human_Price differs from selected option")
                    from grain_prices.canonical.options import currency_name
                    if currency_name(row.get("Human_Currency", "")) != currency_name(money.get("currency")):
                        errors.append(f"row {index+2}: Human_Currency differs from selected option")
                span = _norm(row.get("evidence_span", ""))
                evidence = _norm(candidate.get("evidence_text", ""))
                if span and span not in evidence:
                    errors.append(f"row {index+2}: evidence_span is outside candidate evidence")
    return errors


def merge_relabel_worklist(gold_path: Path, relabel_path: Path, output: Path) -> dict:
    """Merge completed 12-row decisions without mutating legacy labels."""
    gold = pd.read_csv(gold_path, encoding="utf-8-sig", keep_default_na=False)
    relabel = pd.read_csv(relabel_path, encoding="utf-8-sig", keep_default_na=False)
    gold["sample_id"] = gold.sample_id.map(_sample_key)
    relabel["sample_id"] = relabel.sample_id.map(_sample_key)
    if relabel.sample_id.astype(str).duplicated().any():
        raise ValueError("duplicate sample_id in relabel worklist")
    known = set(gold.sample_id.astype(str))
    unknown = sorted(set(relabel.sample_id.astype(str)) - known - {""})
    if unknown:
        raise ValueError(f"unknown relabel sample_id: {unknown[:5]}")
    transfer = [
        "window_id", "canonical_label", "quantity_option", "money_option",
        "final_context_tags", "rate_basis", "evidence_span", *HUMAN_FIELDS,
        "Annotation_Seconds", "Corrected_Fields", "Human_Notes",
    ]
    merged = 0
    for _, row in relabel.iterrows():
        sample_id = _sample_key(row.get("sample_id", ""))
        label = str(row.get("canonical_label", "")).strip()
        if not sample_id or not label:
            continue
        index = gold.index[gold.sample_id.astype(str).eq(sample_id)]
        if len(index) != 1:
            raise ValueError(f"sample_id does not resolve once: {sample_id}")
        for column in transfer:
            if column in row.index:
                if column not in gold.columns:
                    gold[column] = ""
                gold.at[index[0], column] = row.get(column, "")
        merged += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(output, index=False, encoding="utf-8-sig")
    return {"gold_rows": len(gold), "relabel_rows_merged": merged}

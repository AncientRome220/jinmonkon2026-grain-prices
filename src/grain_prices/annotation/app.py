"""Streamlit UI for canonical gold-standard annotation.

Run:
  streamlit run src/grain_prices/annotation/app.py -- --csv data/annotations/gold_canonical_v13_1.csv

Two passes, in this order:
  Pass 1 - window level: canonical_label, annotator_confidence, n_observations,
           and exclusion_reason on NO_PRICE rows.
  Pass 2 - observation level: option ids, commodity, context tags, rate basis
           and evidence span on PRICE rows.

Pass 1 alone is enough to score window-level precision and recall, so it is the
one that must finish. See docs/grain_price_definition.md sections 6, 7 and 11.

The excerpt is rendered with every candidate option labelled in place, from the
candidate JSONL (`--candidates`); the CSV's option_details_json has no offsets.
An optional reading brief (`--briefs`) supplies translation and document context.
Neither ever carries an annotation decision — that would bias the labels this
file exists to keep independent of any model.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from grain_prices.canonical.annotation import EXCLUSION_REASONS
from grain_prices.canonical.contracts import CONTEXT_VALUES, RATE_BASIS_VALUES
from grain_prices.pipeline_shared import normalize_for_search

LABEL_OPTIONS = ["", "PRICE", "NO_PRICE"]
CONFIDENCE_OPTIONS = ["", "high", "medium", "low"]
EXCLUSION_OPTIONS = [""] + sorted(EXCLUSION_REASONS)
# The controlled vocabulary matters: materialize.commodity_class() maps these to
# raw_grain/processed, and anything else becomes "other" -> in_scope=False, which
# drops the observation from every public series without saying so.
COMMODITY_OPTIONS = ["", "wheat", "barley", "emmer", "zea", "grain_unspecified", "flour", "other"]
RATE_BASIS_OPTIONS = sorted(RATE_BASIS_VALUES)
CONTEXT_OPTIONS = sorted(CONTEXT_VALUES)
GRAIN_STEMS = {"πυρ", "σιτ", "κριθ", "ζεα", "ολυρ", "σταχυ", "αλευρ"}

GREEK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Gentium+Plus&display=swap');
.evidence {
  font-family: 'Gentium Plus', 'Palatino Linotype', serif;
  font-size: 1.18rem; line-height: 1.95;
  padding: 0.9rem 1.1rem; border-radius: 6px;
  border: 1px solid rgba(128,128,128,0.35);
}
.evidence mark { padding: 0 .15em; border-radius: 3px; font-weight: 700; }
.evidence .opt { padding: 0 .12em; border-radius: 3px; }
/* user-select:none keeps the option label out of the clipboard. The excerpt is
   meant to be copied into the evidence span, and a label picked up along the
   way ("m1δραχμὰς") fails the substring check for a reason the annotator
   cannot see. */
.evidence .opt sup { font-family: ui-monospace, monospace; font-size: .62em;
                     font-weight: 700; padding: 0 .2em; opacity: .95;
                     user-select: none; -webkit-user-select: none; }
.evidence .money { background: rgba(255,170,0,0.30); box-shadow: inset 0 -2px 0 rgba(255,140,0,.8); }
.evidence .qty   { background: rgba(0,170,255,0.22); box-shadow: inset 0 -2px 0 rgba(0,140,220,.7); }
.evidence .grain { background: rgba(60,220,120,0.30); font-weight: 700; }
</style>
"""


DEFAULT_CSV = "data/annotations/gold_canonical_v13_1.csv"
DEFAULT_CANDIDATES = "outputs/runs/jmc2026-abstract/gold_sample_candidates.jsonl"
DEFAULT_BRIEFS = "data/annotations/reading_briefs_v13_1.json"

# A reading brief may describe the text and where each option sits in it. It may
# not carry an annotation decision, so these are the only keys it can have.
BRIEF_KEYS = {"document", "translation", "options", "cautions"}
OPTION_ID_RE = re.compile(r"\b[mq]\d+\b")
# Stripping a label off pasted text needs no trailing boundary: the excerpt
# renders as "m1δραχμὰς", and \b does not fire between "1" and a Greek letter.
OPTION_LABEL_RE = re.compile(r"(?<!\w)[mq]\d+")
DECISION_WORDS = frozenset({
    "PRICE", "NO_PRICE", "canonical_label", "annotator_confidence",
    "n_observations", "money_option", "quantity_option", "context_tag",
    "rate_basis", "stated_rate", "derived_from_total", "linkage_strength",
})


def _argument(name: str, environment: str, default: str) -> str:
    """One CLI/env/default lookup, shared by every path the app resolves.

    parse_known_args keeps streamlit's own arguments from breaking the parse.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--{name}")
    known, _ = parser.parse_known_args()
    return getattr(known, name.replace("-", "_")) or os.environ.get(environment) or default


def resolve_csv() -> str:
    return _argument("csv", "GRAIN_PRICES_GOLD_CSV", DEFAULT_CSV)


def resolve_candidates() -> str:
    return _argument("candidates", "GRAIN_PRICES_CANDIDATES", DEFAULT_CANDIDATES)


def resolve_briefs() -> str:
    return _argument("briefs", "GRAIN_PRICES_BRIEFS", DEFAULT_BRIEFS)


def load_candidates(path: str) -> dict[str, dict]:
    """Candidate rows by window_id. Only these carry the option char offsets."""
    source = Path(path)
    if not source.is_file():
        return {}
    rows = {}
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["window_id"]] = row
    return rows


def validate_briefs(data: dict) -> list[str]:
    """Refuse a brief that carries an annotation decision instead of a reading.

    The gold standard is the instrument that measures the models, so it has to be
    produced without seeing a model's answer. A brief may say what the text
    means and where each option stands; the moment it says which option to take,
    the labels stop being independent. Structure alone cannot express that, so
    the check is: option ids appear only as keys of `options`, and no field name
    or label value from the annotation schema appears in prose at all.
    """
    problems: list[str] = []
    for sample_id, brief in data.items():
        if sample_id.startswith("_"):
            continue
        where = f"brief {sample_id}"
        if not isinstance(brief, dict):
            problems.append(f"{where}: expected an object")
            continue
        unknown = sorted(set(brief) - BRIEF_KEYS)
        if unknown:
            problems.append(f"{where}: unexpected key(s) {', '.join(unknown)}")
        prose = [str(brief.get(key, "")) for key in ("document", "translation")]
        prose += [str(item) for item in brief.get("cautions") or []]
        prose += [str(value) for value in (brief.get("options") or {}).values()]
        for text in prose:
            found = OPTION_ID_RE.findall(text)
            if found:
                problems.append(f"{where}: names option {found[0]} outside the options mapping")
            leaked = sorted(word for word in DECISION_WORDS if word in text)
            if leaked:
                problems.append(f"{where}: carries decision vocabulary {', '.join(leaked)}")
    return problems


def load_briefs(path: str) -> dict[str, dict]:
    source = Path(path)
    if not source.is_file():
        return {}
    data = json.loads(source.read_text(encoding="utf-8"))
    problems = validate_briefs(data)
    if problems:
        raise ValueError("reading briefs are not neutral: " + "; ".join(problems[:5]))
    return {key: value for key, value in data.items() if not key.startswith("_")}


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False, dtype=str).fillna("")


def save_csv(df: pd.DataFrame, path: str) -> None:
    target = Path(path)
    if target.exists():
        shutil.copy2(target, target.with_suffix(".csv.bak"))
    df.to_csv(target, index=False, encoding="utf-8-sig")


def highlight(text: str, surface: str) -> str:
    if surface and surface in text:
        text = text.replace(surface, f"<mark>{surface}</mark>")
    return text


def option_frame(details: str, kind: str) -> pd.DataFrame:
    try:
        parsed = json.loads(details or "{}")
    except json.JSONDecodeError:
        return pd.DataFrame()
    return pd.DataFrame(parsed.get(kind, []))


def _lands_on_word(text: str, option: dict, offset: int) -> bool:
    start, end = option["char_start"] - offset, option["char_end"] - offset
    if not 0 <= start < end <= len(text):
        return False
    word = normalize_for_search(text[start:end])
    stem = (option.get("currency") or option.get("unit") or "")[:4]
    return bool(word) and (not stem or stem in word)


def window_offset(candidate: dict) -> int | None:
    """Where evidence_text starts inside the block, so options can be located.

    The option offsets are in the block's *surface* coordinates while
    grain_char_start is in *normalised* coordinates, so subtracting one from the
    other lands on the wrong words. Anchoring instead on
    grain_surface_char_start_debug and testing every occurrence of the grain form
    resolves it exactly — verified on all 150 rows of the canonical sample.
    Returns None rather than guessing, so a doubtful row falls back to the table.
    """
    text = candidate.get("evidence_text", "")
    form = candidate.get("grain_form", "")
    anchor = candidate.get("grain_surface_char_start_debug")
    if not text or not form or anchor is None:
        return None
    options = list(candidate.get("money_options") or []) + list(candidate.get("qty_options") or [])
    position = text.find(form)
    while position != -1:
        offset = anchor - position
        if all(_lands_on_word(text, option, offset) for option in options):
            return offset
        position = text.find(form, position + 1)
    return None


def option_spans(candidate: dict, offset: int) -> list[tuple[int, int, str, str]]:
    spans = [(option["char_start"] - offset, option["char_end"] - offset,
              option["option_id"], kind)
             for kind, key in (("money", "money_options"), ("qty", "qty_options"))
             for option in candidate.get(key) or []]
    grain = candidate.get("grain_surface_char_start_debug")
    if grain is not None:
        spans.append((grain - offset,
                      candidate.get("grain_surface_char_end_debug", grain) - offset, "", "grain"))
    return sorted(spans)


def marked_evidence(candidate: dict, offset: int) -> str:
    """The excerpt with every candidate labelled where it actually stands.

    Without this the selectboxes offer bare ids: two money options reading
    `m0`/`m1` can be a metal qualifier and its denomination (χαλκοῦ δραχμὰς),
    indistinguishable in the table because both carry the same value.
    """
    text = candidate["evidence_text"]
    pieces, cursor = [], 0
    for start, end, option_id, kind in option_spans(candidate, offset):
        if start < cursor or not 0 <= start < end <= len(text):
            continue  # overlapping spans: label the first, leave the rest plain
        label = f"<sup>{html.escape(option_id)}</sup>" if option_id else ""
        pieces.append(html.escape(text[cursor:start]))
        pieces.append(f'<span class="opt {kind}">{label}{html.escape(text[start:end])}</span>')
        cursor = end
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


def span_problem(span: str, evidence: str, option_ids: list[str] | None = None) -> str:
    """Why an evidence span is not in the excerpt, in terms the annotator can act on.

    The bare "not found" is unhelpful, because the two ways to get it wrong are
    invisible on screen: the option labels are part of the rendered text, and
    `zzlbzz` is a real token that a quotation crossing a line break has to keep.
    Returns "" when the span is fine (an empty span is fine here; the save gate
    is what requires one).

    Only the ids this row actually offers are stripped, longest first. A pattern
    like `[mq]\\d+` would eat the text's own digit out of "m23 τριώβολον", where
    the label is m2 and the 3 belongs to the amount.
    """
    if not span.strip():
        return ""
    haystack = normalize_for_search(evidence)
    if normalize_for_search(span) in haystack:
        return ""
    pattern = ("|".join(re.escape(i) for i in sorted(option_ids or [], key=len, reverse=True))
               or OPTION_LABEL_RE.pattern)
    if normalize_for_search(re.sub(rf"(?<!\w)(?:{pattern})", "", span)) in haystack:
        return ("The span carries option labels copied from the excerpt (m0, q1 …). "
                "They are display only — remove them and it matches.")
    without = lambda text: " ".join(w for w in text.split() if w != "zzlbzz")
    if without(normalize_for_search(span)) in without(haystack):
        return ("The span crosses a line break. `zzlbzz` is a token of the text, "
                "not decoration, so it has to stay inside the quotation.")
    return ("That span is not in the excerpt. The same check auto-rejects model "
            "output, so the gold has to satisfy it too.")


def mechanical_flags(candidate: dict) -> list[str]:
    """Known extraction failures, stated as facts about the data.

    Deliberately never advice: these say what the scanner did, not what to pick.
    """
    flags: list[str] = []
    for option in candidate.get("qty_options", []) + candidate.get("money_options", []):
        if option.get("inference_rule"):
            flags.append(f"{option['option_id']}: {option['inference_rule']}; denomination is unstated, not assumed.")
        if option.get("ownership_ambiguous"):
            flags.append(f"{option['option_id']}: ambiguous numeral ownership; no amount assigned.")
        if option.get("multiplier", 1) != 1:
            flags.append(f"{option['option_id']}: written {option['written_value']} × {option['multiplier']}.")
    money = list(candidate.get("money_options") or [])

    missing = [o["option_id"] for o in money if o.get("value") in (None, "")]
    if missing:
        flags.append(f"{', '.join(missing)}: a currency word with no number attached.")

    for first, second in zip(money, money[1:]):
        gap = abs(int(second.get("token_index") or 0) - int(first.get("token_index") or 0))
        if first.get("value") is not None and first.get("value") == second.get("value") and gap <= 2:
            flags.append(
                f"{first['option_id']} and {second['option_id']} are adjacent and share the "
                f"value {first['value']} — one expression may have been split into two options.")

    if normalize_for_search(candidate.get("grain_form", "")) in GRAIN_STEMS:
        flags.append("The grain form is a bare stem: a line break cut the word (issue #027).")

    for label, key in (("crosses a block or column boundary", "crosses_block_boundary"),
                       ("the same number was also read as a quantity",
                        "flag_qty_price_value_collision")):
        hits = [o["option_id"] for o in money if o.get(key)]
        if hits:
            flags.append(f"{', '.join(hits)}: {label}.")
    return flags


def render_brief(brief: dict) -> None:
    """Show the philological reading aid. It carries no annotation decision."""
    with st.expander("Reading brief — what the text says (no judgement in here)", expanded=True):
        if brief.get("document"):
            st.markdown(f"**Document.** {brief['document']}")
        if brief.get("translation"):
            st.markdown(f"**Translation.** {brief['translation']}")
        options = brief.get("options") or {}
        if options:
            st.markdown("**Where each option sits**")
            st.table(pd.DataFrame({"option": list(options), "in the text": list(options.values())}))
        for caution in brief.get("cautions") or []:
            st.caption(f"⚠ {caution}")


def pending_indices(df: pd.DataFrame, mode: str) -> list[int]:
    label = df.canonical_label.astype(str).str.strip()
    if mode == "Pass 1 remaining":
        return df.index[label == ""].tolist()
    if mode == "Pass 2 remaining":
        need = (label == "PRICE") & (
            df.money_option.astype(str).str.strip().eq("")
            | df.evidence_span.astype(str).str.strip().eq("")
        )
        return df.index[need].tolist()
    return df.index.tolist()


def main() -> None:
    csv_path = resolve_csv()
    st.set_page_config(page_title="Canonical gold annotation", layout="wide")
    st.markdown(GREEK_CSS, unsafe_allow_html=True)

    if "df" not in st.session_state:
        st.session_state.df = load_csv(csv_path)
        st.session_state.candidates = load_candidates(resolve_candidates())
        st.session_state.briefs = load_briefs(resolve_briefs())
        st.session_state.cursor = 0
        st.session_state.started = time.time()
    df: pd.DataFrame = st.session_state.df

    label = df.canonical_label.astype(str).str.strip()
    done_pass1 = int((label != "").sum())
    price_rows = int((label == "PRICE").sum())
    done_pass2 = int(((label == "PRICE") & df.money_option.astype(str).str.strip().ne("")
                      & df.evidence_span.astype(str).str.strip().ne("")).sum())

    with st.sidebar:
        st.metric("Pass 1", f"{done_pass1} / {len(df)}")
        st.progress(done_pass1 / len(df) if len(df) else 0.0)
        st.metric("Pass 2", f"{done_pass2} / {price_rows}")
        st.progress(done_pass2 / price_rows if price_rows else 0.0)
        st.caption(f"PRICE {price_rows} / NO_PRICE {int((label == 'NO_PRICE').sum())}")
        mode = st.radio("Show", ["Pass 1 remaining", "Pass 2 remaining", "All rows"])
        st.divider()
        st.caption("Annotate before running the model. Seeing a prediction first "
                   "biases the label and invalidates the comparison.")

    pending = pending_indices(df, mode)
    if not pending:
        st.success(f"Nothing left in '{mode}'.")
        st.stop()
    st.session_state.cursor = max(0, min(st.session_state.cursor, len(pending) - 1))
    current = pending[st.session_state.cursor]
    row = df.loc[current]

    head = st.columns([1, 1, 4, 2])
    if head[0].button("Prev", width="stretch") and st.session_state.cursor > 0:
        st.session_state.cursor -= 1
        st.rerun()
    if head[1].button("Next", width="stretch") and st.session_state.cursor < len(pending) - 1:
        st.session_state.cursor += 1
        st.rerun()
    head[2].markdown(f"**{row.sample_id}** &nbsp; `{row.doc_id}` &nbsp; "
                     f"stratum **{row.sampling_stratum}** &nbsp; "
                     f"({st.session_state.cursor + 1} of {len(pending)})")
    head[3].caption(f"{row.place or 'place unknown'} / {row.date_not_before}-{row.date_not_after}")

    candidate = st.session_state.candidates.get(str(row.window_id).strip())
    offset = window_offset(candidate) if candidate else None
    if offset is None:
        body = highlight(row.evidence_text, row.grain_surface)
    else:
        body = marked_evidence(candidate, offset)
    st.markdown(f"<div class='evidence'>{body}</div>", unsafe_allow_html=True)
    if offset is None:
        st.caption("Option positions could not be resolved for this row — use the tables below.")
    st.caption(f"grain: {row.grain_surface} | rule-based guess: linkage={row.rb_linkage_strength}, "
               f"tags={row.rb_context_tags or '-'}, priceword_near={row.rb_priceword_near} "
               "(the pipeline's guess, not ground truth)")
    for flag in (mechanical_flags(candidate) if candidate else []):
        st.warning(flag, icon="🔧")

    brief = st.session_state.briefs.get(str(row.sample_id).strip())
    if brief:
        render_brief(brief)

    st.divider()
    st.subheader("Pass 1 - is there a grain valuation in this window?")
    st.caption("Include any grain quantity tied to money in a way that could express its value. "
               "Institutional setting never excludes: tag tax commutation, requisition, valuation, "
               "penalty and loan rather than rejecting them. Exclude only when the money prices a "
               "service or another commodity, or when nothing is valued in money.")

    p1 = st.columns([2, 2, 2, 2])
    label_value = p1[0].radio("Label", LABEL_OPTIONS,
                              index=LABEL_OPTIONS.index(row.canonical_label)
                              if row.canonical_label in LABEL_OPTIONS else 0,
                              key=f"label_{current}", horizontal=True)
    confidence = p1[1].selectbox("Confidence", CONFIDENCE_OPTIONS,
                                 index=CONFIDENCE_OPTIONS.index(row.annotator_confidence)
                                 if row.annotator_confidence in CONFIDENCE_OPTIONS else 0,
                                 key=f"conf_{current}",
                                 help="Borderline cases are recorded as low, never dropped.")
    exclusion = ""
    observations = row.n_observations
    if label_value == "NO_PRICE":
        exclusion = p1[2].selectbox("Exclusion reason", EXCLUSION_OPTIONS,
                                    index=EXCLUSION_OPTIONS.index(row.exclusion_reason)
                                    if row.exclusion_reason in EXCLUSION_OPTIONS else 0,
                                    key=f"excl_{current}")
        observations = "0"
        p1[3].caption("n_observations fixed at 0")
    elif label_value == "PRICE":
        default_observations = (int(row.n_observations)
                                if str(row.n_observations).isdigit() and int(row.n_observations) >= 1
                                else 1)
        observations = str(p1[2].number_input(
            "Observations in this window", min_value=1, max_value=50, value=default_observations,
            key=f"nobs_{current}",
            help="How many distinct grain-money pairings this window asserts. "
                 "Only the primary one is filled in below."))
        p1[3].caption(f"{row.n_money_options} money / {row.n_qty_options} quantity candidates")

    values = {
        "canonical_label": label_value, "annotator_confidence": confidence,
        "exclusion_reason": exclusion, "n_observations": observations,
    }

    if label_value == "PRICE":
        st.divider()
        st.subheader("Pass 2 - the primary observation")
        # Definition section 7, "Which observation is the primary one". Stated
        # here because the choice is made here, and because a window with six
        # observations offers no hint about which one the file wants.
        st.caption(
            "Primary = the pairing the text most explicitly calls a valuation. "
            "First choice: a price word governs it (τιμή, ἀνά, ὡς τῆς ἀρτάβης / "
            "ἑκάστης, πρὸς χαλκόν, ἀγοράζω / πιπράσκω / ὠνέομαι) and the money "
            "option carries a value — a line break inside the price word does not "
            "weaken it. Failing that, a quantity and an amount standing in one "
            "entry as a unit rate. Failing that, a total the entry's quantity "
            "divides. Text order only breaks ties between equally explicit pairs.")
        tabs = st.tabs(["Money candidates", "Quantity candidates"])
        with tabs[0]:
            st.dataframe(option_frame(row.option_details_json, "money"),
                         width="stretch", hide_index=True)
        with tabs[1]:
            st.dataframe(option_frame(row.option_details_json, "quantity"),
                         width="stretch", hide_index=True)

        mchoices = [v for v in str(row.money_option_choices).split(";") if v]
        qchoices = [v for v in str(row.quantity_option_choices).split(";") if v]
        o1 = st.columns(3)
        values["money_option"] = o1[0].selectbox(
            "Money option", [""] + mchoices,
            index=([""] + mchoices).index(row.money_option) if row.money_option in mchoices else 0,
            key=f"mopt_{current}",
            help="Leave empty when the window states a price that no candidate "
                 "covers — an account writing the sum as a bare numeral, with the "
                 "currency word never restored (issue #034). Never select a "
                 "nearby amount just to fill the field.")
        values["quantity_option"] = o1[1].selectbox(
            "Quantity option", [""] + qchoices,
            index=([""] + qchoices).index(row.quantity_option) if row.quantity_option in qchoices else 0,
            key=f"qopt_{current}")
        # Human_Commodity, not `commodity`: materialize reads this one, and
        # commodity_class is derived from it. `commodity`, `commodity_class` and
        # `linkage_strength` used to be asked for here and were read by nothing —
        # the last two are recomputed downstream from token positions.
        values["Human_Commodity"] = o1[2].selectbox(
            "Commodity", COMMODITY_OPTIONS,
            index=COMMODITY_OPTIONS.index(row.Human_Commodity)
            if row.Human_Commodity in COMMODITY_OPTIONS else 0,
            key=f"comm_{current}")

        o2 = st.columns([3, 2])
        existing_tags = [t.strip() for t in str(row.final_context_tags).replace(";", ",").split(",")
                         if t.strip()]
        values["final_context_tags"] = ";".join(o2[0].multiselect(
            "Context tags", CONTEXT_OPTIONS,
            default=[t for t in existing_tags if t in CONTEXT_OPTIONS],
            key=f"tags_{current}",
            help="The typology is deliberately not frozen yet (issue #018). "
                 "Use the note field for cases these eight do not fit."))
        values["rate_basis"] = o2[1].selectbox(
            "Rate basis", RATE_BASIS_OPTIONS,
            help="A stated rate ('δραχμὰς N the artaba') is already the "
                 "per-artaba figure; a derived one is total divided by quantity. "
                 "Only the first is normalised without a divisor.",
            index=RATE_BASIS_OPTIONS.index(row.rate_basis)
            if row.rate_basis in RATE_BASIS_OPTIONS else RATE_BASIS_OPTIONS.index("unknown"),
            key=f"rate_{current}")

        values["context_note"] = st.text_input(
            "Context note (free text, feeds the typology work)", value=row.context_note,
            key=f"note_{current}",
            help="Free text, with a few machine-read conventions: full_amount=NdrMob when a "
                 "compound sum was split across options (#029), stated_unit_price=Ndr when the "
                 "text spells out a rate no option holds (#030), and "
                 "exclude_from_series=<reason> when the option's figure is wrong at source — "
                 "that one keeps the row out of the analysable series and into "
                 "excluded_from_public_analysis.jsonl (definition §5).")
        values["evidence_span"] = st.text_area(
            "Evidence span - copy the exact words from the excerpt above", value=row.evidence_span,
            key=f"span_{current}", height=70)
        problem = span_problem(values["evidence_span"], row.evidence_text, mchoices + qchoices)
        if problem:
            st.error(problem)

        with st.expander("Raw field values (Human_*)"):
            h = st.columns(5)
            values["Human_Commodity"] = h[0].text_input("Commodity", value=row.Human_Commodity,
                                                        key=f"hc_{current}")
            values["Human_Qty"] = h[1].text_input("Quantity", value=row.Human_Qty, key=f"hq_{current}")
            values["Human_Unit"] = h[2].text_input("Unit", value=row.Human_Unit, key=f"hu_{current}")
            values["Human_Price"] = h[3].text_input("Price", value=row.Human_Price, key=f"hp_{current}")
            values["Human_Currency"] = h[4].text_input("Currency", value=row.Human_Currency,
                                                       key=f"hcur_{current}")

    values["Human_Notes"] = st.text_input("Notes", value=row.Human_Notes, key=f"notes_{current}")

    st.divider()
    actions = st.columns([1, 1, 4])
    if actions[0].button("Save", type="primary", width="stretch"):
        if not values["canonical_label"]:
            st.warning("Pick a label first.")
        elif not values["annotator_confidence"]:
            st.warning("Pick a confidence level first.")
        elif values["canonical_label"] == "NO_PRICE" and not values["exclusion_reason"]:
            st.warning("NO_PRICE needs an exclusion reason.")
        else:
            elapsed = round(time.time() - st.session_state.started, 1)
            for column, value in values.items():
                if column not in df.columns:
                    df[column] = ""
                df.at[current, column] = value
            df.at[current, "Annotation_Seconds"] = str(elapsed)
            save_csv(df, csv_path)
            st.session_state.started = time.time()
            st.session_state.cursor = min(st.session_state.cursor + 1, len(pending) - 1)
            st.rerun()
    if actions[1].button("Skip", width="stretch"):
        st.session_state.started = time.time()
        st.session_state.cursor = min(st.session_state.cursor + 1, len(pending) - 1)
        st.rerun()


if __name__ == "__main__":
    main()

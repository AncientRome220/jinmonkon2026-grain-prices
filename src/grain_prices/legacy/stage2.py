#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Stage 2: grain price extractor and linker — v12 (classify-then-extract)

Architecture change from v11b:
  - Classification gate: each grain mention's context is classified before pairing.
    Negative contexts (tax, admin, ration, loan, transport, seed) are rejected.
  - Signal-first scoring: transactional evidence (τιμή, sale verbs, rate constructions)
    is required, not just a bonus. No signal → score capped at 20.
  - Grain-centered extraction: for each grain mention, find the nearest price/qty
    instead of generating all (G×P×Q) combinations.

CLI arguments
-------------
Required:
  --candidates FILE      Stage 1 candidates CSV
  --ddb-dir DIR          Path to DDB EpiDoc XML directory

Output:
  --out FILE             Linked CSV output (default: outputs/legacy/extracted_price_mentions.csv)
  --out-mentions FILE    Optional block-level debug CSV (default: disabled)
  --out-rejected FILE    Optional rejected-mentions CSV for auditing (default: disabled)

Extraction tuning:
  --scan-commentary      Also scan <div type='commentary'> blocks
  --require-unit         Require a unit term for each grain hit
  --window-tokens N      Token window around each grain hit, used for BOTH
                         pairing and context scanning (default: 80)

Processing:
  --workers N            Number of worker processes; 0 = cpu_count (default: 0)
  --chunksize N          ProcessPoolExecutor map chunk size (default: 10)
  --max-docs N           Process only first N documents; 0 = all (default: 0)
  --encoding ENC         CSV encoding (default: utf-8-sig)
  --debug                Enable verbose (DEBUG-level) logging

Usage examples
--------------
Standard extraction:
  python src/pipeline/2_extract_prices.py ^
    --candidates data/candidate_documents.csv ^
    --ddb-dir "<DDB_DIR>" ^
    --out outputs/legacy/extracted_price_mentions.csv ^
    --global-assign --workers 8

With rejected-mentions audit log:
  python src/pipeline/2_extract_prices.py ^
    --candidates data/candidate_documents.csv ^
    --ddb-dir "<DDB_DIR>" ^
    --out outputs/legacy/extracted_price_mentions.csv ^
    --out-rejected outputs/legacy/rejected_mentions.csv ^
    --global-assign

Notes
-----
- 𐅵 is treated as a fraction (1/2), not as currency evidence.
- Line crossings incur a mild penalty but do not block matching.
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import csv
import logging
import math
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from lxml import etree

from grain_prices.pipeline_shared import (
    NS,
    GRAIN_RE,
    UNIT_RE,
    MONEY_RE,
    PRICEWORD_RE,
    GRAIN_FALSE_STEMS,
    UNIT_CORE,
    CUR_CORE,
    TAX_RE,
    ADMIN_RE,
    TRANSPORT_FEE_RE,
    RATION_SALARY_RE,
    LOAN_RE,
    SEED_RE,
    ACCOUNTING_RE,
    PRICEWORD_EXTENDED_RE,
    RECEIPT_PRICE_RE,
    SALE_VERB_RE,
    RATE_CONSTRUCTION_RE,
    PER_AROURA_RE,
    NEGATIVE_PATTERNS,
    POSITIVE_PATTERNS,
    scan_context_cues,
    context_tags,
    is_hard_exclusion,
    linkage_strength,
    EXPLICIT_LINK_CUES,
    normalize_for_search,
)
from grain_prices.preprocessing.parse_xml import (
    extract_text_from_block as shared_extract_text_from_block,
)


def safe_string(node) -> str:
    if node is None:
        return ""
    try:
        return node.xpath("string()").strip()
    except Exception:
        return ""


# -----------------------------
# XML traversal
# -----------------------------


def iter_blocks(div_node: etree._Element) -> Iterable[Tuple[str, etree._Element]]:
    blocks = div_node.xpath(".//tei:ab | .//tei:p | .//tei:seg | .//tei:l", namespaces=NS)
    for b in blocks:
        yield etree.QName(b).localname, b


def has_num_tag(block: etree._Element) -> Tuple[bool, List[str], List[str]]:
    """
    STRICT requirement: at least one <num value="..."> inside the block.
    Returns: (has_any, values, texts)
    """
    vals: List[str] = []
    txts: List[str] = []
    nodes = block.xpath(".//tei:num[@value]", namespaces=NS)
    if not nodes:
        return False, vals, txts

    for n in nodes:
        v = (n.get("value") or "").strip()
        if v:
            vals.append(v)
        t = safe_string(n)
        if t:
            txts.append(t)

    return (len(vals) > 0), vals, txts


LB_MARK = "zzlbzz"
CB_MARK = "zzcbzz"
PB_MARK = "zzpbzz"


def block_text_with_num_values(elem: etree._Element) -> str:
    """
    Convert <num value="25">κε</num> -> "25" in the output string, injecting
    distinct boundary sentinels so scoring can apply a mild cross-line penalty
    (no hard filtering).

    Delegates to the shared extractor. This used to be a private copy, which
    meant it kept splitting words across <expan>/<ex> markup and discarding the
    currency word inside <num value="3">τριώβολον</num> long after those bugs
    were fixed in the shared version.
    """
    return shared_extract_text_from_block(
        elem,
        lb_mark=f" {LB_MARK} ",
        cb_mark=f" {CB_MARK} ",
        pb_mark=f" {PB_MARK} ",
    )


# -----------------------------
# Fractions
# -----------------------------


def fmt_fraction(fr: Fraction) -> str:
    if fr.denominator == 1:
        return str(fr.numerator)
    v = float(fr)
    return f"{v:.6f}".rstrip("0").rstrip(".")


FRAC_TOKEN_RE = re.compile(r"(?P<n>\d+)\s*/\s*(?P<d>\d+)")


def parse_int_and_fracs(int_str: Optional[str], fracs_str: Optional[str]) -> str:
    total = Fraction(0, 1)
    if int_str is not None and int_str != "":
        try:
            total += Fraction(int(int_str), 1)
        except Exception:
            pass

    if fracs_str:
        for m in FRAC_TOKEN_RE.finditer(fracs_str):
            n = int(m.group("n"))
            d = int(m.group("d"))
            if d != 0:
                total += Fraction(n, d)

    return fmt_fraction(total)


# -----------------------------
# Patterns
# -----------------------------


def compile_patterns() -> Dict[str, re.Pattern]:  # noqa: C901
    """
    Core grain/unit/money/priceword patterns come from pipeline_shared.
    Extraction-specific patterns (qty, price) are local.
    Classification patterns are imported from pipeline_shared.
    """
    grain = GRAIN_RE
    unit = UNIT_RE
    money = MONEY_RE
    priceword = PRICEWORD_RE

    # Derived from the shared stem lists — these were previously hardcoded here
    # and had drifted: the currency copy was missing the compound-obol stems
    # (so "4 τετρώβολον" produced no price event) and spelled ἀσσ with a
    # breathing mark, which can never match accent-stripped text.
    unit_core = UNIT_CORE
    cur_core = CUR_CORE

    # Amounts in papyri are written as an optional integer followed by a SUM of
    # unit fractions ("ἀρτάβης 1/2 1/12" = 7/12). Two fixes over the naive form:
    #   (?!/)  - without it, "1/2" is read as the integer 1, since <int> matches
    #            the numerator and <fracs> then fails and stays empty.
    #   the (?:\W+\d+/\d+)* tail on frac_only - previously only the FIRST
    #            fraction was captured, truncating 1/2 1/12 to 1/2.
    # parse_int_and_fracs() already sums an arbitrary run of fractions.
    int_plus_fracs = r"(?P<int>\d+)(?!/)(?P<fracs>(?:\W+(?:\d+/\d+))*)"
    frac_only = r"(?P<fracs_only>\d+/\d+(?:\W+\d+/\d+)*)"

    qty_num_unit = re.compile(rf"{int_plus_fracs}\W+(?P<unit>{unit_core})", re.UNICODE)
    qty_unit_num = re.compile(rf"(?P<unit>{unit_core})\W+{int_plus_fracs}", re.UNICODE)
    qty_frac_unit = re.compile(rf"{frac_only}\W+(?P<unit>{unit_core})", re.UNICODE)
    qty_unit_frac = re.compile(rf"(?P<unit>{unit_core})\W+{frac_only}", re.UNICODE)

    price_num_cur = re.compile(rf"{int_plus_fracs}\W+(?P<cur>{cur_core})", re.UNICODE)
    price_cur_num = re.compile(rf"(?P<cur>{cur_core})\W+{int_plus_fracs}", re.UNICODE)
    price_frac_cur = re.compile(rf"{frac_only}\W+(?P<cur>{cur_core})", re.UNICODE)
    price_cur_frac = re.compile(rf"(?P<cur>{cur_core})\W+{frac_only}", re.UNICODE)

    # Bug 3a: admin compound words whose grain stem is embedded inside the compound.
    grain_admin_compound = re.compile(
        r"\b(σιτολογ|σιτομετρ|σιτοσπορ|σιτοποι)\w*", re.UNICODE
    )
    # Bug 3b: standalone admin/fee words
    nearby_admin_word = re.compile(
        r"\b(φυλακιτικ|παραναυλ|αλεστρ|χωματικ|πρακτορ)\w*", re.UNICODE
    )

    return {
        "grain": grain,
        "unit": unit,
        "money": money,
        "priceword": priceword,
        "qty_num_unit": qty_num_unit,
        "qty_unit_num": qty_unit_num,
        "qty_frac_unit": qty_frac_unit,
        "qty_unit_frac": qty_unit_frac,
        "price_num_cur": price_num_cur,
        "price_cur_num": price_cur_num,
        "price_frac_cur": price_frac_cur,
        "price_cur_frac": price_cur_frac,
        "grain_admin_compound": grain_admin_compound,
        "nearby_admin_word": nearby_admin_word,
    }


# Normalized (accent-stripped, casefolded) forms of the κρίνω aorist passive stem.
# These share the κριθ- prefix with κριθή (barley) but are not grain words.
KRINO_BLOCKLIST = re.compile(
    r"^κριθ(εν|εντ\w*|εις|εισ\w+|ηναι|ητω|ωσ\w*|ησαν)$"
)


# -----------------------------
# Tokenization / positioning
# -----------------------------


@dataclass(frozen=True)
class Tokens:
    text: str
    tokens: List[str]
    starts: List[int]
    line_ids: List[int]

    def tok_at_char(self, charpos: int) -> int:
        lo, hi = 0, len(self.starts)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.starts[mid] <= charpos:
                lo = mid + 1
            else:
                hi = mid
        idx = lo - 1
        if idx < 0:
            return 0
        if idx >= len(self.tokens):
            return len(self.tokens) - 1
        return idx


def tokenize_with_positions(s: str) -> Tokens:
    toks: List[str] = []
    starts: List[int] = []
    line_ids: List[int] = []
    line = 0
    for m in re.finditer(r"\S+", s):
        tok = m.group(0)
        toks.append(tok)
        starts.append(m.start())
        if tok == LB_MARK:
            line += 1
        line_ids.append(line)
    return Tokens(text=s, tokens=toks, starts=starts, line_ids=line_ids)


# -----------------------------
# Events
# -----------------------------


@dataclass(frozen=True)
class GrainEvent:
    tok: int
    form: str
    char_start: int = 0
    char_end: int = 0
    # Set only when build_events(keep_all=True): the exclusion rules become
    # recorded features instead of silently dropping the mention, so a later
    # adjudicator can see them and their effect can be measured after the fact.
    flag_krino: bool = False
    flag_admin_compound: bool = False
    flag_false_stem: bool = False
    flag_admin_word_near: bool = False

    @property
    def excluded_by_rules(self) -> bool:
        return (self.flag_krino or self.flag_admin_compound
                or self.flag_false_stem or self.flag_admin_word_near)


@dataclass(frozen=True)
class QtyEvent:
    tok: int
    value: str
    unit: str
    char_start: int = 0
    char_end: int = 0


@dataclass(frozen=True)
class PriceEvent:
    tok: int
    value: str
    cur: str
    char_start: int = 0
    char_end: int = 0
    flag_qty_value_collision: bool = False


@dataclass(frozen=True)
class PricewordEvent:
    tok: int
    form: str


# -----------------------------
# Scoring v2 (signal-first)
# -----------------------------





# -----------------------------
# Output rows
# -----------------------------


@dataclass
class MentionRow:
    DDB_ID: str
    Mention_ID: int
    Block_Tag: str

    Title: str
    Place: str
    Date_Text: str
    Date_When: str
    Date_NotBefore: str
    Date_NotAfter: str

    Grain_Hits: str
    Unit_Hits: str
    Money_Hits: str
    Priceword_Hit: str

    Number_Hit: str
    Number_Type: str
    Num_Tag_Values: str
    Num_Tag_Texts: str

    Quantities: str
    Prices: str

    Raw_Block: str
    Value_Block: str


@dataclass
class LinkedRow:
    """
    One grain↔price pairing.

    v13.1: `Score` is gone. It summed proximity, signal strength, a quality
    factor and an ambiguity factor into one uninterpretable 0-100 number that,
    at every threshold the pipeline actually shipped, excluded zero rows.
    `Linkage_Strength` (definition §2) replaces it as the strictness dial, and
    the raw components stay as their own columns.

    `Candidate_Rank` / `Is_Primary` / `Ambiguous` / `AltScore` are also gone:
    the pairing step returns exactly one candidate, so they were constant in
    every row ever produced. Multiple candidates live in Stage 2C.
    """
    DDB_ID: str
    Mention_ID: int
    Grain_Index: int
    Block_Tag: str

    Title: str
    Place: str
    Date_Text: str
    Date_When: str
    Date_NotBefore: str
    Date_NotAfter: str

    Grain_Form: str
    Qty_Value: str
    Qty_Unit: str
    Price_Value: str
    Price_Cur: str

    # §2 strictness dial, and the raw features behind it
    Linkage_Strength: str
    Dist_GP: int
    Dist_GQ: int
    Span_Toks: int
    Lines_Crossed: int
    Priceword_Near: str
    Has_Boundary: str

    Context_Window: str

    # §3 institutional context — multi-label interpretation, plus the raw
    # machine-checkable evidence it was derived from.
    Context_Tags: str
    Cue_Flags: str


@dataclass
class RejectedRow:
    """Row emitted for rejected grain mentions (for auditing)."""
    DDB_ID: str
    Mention_ID: int
    Grain_Form: str
    Context_Type: str
    Signal_Strength: float
    Neg_Signals: str
    Rejection_Reason: str
    Context_Window: str


# -----------------------------
# Candidate model + ranking
# -----------------------------


@dataclass(frozen=True)
class Candidate:
    grain: GrainEvent
    price: PriceEvent
    qty: Optional[QtyEvent]
    d_gp: int
    d_gq: int
    span: int
    lines_crossed: int
    has_priceword_near: bool
    has_boundary: bool
    linkage: str
    context_tags: str
    cue_flags: str


def nearest_tok_distance(tok: int, events: List[int]) -> int:
    if not events:
        return int(1e9)
    return min(abs(tok - t) for t in events)


def make_context_window(value_tokens: List[str], lo_tok: int, hi_tok: int) -> str:
    out: List[str] = []
    for t in value_tokens[lo_tok:hi_tok]:
        if t == LB_MARK:
            out.append("\n")
        elif t in (CB_MARK, PB_MARK):
            out.append("\n")
        else:
            out.append(t)
    s = " ".join(out)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


# -----------------------------
# Event building
# -----------------------------


def build_events(
    pats: Dict[str, re.Pattern],
    value_norm: str,
    tok_norm: Tokens,
    keep_all: bool = False,
) -> Tuple[List[GrainEvent], List[QtyEvent], List[PriceEvent], List[PricewordEvent]]:
    """
    Extract grain / quantity / price / priceword events from a block.

    With keep_all=False (default, Stage 2 behaviour) the exclusion rules drop
    events outright. With keep_all=True nothing is dropped: each rule instead
    sets a flag on the event. That is what the candidate enumerator needs — a
    rule that silently deletes an event also deletes it from the view of the
    LLM and the human reviewer downstream, so the mistake becomes unrecoverable
    and unmeasurable.
    """
    grains: List[GrainEvent] = []
    qtys: List[QtyEvent] = []
    prices: List[PriceEvent] = []
    pws: List[PricewordEvent] = []

    # Bug 3a: character spans of admin compound words containing a grain stem.
    admin_compound_spans = [
        (m.start(), m.end())
        for m in pats["grain_admin_compound"].finditer(value_norm)
    ]

    # Bug 3b: token positions of standalone admin/fee words.
    nearby_admin_toks = {
        tok_norm.tok_at_char(m.start())
        for m in pats["nearby_admin_word"].finditer(value_norm)
    }

    # Additional false-stem filtering from pipeline_shared
    false_stem_spans = [
        (m.start(), m.end())
        for m in GRAIN_FALSE_STEMS.finditer(value_norm)
    ]

    for m in pats["grain"].finditer(value_norm):
        form = m.group(0)
        t = tok_norm.tok_at_char(m.start())

        # Bug 1: κρίνω aorist passive forms share the κριθ- stem with κριθή
        f_krino = bool(KRINO_BLOCKLIST.match(form))
        # Bug 3a: grain stem sitting inside an administrative compound word
        f_admin_compound = any(s <= m.start() < e for s, e in admin_compound_spans)
        # Grain stem inside a known false-stem match
        f_false_stem = any(s <= m.start() < e for s, e in false_stem_spans)
        # Bug 3b: an admin/fee word within 10 tokens
        f_admin_near = any(abs(t - at) <= 10 for at in nearby_admin_toks)

        if not keep_all and (f_krino or f_admin_compound
                             or f_false_stem or f_admin_near):
            continue

        grains.append(GrainEvent(
            tok=t, form=form, char_start=m.start(), char_end=m.end(),
            flag_krino=f_krino,
            flag_admin_compound=f_admin_compound,
            flag_false_stem=f_false_stem,
            flag_admin_word_near=f_admin_near,
        ))

    for m in pats["priceword"].finditer(value_norm):
        t = tok_norm.tok_at_char(m.start())
        pws.append(PricewordEvent(tok=t, form=m.group(0)))

    def add_qty(m: re.Match, n: str, u: str):
        qtys.append(QtyEvent(tok=tok_norm.tok_at_char(m.start()), value=n, unit=u,
                             char_start=m.start(), char_end=m.end()))

    for key in ("qty_num_unit", "qty_unit_num"):
        for m in pats[key].finditer(value_norm):
            n = parse_int_and_fracs(m.groupdict().get("int"), m.groupdict().get("fracs"))
            add_qty(m, n, m.group("unit"))
    for key in ("qty_frac_unit", "qty_unit_frac"):
        for m in pats[key].finditer(value_norm):
            n = parse_int_and_fracs(None, m.group("fracs_only"))
            add_qty(m, n, m.group("unit"))

    def add_price(m: re.Match, n: str, c: str):
        prices.append(PriceEvent(tok=tok_norm.tok_at_char(m.start()), value=n, cur=c,
                                 char_start=m.start(), char_end=m.end()))

    for key in ("price_num_cur", "price_cur_num"):
        for m in pats[key].finditer(value_norm):
            n = parse_int_and_fracs(m.groupdict().get("int"), m.groupdict().get("fracs"))
            add_price(m, n, m.group("cur"))
    for key in ("price_frac_cur", "price_cur_frac"):
        for m in pats[key].finditer(value_norm):
            n = parse_int_and_fracs(None, m.group("fracs_only"))
            add_price(m, n, m.group("cur"))

    # Bug 2: a "price" whose value coincides with a nearby quantity is usually
    # the quantity re-matched under a currency pattern.
    qty_val_toks = {(q.value, q.tok) for q in qtys}

    def collides(p: PriceEvent) -> bool:
        return any(p.value == qv and abs(p.tok - qt) <= 3 for qv, qt in qty_val_toks)

    if keep_all:
        prices = [
            PriceEvent(tok=p.tok, value=p.value, cur=p.cur,
                       char_start=p.char_start, char_end=p.char_end,
                       flag_qty_value_collision=collides(p))
            for p in prices
        ]
    else:
        prices = [p for p in prices if not collides(p)]

    return grains, qtys, prices, pws


# -----------------------------
# Classification gate
# -----------------------------


def scan_mention_cues(
    value_norm: str,
    grain: GrainEvent,
    tok_norm: Tokens,
    window_tokens: int,
) -> Tuple[List[str], str]:
    """
    Record the context cues around a grain mention (evidence, not a verdict).

    `window_tokens` is now passed in rather than hardcoded at 20. It used to be
    fixed at 20 while pairing searched ±80, so a price could be linked from 60
    tokens away while the context deciding its fate was read from only ±20 —
    and a mention could be thrown out on a tax word 15 tokens off without ever
    seeing the sale verb 40 tokens away.
    """
    n_toks = len(tok_norm.tokens)
    if n_toks == 0:
        return ([], "")

    lo_tok = max(0, grain.tok - window_tokens)
    hi_tok = min(n_toks - 1, grain.tok + window_tokens)
    lo_char = tok_norm.starts[lo_tok]
    hi_char = (tok_norm.starts[hi_tok] + len(tok_norm.tokens[hi_tok])
               if hi_tok < n_toks else len(value_norm))

    return scan_context_cues(
        norm_text=value_norm,
        grain_start=grain.char_start,
        grain_end=grain.char_end,
        window_chars=max(hi_char - grain.char_start,
                         grain.char_start - lo_char, 100),
    )


# -----------------------------
# Grain-centered extraction
# -----------------------------

# Boundary tokens that signal accounting list breaks
BOUNDARY_RE = re.compile(r"\b(γινονται|λοιπ|ομοιως)\w*", re.UNICODE)


def has_intervening_boundary(tok_norm: Tokens, tok_a: int, tok_b: int) -> bool:
    """Check if any accounting boundary token lies between tok_a and tok_b."""
    lo = min(tok_a, tok_b)
    hi = max(tok_a, tok_b)
    for i in range(lo + 1, min(hi, len(tok_norm.tokens))):
        if BOUNDARY_RE.match(tok_norm.tokens[i]):
            return True
    return False


def find_nearest_price_for_grain(
    grain: GrainEvent,
    prices: List[PriceEvent],
    qtys: List[QtyEvent],
    tok_norm: Tokens,
    window_tokens: int,
) -> Optional[Tuple[PriceEvent, Optional[QtyEvent], bool]]:
    """Find the nearest price and quantity for a grain mention.

    Returns (best_price, best_qty, has_boundary) or None if no price within window.
    """
    local_prices = [p for p in prices if abs(p.tok - grain.tok) <= window_tokens]
    if not local_prices:
        return None

    # Sort by distance to grain, take nearest
    local_prices.sort(key=lambda p: abs(p.tok - grain.tok))
    best_price = local_prices[0]

    # Check for boundary between grain and price
    boundary = has_intervening_boundary(tok_norm, grain.tok, best_price.tok)

    # Find nearest quantity
    local_qtys = [q for q in qtys if abs(q.tok - grain.tok) <= window_tokens]
    best_qty: Optional[QtyEvent] = None
    if local_qtys:
        local_qtys.sort(key=lambda q: abs(q.tok - grain.tok))
        best_qty = local_qtys[0]
        # Check boundary between grain and qty too
        if has_intervening_boundary(tok_norm, grain.tok, best_qty.tok):
            best_qty = None

    return (best_price, best_qty, boundary)



# -----------------------------
# Core extraction per doc
# -----------------------------


def extract_for_doc(
    xml_path: Path,
    ddb_id: str,
    meta: Dict[str, str],
    scan_commentary: bool,
    require_unit: bool,
    window_tokens: int,
    keep_block_chars: int = 2000,
) -> Tuple[List[LinkedRow], List[MentionRow], List[RejectedRow]]:
    pats = WORKER_PATS or compile_patterns()

    linked: List[LinkedRow] = []
    mentions: List[MentionRow] = []
    rejected: List[RejectedRow] = []

    try:
        root = etree.parse(str(xml_path)).getroot()
    except Exception:
        return linked, mentions, rejected

    div_xpath = "//tei:div[@type='edition']"
    if scan_commentary:
        div_xpath += " | //tei:div[@type='commentary']"
    divs = root.xpath(div_xpath, namespaces=NS)

    mention_id = 0

    for div in divs:
        for tag, block in iter_blocks(div):
            raw = safe_string(block)
            if not raw:
                continue
            raw = " ".join(raw.split())
            norm = normalize_for_search(raw)

            # Gating
            if not pats["grain"].search(norm):
                continue

            has_num, num_vals, num_txts = has_num_tag(block)
            if not has_num:
                continue

            has_unit_sig = bool(pats["unit"].search(norm))
            if require_unit and not has_unit_sig:
                continue

            has_money_sig = bool(pats["money"].search(norm))
            has_priceword_sig = bool(pats["priceword"].search(norm))
            if not (has_money_sig or has_priceword_sig):
                continue

            grain_hits = sorted(set(m.group(0) for m in pats["grain"].finditer(norm)))
            unit_hits = sorted(set(m.group(0) for m in pats["unit"].finditer(norm)))
            money_hits = sorted(set(m.group(0) for m in pats["money"].finditer(norm)))
            priceword_hit = "yes" if has_priceword_sig else "no"

            # Value-rendered text + markers
            value_text = block_text_with_num_values(block)
            value_text = value_text.replace("𐅵", " 1/2 ")
            value_text = " ".join(value_text.split())
            value_norm = normalize_for_search(value_text)

            tok_norm = tokenize_with_positions(value_norm)
            tok_val = value_text.split()

            # Debug lists
            quantities_dbg: List[str] = []
            prices_dbg: List[str] = []
            seenq = set()
            seenp = set()

            def add_qty_dbg(num_s: str, unit_s: str):
                item = f"{num_s} {unit_s}"
                if item not in seenq:
                    seenq.add(item)
                    quantities_dbg.append(item)

            def add_price_dbg(num_s: str, cur_s: str):
                item = f"{num_s} {cur_s}"
                if item not in seenp:
                    seenp.add(item)
                    prices_dbg.append(item)

            for m in pats["qty_num_unit"].finditer(value_norm):
                n = parse_int_and_fracs(m.groupdict().get("int"), m.groupdict().get("fracs"))
                add_qty_dbg(n, m.group("unit"))
            for m in pats["qty_unit_num"].finditer(value_norm):
                n = parse_int_and_fracs(m.groupdict().get("int"), m.groupdict().get("fracs"))
                add_qty_dbg(n, m.group("unit"))
            for m in pats["qty_frac_unit"].finditer(value_norm):
                n = parse_int_and_fracs(None, m.group("fracs_only"))
                add_qty_dbg(n, m.group("unit"))
            for m in pats["qty_unit_frac"].finditer(value_norm):
                n = parse_int_and_fracs(None, m.group("fracs_only"))
                add_qty_dbg(n, m.group("unit"))

            for m in pats["price_num_cur"].finditer(value_norm):
                n = parse_int_and_fracs(m.groupdict().get("int"), m.groupdict().get("fracs"))
                add_price_dbg(n, m.group("cur"))
            for m in pats["price_cur_num"].finditer(value_norm):
                n = parse_int_and_fracs(m.groupdict().get("int"), m.groupdict().get("fracs"))
                add_price_dbg(n, m.group("cur"))
            for m in pats["price_frac_cur"].finditer(value_norm):
                n = parse_int_and_fracs(None, m.group("fracs_only"))
                add_price_dbg(n, m.group("cur"))
            for m in pats["price_cur_frac"].finditer(value_norm):
                n = parse_int_and_fracs(None, m.group("fracs_only"))
                add_price_dbg(n, m.group("cur"))

            grains, qtys, prices_ev, pws = build_events(pats, value_norm, tok_norm)

            mention_id += 1
            mentions.append(
                MentionRow(
                    DDB_ID=ddb_id,
                    Mention_ID=mention_id,
                    Block_Tag=tag,
                    Title=meta.get("Title", ""),
                    Place=meta.get("Place", ""),
                    Date_Text=meta.get("Date_Text", ""),
                    Date_When=meta.get("Date_When", ""),
                    Date_NotBefore=meta.get("Date_NotBefore", ""),
                    Date_NotAfter=meta.get("Date_NotAfter", ""),
                    Grain_Hits=";".join(grain_hits),
                    Unit_Hits=";".join(unit_hits),
                    Money_Hits=";".join(money_hits),
                    Priceword_Hit=priceword_hit,
                    Number_Hit=f"num:{num_vals[0]}" if num_vals else "",
                    Number_Type="num_tag",
                    Num_Tag_Values=";".join([v for v in num_vals if v]),
                    Num_Tag_Texts=";".join([t for t in num_txts if t]),
                    Quantities=";".join(quantities_dbg),
                    Prices=";".join(prices_dbg),
                    Raw_Block=raw[:keep_block_chars],
                    Value_Block=value_text[:keep_block_chars],
                )
            )

            if not grains or not prices_ev:
                continue

            # Priceword token positions (for Priceword_Near column)
            priceword_toks = [pw.tok for pw in pws]

            # --- Grain-centered extraction: label, do not reject ---
            # Per definition §3, fiscal / administrative / loan settings do NOT
            # exclude an observation; they are recorded as a context_tag and
            # tested empirically. Only the §6 residual set — where there is no
            # monetary valuation of the grain to recover at all — is dropped.
            ranked: Dict[int, List[Candidate]] = {}

            for gi, grain in enumerate(grains, start=1):
                cues, _cue_window = scan_mention_cues(
                    value_norm, grain, tok_norm, window_tokens
                )

                # A price event exists in this block by construction, so the
                # cash figure is present; only the unconditional §6 categories
                # (freight/milling, in-kind rations) can apply here.
                hard = is_hard_exclusion(cues, has_money=True)
                if hard:
                    lo_c = max(0, grain.tok - 8)
                    hi_c = min(len(tok_val), grain.tok + 9)
                    ctx_text = make_context_window(tok_val, lo_c, hi_c)
                    rejected.append(RejectedRow(
                        DDB_ID=ddb_id,
                        Mention_ID=mention_id,
                        Grain_Form=grain.form,
                        Context_Type=",".join(context_tags(cues)),
                        Signal_Strength=0.0,
                        Neg_Signals=",".join(cues),
                        Rejection_Reason=hard,
                        Context_Window=ctx_text[:500],
                    ))
                    continue

                # Find nearest price/qty for this grain
                result = find_nearest_price_for_grain(
                    grain, prices_ev, qtys, tok_norm, window_tokens,
                )
                if result is None:
                    continue

                best_price, best_qty, has_boundary = result

                d_gp = abs(grain.tok - best_price.tok)
                d_gq = abs(grain.tok - best_qty.tok) if best_qty else int(1e9)
                span = max(grain.tok, best_price.tok,
                           best_qty.tok if best_qty else grain.tok) - \
                       min(grain.tok, best_price.tok,
                           best_qty.tok if best_qty else grain.tok)
                lines_crossed = abs(
                    tok_norm.line_ids[grain.tok] - tok_norm.line_ids[best_price.tok]
                ) if tok_norm.line_ids else 0

                d_pw = nearest_tok_distance(best_price.tok, priceword_toks)
                has_pw_near = (d_pw <= 8)

                cand = Candidate(
                    grain=grain,
                    price=best_price,
                    qty=best_qty,
                    d_gp=d_gp,
                    d_gq=d_gq if d_gq < 1e9 else -1,
                    span=span,
                    lines_crossed=lines_crossed,
                    has_priceword_near=has_pw_near,
                    has_boundary=has_boundary,
                    linkage=linkage_strength(
                        has_explicit_cue=any(c in cues for c in EXPLICIT_LINK_CUES),
                        dist_grain_money=d_gp,
                        has_quantity=best_qty is not None,
                    ),
                    context_tags=",".join(context_tags(cues)),
                    cue_flags=",".join(cues),
                )

                ranked[gi] = [cand]

            if not ranked:
                continue

            for gi, cand_list in ranked.items():
                for c in cand_list:
                    # Context window
                    lo = min(c.grain.tok, c.price.tok, c.qty.tok if c.qty else c.grain.tok)
                    hi = max(c.grain.tok, c.price.tok, c.qty.tok if c.qty else c.grain.tok)
                    pad = 8
                    lo2 = max(0, lo - pad)
                    hi2 = min(len(tok_val), hi + pad + 1)
                    ctx = make_context_window(tok_val, lo2, hi2)

                    linked.append(
                        LinkedRow(
                            DDB_ID=ddb_id,
                            Mention_ID=mention_id,
                            Grain_Index=gi,
                            Block_Tag=tag,
                            Title=meta.get("Title", ""),
                            Place=meta.get("Place", ""),
                            Date_Text=meta.get("Date_Text", ""),
                            Date_When=meta.get("Date_When", ""),
                            Date_NotBefore=meta.get("Date_NotBefore", ""),
                            Date_NotAfter=meta.get("Date_NotAfter", ""),
                            Grain_Form=c.grain.form,
                            Qty_Value=c.qty.value if c.qty else "",
                            Qty_Unit=c.qty.unit if c.qty else "",
                            Price_Value=c.price.value,
                            Price_Cur=c.price.cur,
                            Linkage_Strength=c.linkage,
                            Dist_GP=c.d_gp,
                            Dist_GQ=c.d_gq,
                            Span_Toks=c.span,
                            Lines_Crossed=c.lines_crossed,
                            Priceword_Near="yes" if c.has_priceword_near else "no",
                            Has_Boundary="yes" if c.has_boundary else "no",
                            Context_Window=ctx[:keep_block_chars],
                            Context_Tags=c.context_tags,
                            Cue_Flags=c.cue_flags,
                        )
                    )

    return linked, mentions, rejected


# -----------------------------
# Multiprocessing worker
# -----------------------------

WORKER_PATS: Optional[Dict[str, re.Pattern]] = None
WORKER_DDB_DIR: Optional[Path] = None
WORKER_SCAN_COMMENTARY: bool = False
WORKER_REQUIRE_UNIT: bool = False
WORKER_WINDOW_TOKENS: int = 80


def _init_worker(
    ddb_dir: str,
    scan_commentary: bool,
    require_unit: bool,
    window_tokens: int,
):
    global WORKER_PATS, WORKER_DDB_DIR
    global WORKER_SCAN_COMMENTARY, WORKER_REQUIRE_UNIT
    global WORKER_WINDOW_TOKENS

    WORKER_PATS = compile_patterns()
    WORKER_DDB_DIR = Path(ddb_dir)
    WORKER_SCAN_COMMENTARY = scan_commentary
    WORKER_REQUIRE_UNIT = require_unit
    WORKER_WINDOW_TOKENS = window_tokens


def _process_one_candidate(job: Tuple[str, str, Dict[str, str]]) -> Tuple[bool, List[dict], List[dict], List[dict]]:
    """
    Worker entry point. Returns:
      (ok, linked_rows_as_dicts, mention_rows_as_dicts, rejected_rows_as_dicts)
    """
    ddb_id, rel_path, meta = job
    try:
        assert WORKER_DDB_DIR is not None
        xml_path = WORKER_DDB_DIR / rel_path
        if not xml_path.exists():
            return False, [], [], []

        linked_rows, mention_rows, rejected_rows = extract_for_doc(
            xml_path=xml_path,
            ddb_id=ddb_id,
            meta=meta,
            scan_commentary=WORKER_SCAN_COMMENTARY,
            require_unit=WORKER_REQUIRE_UNIT,
            window_tokens=WORKER_WINDOW_TOKENS,
        )
        return (True,
                [asdict(r) for r in linked_rows],
                [asdict(r) for r in mention_rows],
                [asdict(r) for r in rejected_rows])
    except Exception:
        return False, [], [], []


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 v12: grain price extractor (classify-then-extract)."
    )
    parser.add_argument("--candidates", required=True, help="Stage 1 candidates CSV")
    parser.add_argument("--ddb-dir", required=True, help="Path to DDB EpiDoc XML directory")

    parser.add_argument(
        "--out",
        default="outputs/legacy/extracted_price_mentions.csv",
        help="Output LINKED (grain<->qty<->price) CSV",
    )
    parser.add_argument(
        "--out-mentions",
        default="",
        help="Optional output block-level debug CSV (old-style mention rows)",
    )
    parser.add_argument(
        "--out-rejected",
        default="",
        help="Optional output CSV for rejected grain mentions (audit log)",
    )

    parser.add_argument("--scan-commentary", action="store_true", help="Also scan <div type='commentary'>")
    parser.add_argument("--require-unit", action="store_true", help="Require unit term (may miss some)")
    parser.add_argument("--window-tokens", type=int, default=80, help="Token window around each grain hit")

    parser.add_argument("--workers", type=int, default=0, help="Number of worker processes (0=cpu_count)")
    parser.add_argument("--chunksize", type=int, default=10, help="ProcessPoolExecutor map chunksize")
    parser.add_argument("--max-docs", type=int, default=0, help="Process only N docs (0 = all)")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding (default utf-8-sig for Excel)")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("2_extract_prices")

    cand_path = Path(args.candidates)
    ddb_dir = Path(args.ddb_dir)
    out_linked = Path(args.out)
    out_mentions = Path(args.out_mentions) if args.out_mentions else None
    out_rejected = Path(args.out_rejected) if args.out_rejected else None

    if not cand_path.exists():
        raise SystemExit(f"Candidates CSV not found: {cand_path}")
    if not ddb_dir.exists():
        raise SystemExit(f"DDB dir not found: {ddb_dir}")

    df = pd.read_csv(cand_path).fillna("")
    if not {"DDB_ID", "XML_RelPath"}.issubset(df.columns):
        raise SystemExit("Candidates CSV must contain columns: DDB_ID, XML_RelPath")

    # Build jobs
    jobs: List[Tuple[str, str, Dict[str, str]]] = []
    for _, r in df.iterrows():
        ddb_id = str(r.get("DDB_ID", "")).strip()
        rel = str(r.get("XML_RelPath", "")).strip().replace("\\", "/")
        if not ddb_id or not rel:
            continue

        meta = {
            "Title": str(r.get("Title", "")),
            "Place": str(r.get("Place", "")),
            "Date_Text": str(r.get("Date_Text", "")),
            "Date_When": str(r.get("Date_When", "")),
            "Date_NotBefore": str(r.get("Date_NotBefore", "")),
            "Date_NotAfter": str(r.get("Date_NotAfter", "")),
        }
        jobs.append((ddb_id, rel, meta))

    if args.max_docs and args.max_docs > 0:
        jobs = jobs[: args.max_docs]

    total = len(jobs)
    if total == 0:
        logger.warning("No jobs found in candidates CSV.")
        return

    # Prepare CSV writers (streaming)
    linked_fields = [f.name for f in fields(LinkedRow)]
    mention_fields = [f.name for f in fields(MentionRow)]
    rejected_fields = [f.name for f in fields(RejectedRow)]

    out_linked.parent.mkdir(parents=True, exist_ok=True)
    f_linked = out_linked.open("w", newline="", encoding=args.encoding)
    linked_writer = csv.DictWriter(f_linked, fieldnames=linked_fields)
    linked_writer.writeheader()

    f_mentions = None
    mention_writer = None
    if out_mentions is not None:
        out_mentions.parent.mkdir(parents=True, exist_ok=True)
        f_mentions = out_mentions.open("w", newline="", encoding=args.encoding)
        mention_writer = csv.DictWriter(f_mentions, fieldnames=mention_fields)
        mention_writer.writeheader()

    f_rejected = None
    rejected_writer = None
    if out_rejected is not None:
        out_rejected.parent.mkdir(parents=True, exist_ok=True)
        f_rejected = out_rejected.open("w", newline="", encoding=args.encoding)
        rejected_writer = csv.DictWriter(f_rejected, fieldnames=rejected_fields)
        rejected_writer.writeheader()

    workers = args.workers if args.workers and args.workers > 0 else None
    logger.info("Stage 2 v12 (classify-then-extract) processing %d candidate docs with workers=%s ...", total, workers or "cpu_count")

    processed = 0
    ok_docs = 0
    linked_rows_written = 0
    mention_rows_written = 0
    rejected_rows_written = 0

    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(
                str(ddb_dir),
                args.scan_commentary,
                args.require_unit,
                args.window_tokens,
            ),
        ) as ex:
            for ok, linked_dicts, mention_dicts, rejected_dicts in ex.map(_process_one_candidate, jobs, chunksize=args.chunksize):
                processed += 1
                if ok:
                    ok_docs += 1

                for d in linked_dicts:
                    linked_writer.writerow(d)
                linked_rows_written += len(linked_dicts)

                if mention_writer is not None:
                    for d in mention_dicts:
                        mention_writer.writerow(d)
                    mention_rows_written += len(mention_dicts)

                if rejected_writer is not None:
                    for d in rejected_dicts:
                        rejected_writer.writerow(d)
                    rejected_rows_written += len(rejected_dicts)

                if processed % 200 == 0:
                    logger.info(
                        "Progress: %d/%d docs; ok=%d; linked=%d; rejected=%d",
                        processed,
                        total,
                        ok_docs,
                        linked_rows_written,
                        rejected_rows_written,
                    )
    finally:
        f_linked.close()
        if f_mentions is not None:
            f_mentions.close()
        if f_rejected is not None:
            f_rejected.close()

    logger.info(
        "Done. Docs processed: %d/%d (ok=%d). Linked rows: %d. Rejected: %d. Mentions: %d.",
        processed,
        total,
        ok_docs,
        linked_rows_written,
        rejected_rows_written,
        mention_rows_written,
    )
    logger.info("Wrote: %s", out_linked.resolve())
    if out_mentions is not None:
        logger.info("Wrote mentions: %s", out_mentions.resolve())
    if out_rejected is not None:
        logger.info("Wrote rejected: %s", out_rejected.resolve())


if __name__ == "__main__":
    main()

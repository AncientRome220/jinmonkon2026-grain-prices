# -*- coding: utf-8 -*-
"""
Shared constants, conversion tables, regex patterns, normalization functions,
and canonicalization logic used across the grain-price extraction pipeline.

Imported by Stages 1–4. Changes here propagate to every script automatically.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd


# ============================================================
# Section 1 — XML namespace
# ============================================================

NS = {"tei": "http://www.tei-c.org/ns/1.0"}


# ============================================================
# Section 2 — Conversion tables
# ============================================================

CUR_TO_DRACHMA = {
    "drachma": 1.0,
    "obol": 1.0 / 6.0,
    "chalkous": 1.0 / 48.0,
    "denarius": 4.0,
    "sestertius": 1.0,
    "mina": 100.0,
    "talent": 6000.0,
}

# Compound obol denominations. These are single words carrying BOTH an amount
# and a currency (τριώβολον = "three-obol piece"), and in EpiDoc they normally
# sit inside <num value="..."> as an editorial expansion of a symbol.
#
# Order matters: τριημιωβελιον is THREE HALF-obols = 1.5, not 3, so the
# τριημι- prefix must be tested before the bare τρι- prefix. Longest-first
# ordering is enforced at match time.
OBOL_COMPOUNDS = {
    "τριημιωβελ": 1.5,   # 3 half-obols
    "τριημιωβολ": 1.5,
    "ημιωβελ": 0.5,
    "ημιωβολ": 0.5,
    "διωβολ": 2.0,
    "δυωβολ": 2.0,
    "τριωβολ": 3.0,
    "τετρωβολ": 4.0,
    "πεντωβολ": 5.0,
}

UNIT_TO_LITER = {
    "artaba": 38.8,      # approximate placeholder
    "choenix": 1.08,     # approximate placeholder
    "medimnos": 52.5,    # approximate placeholder
    "kotyle": 0.27,      # approximate placeholder
    "metretes": 39.0,    # approximate placeholder
}


# ============================================================
# Section 3 — Text normalization
# ============================================================

def strip_accents(s: str) -> str:
    """Remove combining diacritics. Handles None and non-str input."""
    if s is None:
        return ""
    s = str(s)
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def normalize_for_search(s: str) -> str:
    """NFKC → strip accents → casefold → collapse whitespace."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = strip_accents(s).casefold()
    s = " ".join(s.split())
    return s


# Backward-compatible alias used in Stage 3
norm_greek = normalize_for_search


# ============================================================
# Section 4 — Numeric helpers
# ============================================================

def to_float(x) -> float:
    """Fraction-aware scalar parser (e.g. '3/4' → 0.75)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    if re.fullmatch(r"\d+/\d+", s):
        n, d = s.split("/")
        d = int(d)
        return float(int(n) / d) if d else np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def resolve_money_token(token: str):
    """
    Resolve a single money token to (value, currency).

    `value` is None when the word names a currency without an amount
    (δραχμαί, ὀβολός) — the amount then comes from the neighbouring
    <num value="...">. It is non-None only for compound denominations that
    encode their own amount (τριώβολον → 3.0 obols).

    Returns None if the token is not a money word.

    >>> resolve_money_token("τριωβολον")
    (3.0, 'obol')
    >>> resolve_money_token("τριημιωβελιον")   # 3 half-obols, not 3
    (1.5, 'obol')
    >>> resolve_money_token("δραχμαι")
    (None, 'drachma')
    """
    if not token:
        return None
    t = normalize_for_search(token)
    # Longest stem first so τριημιωβελ- wins over τρι-/ημιωβελ-.
    for stem in sorted(OBOL_COMPOUNDS, key=len, reverse=True):
        if stem in t:
            return (OBOL_COMPOUNDS[stem], "obol")
    for stem, cur in (
        ("δραχμ", "drachma"),
        ("οβολ", "obol"),
        ("οβελ", "obol"),
        ("δηναρ", "denarius"),
        ("ταλαντ", "talent"),
        ("μνα", "mina"),
        ("σεστ", "sestertius"),
        ("ασσ", "assarion"),
        ("χαλκ", "chalkous"),
    ):
        if t.startswith(stem):
            return (None, cur)
    # Unprefixed compound stems (ωβολ/ωβελ) with an unknown numeral prefix:
    # currency is certain, amount is not.
    if "ωβολ" in t or "ωβελ" in t:
        return (None, "obol")
    return None


def to_num(s) -> pd.Series:
    """
    Vectorized numeric parser, fraction-aware.

    pd.to_numeric alone coerces '1/2' to NaN, and fractions are ~20% of the
    <num value="..."> attributes in DDbDP (1/2, 1/4, 1/3, 1/6, 1/8, 1/16),
    so plain coercion silently drops a fifth of all tagged amounts.
    """
    if not isinstance(s, pd.Series):
        s = pd.Series(s if isinstance(s, (list, tuple, np.ndarray)) else [s])
    out = pd.to_numeric(s, errors="coerce")
    missing = out.isna() & s.notna()
    if missing.any():
        out.loc[missing] = s.loc[missing].map(to_float)
    return out


# ============================================================
# Section 5 — Regex vocabulary (compiled patterns)
# ============================================================

GRAIN_RE = re.compile(
    r"\b(σιτ|πυρ|κριθ|ζεα|ολυρ|σταχυ|αλευρ)\w*", re.UNICODE
)

# Vocabulary stems live here ONCE. Stage 2 used to hardcode its own copies of
# these lists for its pairing patterns, so fixes applied here never reached it
# (its copy still lacked the compound-obol stems and carried the same ἀσσ
# breathing-mark bug). Both the *_RE patterns and Stage 2's *_CORE fragments are
# now derived from these tuples.
#
# All stems must be written WITHOUT accents or breathings: every pattern here is
# applied to normalize_for_search() output, which strips them.
UNIT_STEMS = ("αρταβ", "μεδιμν", "χοινι", "μετρ", "κοτυλ", "χοινιξ", "γομφ", "καλαθ")

# Currency words that head their own token.
CUR_STEMS = ("δραχμ", "οβολ", "οβελ", "δηναρ", "μνα", "ταλαντ", "σεστ", "ασσ", "χαλκ")

# Compound-obol stems that appear WORD-INTERNALLY (τρι+ώβολον → "τριωβολον").
# Note the omega: these never match the omicron-initial "οβολ" stem, and being
# mid-word they defeat a \b anchor as well.
CUR_INFIX_STEMS = ("ωβολ", "ωβελ")

UNIT_CORE = "(?:" + "|".join(rf"{s}\w*" for s in UNIT_STEMS) + ")"
CUR_CORE = (
    "(?:"
    + "|".join(rf"{s}\w*" for s in CUR_STEMS)
    + "|"
    + "|".join(rf"\w*{s}\w*" for s in CUR_INFIX_STEMS)
    + ")"
)

UNIT_RE = re.compile(rf"\b{UNIT_CORE}", re.UNICODE)

# \b-anchored for currencies heading their own word; unanchored for the
# word-internal compound-obol stems (see CUR_INFIX_STEMS).
MONEY_RE = re.compile(
    "("
    + r"\b(?:" + "|".join(rf"{s}\w*" for s in CUR_STEMS) + ")"
    + "|"
    + "|".join(rf"\w*{s}\w*" for s in CUR_INFIX_STEMS)
    + ")",
    re.UNICODE,
)

PRICEWORD_RE = re.compile(
    r"(τιμη|τιμ(?=[\.\)\(]))", re.UNICODE
)


# ============================================================
# Section 5b — Classification patterns (negative & positive evidence)
# ============================================================

# --- Negative evidence patterns ---

# False grain stems: compound words whose grain stem is not a commodity reference.
GRAIN_FALSE_STEMS = re.compile(
    r"\b(?:σιτολογ|σιτομετρ|σιταρχ|σιτοφορ|σιτοποι|σιτικ|σιτηγ|σιτηρεσ|σιτωνι)\w*"
    r"|\b(?:πυρρ|πυριν)\w*|\bπυρι\b",
    re.UNICODE,
)

# Tax / fiscal vocabulary
TAX_RE = re.compile(
    r"\b(φυλακιτικ|χωματικ|δημοσι|εκφορ|φορολογ|διοικησ|προσοδ|λαογραφ"
    r"|μερισμ|επιβολ)\w*",
    re.UNICODE,
)

# Administrative titles
ADMIN_RE = re.compile(
    r"\b(πρακτορ|στρατηγ|βασιλικ\w*γραμματ|κωμογραμματ|θησαυροφυλακ|γεωμετρ)\w*",
    re.UNICODE,
)

# Transport / processing fees
TRANSPORT_FEE_RE = re.compile(
    r"\b(ναυλ|φορετρ|αλεστρ|παραναυλ|κοσκινευτ|κατεργ)\w*",
    re.UNICODE,
)

# Ration / salary / wages (narrow: μισθ/τροφ removed — too common in contracts/leases)
RATION_SALARY_RE = re.compile(
    r"\b(οψων|σιτηρεσ|διατροφ|επισιτισμ)\w*",
    re.UNICODE,
)

# Loan vocabulary
LOAN_RE = re.compile(
    r"\b(ατοκ|δανε|εντοκ|τοκος)\w*",
    re.UNICODE,
)

# Seed / sowing (σπορ narrowed to σπορα/σπορο/σπορι to avoid false matches)
SEED_RE = re.compile(
    r"\b(σπερμ|σπειρ|σπορα|σπορο|σπορι)\w*",
    re.UNICODE,
)

# Accounting / list markers (only strong accounting terms; λοιπ/ομοιως are too common)
ACCOUNTING_RE = re.compile(
    r"\b(γινονται|αναλωμ)\w*",
    re.UNICODE,
)

# --- Extended positive evidence patterns ---

# Extended τιμή pattern (all case forms)
PRICEWORD_EXTENDED_RE = re.compile(
    r"\b(τιμη|τιμης|τιμην|τιμας|τιμαι|τιμων)\w*",
    re.UNICODE,
)

# Receipt-price constructions: ἀπὸ τιμ-, εἰς τιμ-, ὑπὲρ τιμ-, ἐκ τιμ-
RECEIPT_PRICE_RE = re.compile(
    r"(απο\s+τιμ|εις\s+τιμ|υπερ\s+τιμ|εκ\s+τιμ)\w*",
    re.UNICODE,
)

# Sale / purchase verb stems
SALE_VERB_RE = re.compile(
    r"\b(αγοραζ|ηγορακ|επριατ|ωνε|πωλ|πιπρασκ|πεπρακ|πρια|ονησ)\w*",
    re.UNICODE,
)

# Per-unit rate constructions (strong positive signal)
RATE_CONSTRUCTION_RE = re.compile(
    r"(ως\s+της\s+αρταβης|ανα\s+δραχμ\w*|εκ\s+δραχμ\w*|ανα\s+αρταβ\w*"
    r"|εκαστης\s+\w*\s*δραχμ\w*)",
    re.UNICODE,
)

# Per-aroura rate (land rent, NOT grain price) — weakens positive signal
PER_AROURA_RE = re.compile(
    r"\b(αρουρ)\w*",
    re.UNICODE,
)

# Compulsory / state purchase — σιτωνία, συναγορασμός, embolē, annona.
# σιτωνι- is also in GRAIN_FALSE_STEMS (it names the purchasing office, not the
# commodity), so it is filtered out as a grain hit but is still a context cue.
REQUISITION_RE = re.compile(
    r"\b(σιτωνι|συναγορασμ|εμβολ|αννων)\w*",
    re.UNICODE,
)

# Penalty clauses are explicitly market-referencing ("at the highest price",
# "whatever it may be") and make a useful upper-bound anchor — definition §3.
PENALTY_RE = re.compile(
    r"(τησ\s+πλειστησ\s+τιμ|οσου\s+εαν\s+η|πλειστ\w*\s+τιμ)",
    re.UNICODE,
)

# Cue vocabulary. These are EVIDENCE, not verdicts: firing a pattern records a
# fact about the text, it does not decide whether the row is a price.
# ACCOUNTING_RE is excluded: γίνονται/ἀνάλωμα are too common as general vocabulary;
# accounting boundaries are handled by has_intervening_boundary() in Stage 2.
NEGATIVE_PATTERNS = {
    "TAX": TAX_RE,
    "ADMIN": ADMIN_RE,
    "TRANSPORT": TRANSPORT_FEE_RE,
    "RATION": RATION_SALARY_RE,
    "LOAN": LOAN_RE,
    "SEED": SEED_RE,
    "REQUISITION": REQUISITION_RE,
    "PENALTY": PENALTY_RE,
}

# All positive pattern keys, their compiled regex, and base strength
POSITIVE_PATTERNS = {
    "receipt_price": (RECEIPT_PRICE_RE, 0.7),
    "rate_construction": (RATE_CONSTRUCTION_RE, 0.8),
    "priceword_ext": (PRICEWORD_EXTENDED_RE, 0.6),
    "sale_verb": (SALE_VERB_RE, 0.5),
}


# §6 residual hard exclusions: settings where there is no monetary valuation of
# the GRAIN to recover at all. Everything else is included and tagged (§3).
#   TRANSPORT — ναῦλον/φόρετρον/ἄλεστρα price the carriage or the milling, not the grain
#   RATION    — in-kind rations and wages carry no monetary valuation
# TAX / ADMIN / LOAN are deliberately NOT here: a tax or rent reckoned in cash is
# a price (adaeratio), tagged `tax_commutation`; a loan with a τιμή is a price,
# tagged `loan`. Treating them as gates is exactly the bug definition §11 warns
# about. SEED is conditional — see is_hard_exclusion().
HARD_EXCLUSION_CUES = ("TRANSPORT", "RATION")


def is_hard_exclusion(cue_flags, has_money: bool = True) -> str:
    """
    Return a reason string if this context falls in the §6 residual set, else "".

    `has_money` distinguishes the two conditional cases: grain-in/grain-out seed
    loans and pure tax-in-kind have no cash figure to recover, but the same
    vocabulary with a money amount attached is a genuine price observation.
    """
    flags = set(cue_flags)
    for cue in HARD_EXCLUSION_CUES:
        if cue in flags:
            return f"hard_exclusion({cue.lower()})"
    if not has_money:
        if "SEED" in flags:
            return "hard_exclusion(seed_in_kind)"
        if "TAX" in flags:
            return "hard_exclusion(tax_in_kind)"
    return ""


# §3 institutional context. Multi-label on purpose: a document can be both a
# loan and a tax commutation, and the old code let whichever negative pattern
# happened to come first in the dict win.
_CUE_TO_TAG = {
    "TAX": "tax_commutation",
    "REQUISITION": "requisition",
    "ADMIN": "requisition",
    "LOAN": "loan",
    "PENALTY": "penalty",
    "sale_verb": "sale",
}


def context_tags(cue_flags) -> list[str]:
    """
    Map fired cues onto the §3 `context_tag` set.

    Returns a sorted list; empty input yields ["unknown"]. The final category
    set is still being closed under issue #018 (freeze due Oct 2026) — this
    implements §3 as written, and because the raw cue flags are stored
    alongside, re-tagging later needs no re-extraction.
    """
    flags = set(cue_flags)
    tags = {tag for cue, tag in _CUE_TO_TAG.items() if cue in flags}

    # A τιμή with no sale verb and no institutional cue is an accounting or
    # dowry valuation, not a transaction (§3: "most τιμή are valuation").
    if not tags and ("priceword_ext" in flags or "receipt_price" in flags):
        tags.add("valuation")
    if not tags and "rate_construction" in flags:
        tags.add("rate_other")
    return sorted(tags) if tags else ["unknown"]


def linkage_strength(
    has_explicit_cue: bool,
    dist_grain_money: int,
    has_quantity: bool,
    proximate_max_tokens: int = 12,
) -> str:
    """
    §2 linkage strength — replaces the old 0-100 score as the strictness dial.

    The score summed four heterogeneous quantities into one uninterpretable
    number and, as configured, never excluded a single row. This is a 3-level
    ordinal with a stated meaning, so a strict (explicit-only) series and a
    broad series can both be drawn from the same table — strictness applied at
    analysis time, which is where §2 says it belongs.
    """
    if has_explicit_cue:
        return "explicit"
    if dist_grain_money <= proximate_max_tokens and has_quantity:
        return "proximate"
    return "weak"


def scan_context_cues(
    norm_text: str,
    grain_start: int = 0,
    grain_end: int | None = None,
    window_chars: int = 300,
) -> tuple[list[str], str]:
    """
    Record which cue patterns fire around a grain mention.

    This reports EVIDENCE only — it makes no include/exclude decision. That
    separation is the point: `pipeline_shared` states the machine-checkable
    facts, and the tagging (§3) and the residual exclusion (§6) are applied on
    top, so a later re-tagging never needs the extraction re-run.

    Returns (cue_flags, window_text).
    """
    if grain_end is None:
        grain_end = grain_start + 10

    lo = max(0, grain_start - window_chars)
    hi = min(len(norm_text), grain_end + window_chars)
    window = norm_text[lo:hi]

    cues: list[str] = [
        label for label, pat in NEGATIVE_PATTERNS.items() if pat.search(window)
    ]
    cues += [key for key, (pat, _s) in POSITIVE_PATTERNS.items() if pat.search(window)]

    # ἄρουρα nearby means a per-aroura land rent rather than a grain price.
    if PER_AROURA_RE.search(window):
        cues.append("per_aroura")

    return cues, window


# Positive cues that count as an explicit grain↔money link for §2.
EXPLICIT_LINK_CUES = ("receipt_price", "rate_construction", "priceword_ext", "sale_verb")


# ============================================================
# Section 6 — Canonicalization
# ============================================================

PLACE_PATTERNS = [
    ("Oxyrhynchite", re.compile(r"oxyrh|οξυρ", re.I)),
    ("Arsinoite", re.compile(r"arsin|αρσιν|φαγιουμ|fayum", re.I)),
    ("Hermopolite", re.compile(r"hermop|ερμοπ", re.I)),
    ("Herakleopolite", re.compile(r"herakleop|ηρακλεοπ", re.I)),
    ("Memphite", re.compile(r"memph|μεμφ", re.I)),
    ("Panopolite", re.compile(r"panop|πανοπ|akhmim", re.I)),
    ("Thebaid", re.compile(r"theb|θηβ|λυκοπ|lycop", re.I)),
    ("Alexandria", re.compile(r"alex|αλεξανδρ", re.I)),
    ("Unknown", re.compile(r".*", re.I)),
]


def canon_grain(form: str) -> str:
    g = norm_greek(form)
    if not g:
        return ""
    if g.startswith("πυρ"):
        return "wheat (pyros)"
    if g.startswith("σιτ"):
        return "grain (sitos)"
    if g.startswith("κριθ"):
        return "barley (krithē)"
    if g.startswith("ζε"):
        return "spelt/zea"
    if g.startswith("ολυρ"):
        return "emmer (olyra)"
    if g.startswith("αλευρ"):
        return "flour (aleuron)"
    if g.startswith("σταχυ"):
        return "ear/spike (stachys)"
    return g


def canon_unit(unit: str) -> str:
    u = norm_greek(unit)
    if not u:
        return ""
    if u.startswith("αρταβ"):
        return "artaba"
    if u.startswith("μεδιμν"):
        return "medimnos"
    if u.startswith("χοιν"):
        return "choenix"
    if u.startswith("κοτυλ"):
        return "kotyle"
    # μετρητής is a *liquid* measure of about 39 litres; μέτρον is the generic
    # dry measure a grain account reckons in ("κριθῆς μέτρα γ", "πυροῦ μέτρα 12").
    # Folding both onto "metretes" labelled every grain μέτρον as a wine measure,
    # and made the gold unsatisfiable: an annotator writing the true unit was
    # told it differed from the selected option. Longest stem first.
    if u.startswith("μετρητ"):
        return "metretes"
    if u.startswith("μετρ"):
        return "metron"
    return u


def canon_currency(cur: str) -> str:
    c = norm_greek(cur)
    if not c:
        return ""
    if c.startswith("δραχ"):
        return "drachma"
    if c.startswith("οβολ"):
        return "obol"
    if c.startswith("δηναρ"):
        return "denarius"
    if c.startswith("μνα"):
        return "mina"
    if c.startswith("ταλαν"):
        return "talent"
    if c.startswith("σεστ"):
        return "sestertius"
    if c.startswith("χαλκ"):
        return "chalkous"
    if c.startswith("ασσ"):
        return "as"
    # Compound denominations (τριώβολον, ἡμιωβέλιον …) are obols. Their stem is
    # ωβολ/ωβελ with an omega, word-internally, so none of the tests above fire
    # and they would otherwise survive as raw Greek — leaving the row with a
    # currency absent from CUR_TO_DRACHMA and so unconvertible.
    if "ωβολ" in c or "ωβελ" in c or c.startswith("οβελ"):
        return "obol"
    return c


def canon_place(place: str) -> str:
    p = norm_greek(place)
    if not p:
        return "Unknown"
    for label, pat in PLACE_PATTERNS:
        if pat.search(p):
            return label
    return "Unknown"

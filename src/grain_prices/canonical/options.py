"""Position-aware canonical amount scanner (#032–#034).

Offsets here refer to normalized, numeral-expanded block text. The enumerator
maps *all* spans back to the displayed block before publishing options.
Legacy scanners and historical artifacts are deliberately unaffected.
"""
from __future__ import annotations

import re
from fractions import Fraction

from grain_prices.pipeline_shared import (
    GRAIN_RE, MONEY_RE as OLD_MONEY_RE, UNIT_RE, canon_currency,
    normalize_for_search, resolve_money_token,
)

LATE_CURRENCIES = {"νομισμ": "nomisma", "κερατι": "keration",
                   "σολιδ": "solidus", "χρυσιν": "chrysinon"}
MONEY_RE = re.compile(OLD_MONEY_RE.pattern + r"|\b(?:νομισμ|κερατι|σολιδ|χρυσιν)\w*")
NUM_RE = re.compile(r"(?<![\w/])(?:\d+/\d+|\d+)(?:\s+\d+/\d+)*(?![\w/])")
MYRIAD_RE = re.compile(r"\W*μυρια(?:δ\w*|[σς])\W*")
DATE_RE = re.compile(r"\W*(?:ετ(?:ου[σς]|ει|ο[σς]|ων)|μην\w*|ημερ\w*)\b")
DATE_WORD = r"(?:ετου[σς]|ετει|ετο[σς]|ετων|ετη|μηνο[σς]|μηνι|μηνων|ημερα[σς]?|ημερων)"
DATE_AFTER_RE = re.compile(r"\s+" + DATE_WORD + r"\b")
DATE_BEFORE_RE = re.compile(r"\b" + DATE_WORD + r"\s+$")
BOUNDARY_RE = re.compile(r"zz(?:lb|cb|pb)zz|[|;·.]")


def currency_name(value: str | None) -> str:
    normalized = normalize_for_search(value or "")
    for stem, name in LATE_CURRENCIES.items():
        if normalized.startswith(stem):
            return name
    return canon_currency(value or "")


def numeric_value(text: str, multiplier: int = 1) -> str | None:
    try:
        value = sum((Fraction(part) for part in text.split()), Fraction()) * multiplier
        return str(value.numerator) if value.denominator == 1 else format(float(value), ".12g")
    except (ValueError, ZeroDivisionError):
        return None


def date_owned(text: str, number) -> bool:
    """Reserve explicit date/duration numerals before all denomination binding.

    Do not cross punctuation: ``drachmas 8, year 10`` must keep eight.
    Bare day numerals without a date word remain for contextual adjudication.
    """
    return bool(DATE_AFTER_RE.match(text[number.end():])
                or DATE_BEFORE_RE.search(text[:number.start()]))


def scan_options(text: str, tokens) -> tuple[list[dict], list[dict]]:
    """Bind strong amounts first, then unclaimed loose amounts, then ellipses.

    Following adjacency takes precedence over preceding adjacency, avoiding the
    shared numeral in ``artaba 1 drachmas 8``. Equal-ranked competing bindings
    remain value-less and flagged; traversal order never decides ownership.
    """
    numbers = list(NUM_RE.finditer(text))
    date_numbers = {ni for ni, n in enumerate(numbers) if date_owned(text, n)}
    anchors = [(kind, match) for kind, regex in (("quantity", UNIT_RE), ("money", MONEY_RE))
               for match in regex.finditer(text)]
    anchors.sort(key=lambda item: (item[1].start(), item[0]))
    options = []
    for kind, word in anchors:
        resolved = resolve_money_token(word.group()) if kind == "money" else None
        options.append({
            "kind": kind, "value": None,
            "unit" if kind == "quantity" else "currency": word.group(),
            "char_start": word.start(), "char_end": word.end(),
            "denomination_char_start": word.start(), "denomination_char_end": word.end(),
            "value_char_start": None, "value_char_end": None,
            "token_index": tokens.tok_at_char(word.start()),
            "adjacency": "word_only", "gap_tokens": None,
            "inference_rule": None, "ownership_ambiguous": False,
            "intrinsic_value": resolved[0] if resolved else None,
            "multiplier": 1, "written_value": None,
        })
    claimed: set[int] = set()
    assigned: set[int] = set()
    disputed: set[int] = set()
    proposals = []
    for index, (kind, word) in enumerate(anchors):
        for ni, number in enumerate(numbers):
            if ni in date_numbers:
                continue
            after = number.start() >= word.end()
            if not after and number.end() > word.start():
                continue
            gap = text[word.end():number.start()] if after else text[number.end():word.start()]
            if BOUNDARY_RE.search(gap):
                continue
            # Never cross another denomination or numeral to find an amount.
            lo, hi = (word.end(), number.start()) if after else (number.end(), word.start())
            if any(lo <= other.start() < hi for _, other in anchors) or NUM_RE.search(gap):
                continue
            distance = abs(tokens.tok_at_char(number.start()) - tokens.tok_at_char(word.start()))
            myriad = kind == "money" and after and bool(MYRIAD_RE.fullmatch(gap))
            if myriad or not re.search(r"\w", gap):
                rank = 0 if after else 1
            elif distance <= 3:
                rank = 2 + distance
            else:
                continue
            proposals.append((rank, index, ni, 10000 if myriad else 1))
    for rank in sorted({p[0] for p in proposals}):
        available = [p for p in proposals if p[0] == rank and p[1] not in assigned
                     and p[2] not in claimed and p[2] not in disputed]
        for _, index, ni, multiplier in available:
            contenders = [p for p in available if p[2] == ni or p[1] == index]
            if len(contenders) > 1:
                options[index]["ownership_ambiguous"] = True
                disputed.add(ni)
        for _, index, ni, multiplier in available:
            if ni in disputed or options[index]["ownership_ambiguous"]:
                continue
            number, option = numbers[ni], options[index]
            option.update(value=numeric_value(number.group(), multiplier),
                          written_value=number.group(), multiplier=multiplier,
                          value_char_start=number.start(), value_char_end=number.end(),
                          char_start=min(option["char_start"], number.start()),
                          char_end=max(option["char_end"], number.end()),
                          adjacency="strict" if rank <= 1 else "loose",
                          gap_tokens=abs(tokens.tok_at_char(number.start()) - option["token_index"]))
            if rank <= 1:
                # Preserve the historical strict-event anchor at the expression
                # start, including number-before-denomination forms. Anchoring
                # only on the word would drop amounts on the 80-token edge.
                option["token_index"] = tokens.tok_at_char(option["char_start"])
            claimed.add(ni)
            assigned.add(index)

    # Ellipses require a local construction, not merely an unclaimed integer.
    cues = [("quantity", g, "grain_number") for g in GRAIN_RE.finditer(text)]
    cues += [("money", m, "ana_number") for m in re.finditer(r"\bανα\b", text)]
    cues += [("money", m, "time_number") for m in re.finditer(r"\bτιμη\w*", text)]
    for kind, cue, rule in sorted(cues, key=lambda item: item[1].start()):
        for ni, number in enumerate(numbers):
            if number.start() < cue.end() or ni in claimed or ni in disputed:
                continue
            gap = text[cue.end():number.start()]
            # A price word followed by a grain and a number is a quantity
            # construction; do not steal that number as the money side.
            if re.search(r"\w", gap) or BOUNDARY_RE.search(gap):
                break
            if ni in date_numbers:
                break
            options.append({
                "kind": kind, "unit" if kind == "quantity" else "currency": None,
                "value": numeric_value(number.group()), "written_value": number.group(),
                "multiplier": 1, "intrinsic_value": None,
                "char_start": cue.start(), "char_end": number.end(),
                "denomination_char_start": None, "denomination_char_end": None,
                "value_char_start": number.start(), "value_char_end": number.end(),
                "token_index": tokens.tok_at_char(cue.start()),
                "adjacency": "inferred", "gap_tokens": 1,
                "inference_rule": rule, "ownership_ambiguous": False,
            })
            claimed.add(ni)
            break
    return ([o for o in options if o["kind"] == "quantity"],
            [o for o in options if o["kind"] == "money"])

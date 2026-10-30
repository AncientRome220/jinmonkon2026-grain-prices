"""Local, source-bound suggestions from a selected quotation, without gold/LLM use."""
from copy import deepcopy
from fractions import Fraction
import re
import unicodedata

from grain_prices.annotation.span_selection import quote_spans
from grain_prices.canonical.options import scan_options, currency_name
from grain_prices.canonical.text import NormalizedText, normalize_with_map
from grain_prices.legacy.stage2 import tokenize_with_positions
from grain_prices.pipeline_shared import GRAIN_RE, canon_unit


def join_split_units(mapped):
    """Join only attested denomination forms across a line marker, retaining offsets."""
    forms = re.compile(r"(?:αρταβ(?:η|ησ|ην|αι|αισ|ασ|ων)|δραχμ(?:η|ησ|ην|αι|αισ|ασ|ων)|οβολ(?:οσ|ου|ον|οι|οισ|ουσ|ων))")
    cuts = set()
    # Lookahead permits adjacent candidates without greedily consuming a whole line.
    for match in re.finditer(r"(?=\b([α-ω]+)(\s+zzlbzz\s+)([α-ω]+)\b)", mapped.text):
        if forms.fullmatch(match[1]+match[3]):
            cuts.update(range(match.start(2), match.end(2)))
    return NormalizedText("".join(c for i, c in enumerate(mapped.text) if i not in cuts),
                          tuple(p for i, p in enumerate(mapped.original_indexes) if i not in cuts))


def suggest_relation(quote: str, row: dict) -> tuple[dict, list[str]]:
    text = row["source_block_text"]
    ga, gb = row["grain_span"]
    matches = [s for s in quote_spans(text, quote.strip()) if s[0] <= ga < gb <= s[1]]
    if len(matches) != 1:
        raise ValueError("現在の対象穀物を含む原文の一節を、そのままコピーしてください。")
    lo, hi = matches[0]
    surface = text[lo:hi]
    mapped = join_split_units(normalize_with_map(surface))

    def source_span(a, b):
        start, end = mapped.original_span(a, b)
        # Keep combining accents belonging to the final source character.
        while end < len(surface) and unicodedata.combining(surface[end]):
            end += 1
        return [lo + start, lo + end]

    grains = list(GRAIN_RE.finditer(mapped.text))
    if len(grains) != 1 or source_span(grains[0].start(), grains[0].end()) != row["grain_span"]:
        raise ValueError("引用に複数の穀物表現があります。対象の穀物と対応する数量・金額だけを含む範囲に絞ってください。")
    grain = grains[0].group()
    commodity = next((name for stem, name in (("πυρ", "wheat"), ("κριθ", "barley"),
                     ("ολυρ", "emmer"), ("ζε", "zea"), ("αλευρ", "flour"),
                     ("σιτ", "grain_unspecified"), ("σταχυ", "grain_unspecified")) if grain.startswith(stem)), "other")
    quantities, monies = scan_options(mapped.text, tokenize_with_positions(mapped.text))
    notes = []
    basis = "unknown"
    for q in quantities:
        # Explicit singular unit + 'each': do not infer 1 from a bare unit.
        end = q["char_end"]
        each = re.match(r"\s+εκαστ(?:ησ|ου)\b", mapped.text[end:])
        singular = q.get("unit") in {"αρταβησ", "μεδιμνου", "χοινικοσ"}
        if each and singular and q["value"] is None and not q["ownership_ambiguous"]:
            q.update(value="1", char_end=end+each.end())
            basis = "stated_rate"
            notes.append("「各単位」の表現から数量1と単価を提案しました。")
    relation = {"grain_span": list(row["grain_span"]), "commodity": commodity,
                "evidence_span": [lo, hi], "rate_basis": basis}
    for side, label, options in (("quantity", "数量", quantities), ("money", "金額", monies)):
        if len(options) > 1:
            # Preserve adjacent compound money as components, never sum currencies.
            units = [currency_name(o.get("currency")) for o in options] if side == "money" else []
            compound = (side == "money" and len(set(units)) == len(options) and all(units)
                        and all(re.fullmatch(r"\s*(?:και\s*)?", mapped.text[a["char_end"]:b["char_start"]])
                                for a, b in zip(options, options[1:])))
            if not compound:
                candidates = " ／ ".join(text[slice(*source_span(o["char_start"], o["char_end"]))] for o in options)
                raise ValueError(f"{label}の候補が複数あります：{candidates}。引用を絞るか、個別入力で対応関係を指定してください。")
            notes.append("複合金額を別々の成分として提案しました。対応関係を確認してください。")
        components = []
        unresolved = not options
        for o in options:
            value = o["value"]
            if o["ownership_ambiguous"]:
                value = None
                unresolved = True
            intrinsic = o.get("intrinsic_value")
            if intrinsic is not None:
                if value is None and o["adjacency"] == "word_only":
                    value = str(Fraction(str(intrinsic)))
                else:
                    value = None
                    unresolved = True
            denomination = (canon_unit(o.get("unit") or "") if side == "quantity"
                            else currency_name(o.get("currency"))) or None
            if value is None:
                # Missing numeral may be omitted or unreadable; leave that decision open.
                unresolved = True
            components.append({"span": source_span(o["char_start"], o["char_end"]),
                               "value": value, "denomination": denomination})
        relation[side] = {"status": "unresolved" if unresolved else "present", "components": components}
        if unresolved:
            notes.append(f"{label}を読み切れませんでした。未確認の欄を個別に確認してください（記載なしとは判定していません）。")
    if basis == "unknown":
        notes.append("単価か総額かは未判定です。Rate basisを確認してください。")
    return deepcopy(relation), notes

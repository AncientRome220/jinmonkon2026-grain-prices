"""Canonical enumeration; text preparation retained from frozen Stage 2C.

Option scanning and source coordinates are canonical-owned (#032–#036).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List
from lxml import etree
from grain_prices.legacy import stage2 as S2
from grain_prices.canonical.options import scan_options
from grain_prices.canonical.relation import relation_linkage
from grain_prices.canonical.text import normalize_with_map
from grain_prices.pipeline_shared import NS, POSITIVE_PATTERNS, EXPLICIT_LINK_CUES, scan_context_cues, context_tags
from grain_prices.preprocessing.parse_xml import HgvMeta, get_ddb_id

def count_lines_crossed(tokens: List[str], a: int, b: int) -> int:
    lo, hi = sorted((a, b))
    return sum(1 for t in tokens[lo:hi] if t == S2.LB_MARK)


def crosses_block(tokens: List[str], a: int, b: int) -> bool:
    lo, hi = sorted((a, b))
    return any(t in (S2.CB_MARK, S2.PB_MARK) for t in tokens[lo:hi])


def enumerate_for_doc(xml_path: Path, pats, meta: HgvMeta,
                      window_tokens: int, evidence_chars: int,
                      max_evidence: int = 1200) -> List[dict]:
    """
    Emit one record per grain mention, carrying the quantity options and the
    money options found near it as two parallel lists.

    Deliberately NOT the Cartesian product of the two. The product is quadratic
    — in an account listing many amounts a single mention reached 990 tuples
    (3.4% of mentions exceeded 200), far past what fits in an adjudication
    prompt. Two lists are linear (max 61, median 8) and lose nothing: choosing
    one quantity and one money option reconstructs any tuple, and "none" is
    always an allowed answer on either axis.
    """
    try:
        root = etree.parse(str(xml_path)).getroot()
    except Exception:
        return []
    ddb_id = get_ddb_id(root)
    if not ddb_id:
        return []

    out: List[dict] = []
    mention_id = 0

    for edition_index, div in enumerate(root.xpath("//tei:div[@type='edition']", namespaces=NS)):
        for block_index, (_tag, block) in enumerate(S2.iter_blocks(div)):
            value_text = S2.block_text_with_num_values(block)
            value_text = " ".join(value_text.replace("𐅵", " 1/2 ").split())
            if not value_text:
                continue
            mapped_text = normalize_with_map(value_text)
            value_norm = mapped_text.text
            tok_norm = S2.tokenize_with_positions(value_norm)
            explicit_cue_positions = []
            for label, (pattern, _strength) in POSITIVE_PATTERNS.items():
                if label not in EXPLICIT_LINK_CUES:
                    continue
                for cue_match in pattern.finditer(value_norm):
                    explicit_cue_positions.append({
                        "label": label,
                        "token_index": tok_norm.tok_at_char(cue_match.start()),
                    })

            grains, qtys, prices, pws = S2.build_events(
                pats, value_norm, tok_norm, keep_all=True
            )
            # A block qualifies if it has a grain mention and ANY currency word —
            # not only ones the strict price patterns could bind to a digit.
            if not grains:
                continue

            block_qty, block_money = scan_options(value_norm, tok_norm)
            for g in grains:
                mention_id += 1
                q_opts, m_opts = [], []
                for prefix, source_options, target in (("q", block_qty, q_opts), ("m", block_money, m_opts)):
                    for option in sorted(source_options, key=lambda o: (abs(o["token_index"] - g.tok), o["char_start"])):
                        tok = option["token_index"]
                        if abs(tok - g.tok) > window_tokens:
                            continue
                        target.append({**option, "option_id": f"{prefix}{len(target)}",
                            "dist_from_grain": abs(tok - g.tok),
                            "lines_crossed": count_lines_crossed(tok_norm.tokens, g.tok, tok),
                            "crosses_block_boundary": crosses_block(tok_norm.tokens, g.tok, tok),
                            "flag_qty_price_value_collision": False})

                # Span the grain and its options, but cap the width: in a long
                # account block the options reach across the whole block, and
                # storing that per mention duplicates the block many times over.
                if not m_opts:
                    continue

                # Linkage is relation-specific.  Record the deterministic
                # features for every money option; the final value is computed
                # again after the LLM/human selects a quantity option.
                for money_opt in m_opts:
                    nearest_q = min(
                        q_opts,
                        key=lambda q: abs(q["token_index"] - g.tok),
                        default=None,
                    )
                    crosses = crosses_block(
                        tok_norm.tokens, g.tok, money_opt["token_index"]
                    ) or bool(money_opt.get("crosses_block_boundary"))
                    strength, local_cues = relation_linkage(
                        grain_token=g.tok,
                        money_token=money_opt["token_index"],
                        quantity_token=(nearest_q["token_index"] if nearest_q else None),
                        explicit_cues=explicit_cue_positions,
                        crosses_boundary=crosses,
                    )
                    money_opt["relation_cues"] = local_cues
                    money_opt["crosses_relation_boundary"] = crosses
                    money_opt["linkage_with_nearest_quantity"] = strength

                # Option spans are exposed in original/displayed coordinates.
                # The normalized offsets remain as explicit debug attributes.
                for option in q_opts + m_opts:
                    for prefix in ("", "value_", "denomination_"):
                        start, end = option.get(prefix + "char_start"), option.get(prefix + "char_end")
                        if start is None or end is None:
                            continue
                        option[prefix + "norm_char_start_debug"] = start
                        option[prefix + "norm_char_end_debug"] = end
                        option[prefix + "char_start"], option[prefix + "char_end"] = mapped_text.original_span(start, end)

                # §3 institutional context / §2 linkage, from the same cue scan
                # Stage 2 uses. Recorded, never used to drop a candidate.
                cues, _w = scan_context_cues(
                    value_norm, g.char_start, g.char_end,
                    window_chars=max(200, evidence_chars * 4),
                )
                order = {"explicit": 0, "proximate": 1, "weak": 2}
                link = min(
                    (o["linkage_with_nearest_quantity"] for o in m_opts),
                    key=lambda value: order[value],
                    default="weak",
                )

                spans = ([o["norm_char_start_debug"] for o in q_opts + m_opts]
                         + [o["norm_char_end_debug"] for o in q_opts + m_opts]
                         + [g.char_start, g.char_end])
                norm_lo = max(0, min(spans) - evidence_chars)
                norm_hi = min(len(value_norm), max(spans) + evidence_chars)
                if norm_hi - norm_lo > max_evidence:
                    centre = (g.char_start + g.char_end) // 2
                    norm_lo = max(0, centre - max_evidence // 2)
                    norm_hi = min(len(value_norm), norm_lo + max_evidence)
                surface_lo, surface_hi = mapped_text.original_span(norm_lo, norm_hi)
                grain_surface_start, grain_surface_end = mapped_text.original_span(
                    g.char_start, g.char_end
                )
                identity_norm_lo = max(0, g.char_start - 100)
                identity_norm_hi = min(len(value_norm), g.char_end + 100)
                identity_lo, identity_hi = mapped_text.original_span(
                    identity_norm_lo, identity_norm_hi
                )
                identity_neighborhood = value_text[identity_lo:identity_hi]
                identity_neighborhood = (
                    identity_neighborhood.replace(S2.LB_MARK, "|")
                    .replace(S2.CB_MARK, "||").replace(S2.PB_MARK, "||")
                )

                out.append({
                    "ddb_id": ddb_id,
                    "source_block_id": f"e{edition_index}:b{block_index}",
                    "source_block_sha256": hashlib.sha256(value_text.encode("utf-8")).hexdigest(),
                    "source_block_text": value_text,
                    "coordinate_system": "expanded-block-surface-v1",
                    "evidence_char_start": surface_lo,
                    "evidence_char_end": surface_hi,
                    "mention_id": f"{ddb_id}#{mention_id}",
                    "grain_form": value_text[grain_surface_start:grain_surface_end],
                    "grain_normalized": g.form,
                    "grain_char_start": g.char_start,
                    "grain_char_end": g.char_end,
                    "grain_surface_char_start_debug": grain_surface_start,
                    "grain_surface_char_end_debug": grain_surface_end,
                    "identity_source_text": identity_neighborhood,
                    "identity_grain_start_debug": grain_surface_start - identity_lo,
                    "identity_grain_end_debug": grain_surface_end - identity_lo,
                    "identity_neighborhood": identity_neighborhood,
                    "grain_token_index": g.tok,
                    "explicit_cue_positions": explicit_cue_positions,
                    "flag_krino": g.flag_krino,
                    "flag_admin_compound": g.flag_admin_compound,
                    "flag_false_stem": g.flag_false_stem,
                    "flag_admin_word_near": g.flag_admin_word_near,
                    "priceword_near": any(
                        abs(w.tok - g.tok) <= window_tokens for w in pws),
                    "context_tags": context_tags(cues),
                    "cue_flags": cues,
                    "linkage_strength": link,
                    "qty_options": q_opts,
                    "money_options": m_opts,
                    "n_qty_options": len(q_opts),
                    "n_money_options": len(m_opts),
                    # What the Cartesian product WOULD have been; kept so the
                    # reduction is reportable, not to be materialised.
                    "n_combinations": (len(q_opts) + 1) * len(m_opts),
                    "place": meta.place,
                    "date_not_before": meta.date_notbefore,
                    "date_not_after": meta.date_notafter,
                    "evidence_text": value_text[surface_lo:surface_hi],
                })
    return out

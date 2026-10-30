#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2C — Candidate tuple enumeration (recall-first).

Stage 2 commits to one answer per grain mention: `find_nearest_price_for_grain`
keeps `local_prices[0]` and throws the rest away, `pick_primary_global` greedily
forbids reusing one money amount across grains, and `build_events` silently
deletes events matched by five exclusion rules. The output CSV is 100%
`Is_Primary=yes` — the alternatives were never written down. So when the rule
layer links the wrong pair, the right pair is invisible to the LLM and to the
human reviewer afterwards; the error is unrecoverable and unmeasurable.

This stage inverts that. The rule layer becomes a *candidate generator*:

  * every grain mention is kept (no grid dedup),
  * every (grain, quantity, money) tuple within the span limit is emitted,
    not just the nearest,
  * the five exclusion rules become boolean features on the tuple instead of
    deletions (build_events(keep_all=True)),
  * each element carries its character span, so a reviewer can be shown exactly
    what was linked.

Adjudication — choosing which tuple, if any, is the real price — is deferred to
Stage 3 (LLM) and human review. The metric for THIS stage is therefore not
precision but candidate-set recall: does the correct pairing appear anywhere in
the emitted set?

Output: data/candidate_tuples.jsonl (canonical) + .csv (flat projection).

Usage:
    python src/pipeline/2c_enumerate_candidates.py --limit 200
    python src/pipeline/2c_enumerate_candidates.py
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import re  # noqa: E402

from grain_prices.pipeline_shared import (  # noqa: E402
    MONEY_RE,
    NS,
    UNIT_RE,
    EXPLICIT_LINK_CUES,
    POSITIVE_PATTERNS,
    context_tags,
    linkage_strength,
    normalize_for_search,
    scan_context_cues,
)
from grain_prices.canonical.relation import relation_linkage  # noqa: E402
from grain_prices.canonical.text import normalize_with_map  # noqa: E402
from grain_prices.preprocessing.parse_xml import (  # noqa: E402
    HgvMeta,
    build_hgv_index,
    get_ddb_id,
    iter_xml_files,
)
from lxml import etree  # noqa: E402


from grain_prices.legacy import stage2 as S2  # noqa: E402

log = logging.getLogger("stage2c")

DEFAULT_DDB = Path(os.environ["GRAIN_PRICES_DDB_DIR"]) if os.environ.get("GRAIN_PRICES_DDB_DIR") else None
DEFAULT_HGV = Path(os.environ["GRAIN_PRICES_HGV_DIR"]) if os.environ.get("GRAIN_PRICES_HGV_DIR") else None
DEFAULT_OUT = Path("data/candidate_tuples.jsonl")

# Flat CSV projection: one row per mention, options rendered compactly.
FIELDNAMES = [
    "ddb_id", "mention_id",
    "grain_form", "grain_char_start", "grain_char_end",
    "n_qty_options", "n_money_options", "n_combinations",
    "qty_options", "money_options",
    "priceword_near", "linkage_strength", "context_tags", "cue_flags",
    "flag_krino", "flag_admin_compound", "flag_false_stem", "flag_admin_word_near",
    "place", "date_not_before", "date_not_after",
    "evidence_text",
]


# A numeric amount: an integer and/or a run of unit fractions (1/2 1/12).
NUM_RE = re.compile(r"\d+(?:\W+\d+/\d+)*|\d+/\d+(?:\W+\d+/\d+)*")


def loose_options(word_re, label_key: str, value_norm: str, tok_norm,
                  grain_tok: int, window_tokens: int, strict_spans: set,
                  max_gap_tokens: int = 3) -> List[dict]:
    """
    Unit/currency words whose amount is NOT adjacent, plus ones with no amount
    at all.

    The strict patterns require `word \\W+ digits`. Papyri routinely spell the
    numeral out and print both forms — "δραχμὰς δέκα 10", "ἀρτάβας ὀκτὼ 8",
    "δραχμὰς τρισχιλίας γίνονται 3000" — so a word sits between the unit or
    currency and the digits, and the strict pattern misses the amount entirely.
    Three of the twenty gold-standard TP documents were unreachable for exactly
    this reason. A generator must not lose them; the looseness is recorded as a
    flag so the adjudicator can weigh it and its cost stays measurable.
    """
    out: List[dict] = []
    nums = [(m, tok_norm.tok_at_char(m.start())) for m in NUM_RE.finditer(value_norm)]
    for mm in word_re.finditer(value_norm):
        if any(s <= mm.start() < e for s, e in strict_spans):
            continue  # already offered as a strict option
        wtok = tok_norm.tok_at_char(mm.start())
        if abs(wtok - grain_tok) > window_tokens:
            continue
        best = None
        for nm, ntok in nums:
            d = abs(ntok - wtok)
            if d <= max_gap_tokens and (best is None or d < best[0]):
                best = (d, nm)
        out.append({
            "value": best[1].group(0) if best else None,
            label_key: mm.group(0),
            "char_start": mm.start(),
            "char_end": mm.end(),
            "value_char_start": best[1].start() if best else None,
            "value_char_end": best[1].end() if best else None,
            "gap_tokens": best[0] if best else None,
            "adjacency": "loose" if best else "word_only",
            "_tok": wtok,
        })
    return out


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

    for div in root.xpath("//tei:div[@type='edition']", namespaces=NS):
        for _tag, block in S2.iter_blocks(div):
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
            if not grains or not MONEY_RE.search(value_norm):
                continue

            for g in grains:
                mention_id += 1
                local_p = [p for p in prices if abs(p.tok - g.tok) <= window_tokens]
                local_q = [q for q in qtys if abs(q.tok - g.tok) <= window_tokens]

                # Several patterns can match the same span (num-unit vs
                # unit-num); collapse those to one option.
                q_opts, seen_q, q_strict_spans = [], set(), set()
                for q in sorted(local_q, key=lambda x: abs(x.tok - g.tok)):
                    key = (q.char_start, q.char_end, q.value, q.unit)
                    if key in seen_q:
                        continue
                    seen_q.add(key)
                    q_strict_spans.add((q.char_start, q.char_end))
                    q_opts.append({
                        "option_id": f"q{len(q_opts)}",
                        "value": q.value,
                        "unit": q.unit,
                        "char_start": q.char_start,
                        "char_end": q.char_end,
                        "token_index": q.tok,
                        "dist_from_grain": abs(q.tok - g.tok),
                        "lines_crossed": count_lines_crossed(
                            tok_norm.tokens, g.tok, q.tok),
                        "crosses_block_boundary": crosses_block(
                            tok_norm.tokens, g.tok, q.tok),
                        "adjacency": "strict",
                        "gap_tokens": 0,
                    })

                for lo_opt in loose_options(UNIT_RE, "unit", value_norm, tok_norm,
                                            g.tok, window_tokens, q_strict_spans):
                    qtok = lo_opt.pop("_tok")
                    q_opts.append({
                        "option_id": f"q{len(q_opts)}",
                        **lo_opt,
                        "token_index": qtok,
                        "dist_from_grain": abs(qtok - g.tok),
                        "lines_crossed": count_lines_crossed(
                            tok_norm.tokens, g.tok, qtok),
                        "crosses_block_boundary": crosses_block(
                            tok_norm.tokens, g.tok, qtok),
                    })

                m_opts, seen_m = [], set()
                strict_spans = set()
                for p in sorted(local_p, key=lambda x: abs(x.tok - g.tok)):
                    key = (p.char_start, p.char_end, p.value, p.cur)
                    if key in seen_m:
                        continue
                    seen_m.add(key)
                    strict_spans.add((p.char_start, p.char_end))
                    resolved = resolve_compound(p.cur)
                    m_opts.append({
                        "option_id": f"m{len(m_opts)}",
                        "value": p.value,
                        "currency": p.cur,
                        # Non-null only for compound denominations that encode
                        # their own amount (τριώβολον = 3 obols).
                        "intrinsic_value": resolved[0] if resolved else None,
                        "char_start": p.char_start,
                        "char_end": p.char_end,
                        "token_index": p.tok,
                        "dist_from_grain": abs(p.tok - g.tok),
                        "lines_crossed": count_lines_crossed(
                            tok_norm.tokens, g.tok, p.tok),
                        "crosses_block_boundary": crosses_block(
                            tok_norm.tokens, g.tok, p.tok),
                        "flag_qty_price_value_collision":
                            p.flag_qty_value_collision,
                        "adjacency": "strict",
                        "gap_tokens": 0,
                    })

                for lo_opt in loose_options(MONEY_RE, "currency", value_norm,
                                            tok_norm, g.tok, window_tokens,
                                            strict_spans):
                    mtok = lo_opt.pop("_tok")
                    resolved = resolve_compound(lo_opt["currency"])
                    m_opts.append({
                        "option_id": f"m{len(m_opts)}",
                        **lo_opt,
                        "token_index": mtok,
                        "intrinsic_value": resolved[0] if resolved else None,
                        "dist_from_grain": abs(mtok - g.tok),
                        "lines_crossed": count_lines_crossed(
                            tok_norm.tokens, g.tok, mtok),
                        "crosses_block_boundary": crosses_block(
                            tok_norm.tokens, g.tok, mtok),
                        "flag_qty_price_value_collision": False,
                    })

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
                    norm_start, norm_end = option["char_start"], option["char_end"]
                    surface_start, surface_end = mapped_text.original_span(norm_start, norm_end)
                    option["norm_char_start_debug"] = norm_start
                    option["norm_char_end_debug"] = norm_end
                    option["char_start"] = surface_start
                    option["char_end"] = surface_end

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


def resolve_compound(cur: str):
    """Intrinsic value of a compound denomination (τριώβολον → 3 obols)."""
    from grain_prices.pipeline_shared import resolve_money_token
    return resolve_money_token(cur)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ddb-dir", type=Path, default=DEFAULT_DDB)
    ap.add_argument("--hgv-dir", type=Path, default=DEFAULT_HGV)
    ap.add_argument("--hgv-cache", type=Path,
                    default=Path("data/interim/hgv_index_cache.json"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--window-tokens", type=int, default=80,
                    help="max token distance from the grain mention "
                         "(matches Stage 2's default so recall is not narrower)")
    ap.add_argument("--evidence-chars", type=int, default=60,
                    help="context kept either side of the linked span")
    ap.add_argument("--max-evidence-chars", type=int, default=1200,
                    help="hard cap on the evidence excerpt per mention")
    ap.add_argument("--no-hgv", action="store_true")
    ap.add_argument("--log-every", type=int, default=2000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    hgv: Dict[str, HgvMeta] = {}
    if not args.no_hgv:
        if args.hgv_cache.exists():
            with open(args.hgv_cache, encoding="utf-8") as f:
                hgv = {k: HgvMeta(**v) for k, v in json.load(f).items()}
            log.info("HGV index from cache: %d entries", len(hgv))
        elif args.hgv_dir.exists():
            hgv = build_hgv_index(args.hgv_dir)
            args.hgv_cache.parent.mkdir(parents=True, exist_ok=True)
            with open(args.hgv_cache, "w", encoding="utf-8") as f:
                json.dump({k: vars(v) for k, v in hgv.items()}, f, ensure_ascii=False)

    pats = S2.compile_patterns()
    files = list(iter_xml_files(args.ddb_dir))
    if args.limit:
        files = files[: args.limit]
    log.info("enumerating candidates over %d files", len(files))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_rows = 0
    n_docs = 0
    opts_per_mention: List[int] = []
    combos_avoided = 0

    with open(args.out, "w", encoding="utf-8", newline="\n") as fout:
        for i, fp in enumerate(files, 1):
            if i % args.log_every == 0:
                log.info("%d/%d files, %d mentions", i, len(files), n_rows)
            rows = enumerate_for_doc(fp, pats, HgvMeta(), args.window_tokens,
                                     args.evidence_chars, args.max_evidence_chars)
            if not rows:
                continue
            meta = hgv.get(rows[0]["ddb_id"])
            n_docs += 1
            for r in rows:
                if meta:
                    r["place"] = meta.place
                    r["date_not_before"] = meta.date_notbefore
                    r["date_not_after"] = meta.date_notafter
                opts_per_mention.append(r["n_qty_options"] + r["n_money_options"])
                combos_avoided += r["n_combinations"]
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_rows += 1

    log.info("done: %d docs, %d mentions in %.1f min",
             n_docs, n_rows, (time.time() - t0) / 60)
    if opts_per_mention:
        v = sorted(opts_per_mention)
        log.info("options per mention: mean=%.1f median=%d p99=%d max=%d",
                 sum(v) / len(v), v[len(v) // 2], v[int(len(v) * 0.99)], v[-1])
        log.info("total options=%d vs %d Cartesian combinations (%.1fx smaller)",
                 sum(v), combos_avoided, combos_avoided / max(sum(v), 1))

    csv_path = args.out.with_suffix(".csv")
    with open(args.out, encoding="utf-8") as fin, \
            open(csv_path, "w", encoding="utf-8-sig", newline="") as fcsv:
        wtr = csv.DictWriter(fcsv, fieldnames=FIELDNAMES, extrasaction="ignore")
        wtr.writeheader()
        for line in fin:
            rec = json.loads(line)
            rec["evidence_text"] = " ".join(str(rec["evidence_text"]).split())
            rec["context_tags"] = ",".join(rec.get("context_tags", []))
            rec["cue_flags"] = ",".join(rec.get("cue_flags", []))
            rec["qty_options"] = " | ".join(
                f"{o['option_id']}:{o['value']} {o['unit']}@{o['dist_from_grain']}"
                for o in rec["qty_options"])
            rec["money_options"] = " | ".join(
                f"{o['option_id']}:{o['value']} {o['currency']}@{o['dist_from_grain']}"
                for o in rec["money_options"])
            wtr.writerow(rec)
    log.info("wrote %s and %s", args.out, csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

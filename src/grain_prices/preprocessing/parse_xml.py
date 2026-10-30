#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XML/TEI preprocessing module.

Extracts text, <num> tag values, line spans, and metadata from EpiDoc XML
documents (DDbDP and HGV corpora).

Refactored from Stage 1 (1_harvest_candidates.py) and Stage 2B
(2b_llm_extract_prices.py) to provide a shared, clean interface.

Usage:
    from grain_prices.preprocessing.parse_xml import (
        parse_ddb_document, parse_hgv_meta, build_hgv_index,
        extract_text_from_block, iter_text_blocks,
    )
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from lxml import etree

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from grain_prices.pipeline_shared import NS, normalize_for_search

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────

# Elements that hold competing readings of the SAME stretch of text. Flattening
# them concatenates every branch into a glued nonsense token
# (ΚοσμᾶΚοσμᾶς, ἀρτάβηςαρταβη, ΜεσορὴΜησορὴ), which is what the LLM and the
# human reviewer end up reading. Keep one branch, preferring the reading the
# editor accepted.
#   choice: <reg>/<orig> regularisation, <corr>/<sic> correction   — 42.5% of docs
#   app:    <lem>/<rdg>  editorial apparatus                       — 17.5% of docs
#   subst:  <add>/<del>  the scribe corrected himself              —  8.9% of docs
# A standalone <del> outside <subst> is left in place: that is a content
# question (should struck-through text count?), not a duplication artefact.
_VARIANT_PREFERENCE = {
    "choice": ("reg", "corr", "expan"),
    "app": ("lem", "rdg"),
    "subst": ("add",),
}


def safe_string(node) -> str:
    """Extract text content from an XML node, returning '' on failure."""
    if node is None:
        return ""
    try:
        return node.xpath("string()").strip()
    except Exception:
        return ""


def iter_xml_files(root: Path):
    """Yield all .xml file paths under *root*."""
    for fp in root.rglob("*.xml"):
        if fp.is_file():
            yield fp


# ──────────────────────────────────────────────
# HGV metadata
# ──────────────────────────────────────────────

@dataclass
class HgvMeta:
    """Metadata from an HGV EpiDoc file."""
    title: str = ""
    place: str = ""
    date_text: str = ""
    date_when: str = ""
    date_notbefore: str = ""
    date_notafter: str = ""


_HGV_DATE_RE = re.compile(r"^(-?)(\d{1,4})(?:-(\d{2}))?(?:-(\d{2}))?$")

# One date bound: (key, HGV's original string). None means the bound is open.
_Bound = Optional[Tuple[Tuple[int, int, int], str]]


def _date_key(value: str, upper: bool = False) -> Optional[Tuple[int, int, int]]:
    """
    Signed (year, month, day) sort key for an HGV date string (#043).

    HGV writes "-0010-12-17", "0155", "-0103-10"; BCE strings do not sort
    lexically. A partial date is a span, so a missing month/day fills to the
    start of that span for a lower bound and to its end for an upper bound:
    "0155" sorts before "0155-02-04" as a lower bound and after it as an upper.
    Returns None for a string that is not an HGV date.
    """
    m = _HGV_DATE_RE.match(value.strip())
    if not m:
        return None
    fill = 99 if upper else 0
    year = -int(m.group(2)) if m.group(1) else int(m.group(2))
    month = int(m.group(3)) if m.group(3) else fill
    day = int(m.group(4)) if m.group(4) else fill
    return year, month, day


def _bound(value: str, upper: bool) -> _Bound:
    key = _date_key(value, upper=upper) if value else None
    return (key, value) if key is not None else None


def _envelope(spans: List[Tuple[_Bound, _Bound]]) -> Tuple[str, str]:
    """
    Earliest lower bound to latest upper bound over date spans (#043).

    A span open on one side (only @notBefore, or only @notAfter) leaves the
    envelope open on that side; it is not filled with a guess. Spans with no
    usable bound at all are ignored.
    """
    spans = [(lo, hi) for lo, hi in spans if lo is not None or hi is not None]
    if not spans:
        return "", ""
    los = [lo for lo, _ in spans]
    his = [hi for _, hi in spans]
    lo = "" if any(b is None for b in los) else min(los, key=lambda b: b[0])[1]
    hi = "" if any(b is None for b in his) else max(his, key=lambda b: b[0])[1]
    return lo, hi


def _meta_span(meta: HgvMeta) -> Tuple[_Bound, _Bound]:
    return _bound(meta.date_notbefore, False), _bound(meta.date_notafter, True)


def parse_hgv_meta(root: etree._Element) -> Tuple[str, HgvMeta]:
    """
    Extract ddb-hybrid id and metadata fields from an HGV EpiDoc tree.

    ``date_notbefore``/``date_notafter`` hold the envelope over every
    ``origDate`` alternative, with ``@when`` counting as both bounds (#043).
    ``date_text`` and ``date_when`` keep the first ``origDate`` only.
    """
    ddb_id = root.xpath(
        'string(//tei:idno[@type="ddb-hybrid"][1])', namespaces=NS
    ).strip()
    if not ddb_id:
        ddb_id = root.xpath("string(//tei:idno[1])", namespaces=NS).strip()

    title = root.xpath("string(//tei:titleStmt/tei:title[1])", namespaces=NS).strip()
    place = root.xpath("string(//tei:origPlace[1])", namespaces=NS).strip()
    date_text = root.xpath("string(//tei:origDate[1])", namespaces=NS).strip()
    date_when = root.xpath("string(//tei:origDate[1]/@when)", namespaces=NS).strip()

    spans = []
    for od in root.xpath("//tei:origDate", namespaces=NS):
        when = (od.get("when") or "").strip()
        nb = (od.get("notBefore") or "").strip() or when
        na = (od.get("notAfter") or "").strip() or when
        spans.append((_bound(nb, upper=False), _bound(na, upper=True)))
    date_nb, date_na = _envelope(spans)

    return ddb_id, HgvMeta(
        title=title, place=place, date_text=date_text,
        date_when=date_when, date_notbefore=date_nb, date_notafter=date_na,
    )


def _merge_hgv_meta(records: List[HgvMeta]) -> HgvMeta:
    """
    Merge HGV records that share one ddb-hybrid id (#043).

    Dates become the envelope over all records; differing places are
    de-duplicated and joined with " / ". Title and the first-``origDate``
    fields come from the first record in sorted path order.
    """
    first = records[0]
    if len(records) == 1:
        return first
    places: List[str] = []
    for r in records:
        if r.place and r.place not in places:
            places.append(r.place)
    date_nb, date_na = _envelope([_meta_span(r) for r in records])
    return HgvMeta(
        title=first.title, place=" / ".join(places), date_text=first.date_text,
        date_when=first.date_when, date_notbefore=date_nb, date_notafter=date_na,
    )


def build_hgv_index(
    hgv_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, HgvMeta]:
    """
    Build a ddb_id → HgvMeta index from the HGV metadata directory.

    Files are read in sorted relative-path order so the result does not depend
    on filesystem traversal order (#026), and records sharing an id are merged
    rather than overwritten (#043).
    """
    if logger is None:
        logger = log
    records: Dict[str, List[HgvMeta]] = {}
    total, kept = 0, 0
    files = sorted(
        iter_xml_files(hgv_dir),
        key=lambda path: path.relative_to(hgv_dir).as_posix(),
    )
    for fp in files:
        total += 1
        try:
            root = etree.parse(str(fp)).getroot()
            ddb_id, meta = parse_hgv_meta(root)
            if ddb_id:
                records.setdefault(ddb_id, []).append(meta)
                kept += 1
        except Exception as e:
            logger.debug("HGV parse failed: %s (%s)", fp, e)
    idx = {k: _merge_hgv_meta(v) for k, v in records.items()}
    merged = sum(1 for v in records.values() if len(v) > 1)
    logger.info(
        "HGV index built: %d/%d files with ddb-hybrid id, %d ids (%d merged from several records).",
        kept, total, len(idx), merged,
    )
    return idx


HGV_INDEX_SCHEMA = "hgv-index-v2"
DEFAULT_HGV_CACHE = Path("data/interim/hgv_index_cache_v2.json")


def load_hgv_index(
    hgv_dir: Path,
    cache_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, HgvMeta]:
    """
    HGV index with an on-disk cache.

    Parsing the ~65k HGV files takes roughly 50 minutes, and every stage that
    wants place/date metadata was paying it again on every run. The HGV corpus
    is static reference data, so the result is cached as JSON.

    Delete the cache file to force a rebuild. A cache written under another
    ``HGV_INDEX_SCHEMA`` (including the unversioned pre-#043 format) is
    rebuilt, so a change to the parsing rules cannot be hidden by a stale file.
    """
    if logger is None:
        logger = log
    cache = Path(cache_path) if cache_path else DEFAULT_HGV_CACHE
    if cache.exists():
        try:
            with open(cache, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("schema") == HGV_INDEX_SCHEMA:
                idx = {k: HgvMeta(**v) for k, v in data["index"].items()}
                logger.info("HGV index from cache: %d entries (%s)", len(idx), cache)
                return idx
            logger.warning("HGV cache %s is not schema %s; rebuilding", cache, HGV_INDEX_SCHEMA)
        except Exception as e:
            logger.warning("HGV cache unreadable (%s); rebuilding", e)
    idx = build_hgv_index(hgv_dir, logger)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": HGV_INDEX_SCHEMA, "index": {k: vars(v) for k, v in idx.items()}}
        with open(cache, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False)
        logger.info("HGV index cached to %s", cache)
    except Exception as e:
        logger.warning("could not write HGV cache: %s", e)
    return idx


# ──────────────────────────────────────────────
# DDB text extraction
# ──────────────────────────────────────────────

def get_ddb_id(root: etree._Element) -> str:
    """Extract the ddb-hybrid ID from a DDB EpiDoc document."""
    ddb_id = root.xpath(
        'string(//tei:idno[@type="ddb-hybrid"][1])', namespaces=NS
    ).strip()
    if ddb_id:
        return ddb_id
    return root.xpath("string(//tei:idno[1])", namespaces=NS).strip()


def extract_text_from_block(
    elem: etree._Element,
    lb_mark: str = " | ",
    cb_mark: str = " || ",
    pb_mark: str = " || ",
    gap_mark: str = " ",
) -> str:
    """
    Extract readable text from an EpiDoc XML block element.

    The break markers are parameterised so every stage shares this one
    implementation: Stage 2 needs distinct boundary sentinel tokens for its
    cross-line scoring penalty, while Stage 1/2B want plain " | " separators.
    Stage 2 previously kept a private copy of this function, which is how the
    word-splitting and <num>-discarding bugs below survived in one code path
    after being fixed in the other.

    - <num value="25">κε</num> → "25" (prefer numeric value attribute).
      When the tag also carries an editorial expansion — e.g.
      <num value="3"><expan><ex>τριώβολον</ex></expan></num> — both are
      emitted ("3 τριώβολον") so the currency word survives; without this
      the amount reads as a bare quantity with no currency.
    - Editorial markup (<supplied>, <unclear>, <expan>, <ex>, …) is flattened.
      Fragments are concatenated with NO inserted separator: papyrus editions
      split words across this markup (<expan>ὑπ<ex>ὲρ</ex></expan>), and
      joining with a space would break ὑπὲρ into two tokens and defeat the
      \\b-anchored vocabulary patterns. Inter-word spacing already lives in
      the XML's own text/tail nodes.
    - <choice> → one branch only, preferring the regularised/corrected reading.
      Flattening all branches concatenates them into a glued nonsense token
      (<choice><reg>Κοσμᾶ</reg><orig>Κοσμᾶς</orig></choice> → "ΚοσμᾶΚοσμᾶς",
      ἀρτάβηςαρταβη, τιμῆςτειμῆς). 42% of documents contain a <choice>, so this
      is what the LLM and the human reviewer would otherwise be reading.
    - <lb/> → " | " (line break marker)
    - <cb/>, <pb/> → " || " (column/page break)
    - <gap/> → " " — a lacuna must not fuse the words on either side of it.
    """
    pieces: List[str] = []

    def walk(node: etree._Element):
        if node.text:
            pieces.append(node.text)
        for ch in node:
            ln = etree.QName(ch).localname
            if ln in _VARIANT_PREFERENCE:
                chosen = None
                for want in _VARIANT_PREFERENCE[ln]:
                    found = ch.find(f"{{{NS['tei']}}}{want}")
                    if found is not None:
                        chosen = found
                        break
                if chosen is None and len(ch):
                    chosen = ch[0]
                if chosen is not None:
                    walk(chosen)
            elif ln == "num":
                v = (ch.get("value") or "").strip()
                # Only the *editorial expansion* is worth keeping alongside the
                # value; bare numeral letters (η, ιδ) would just add noise.
                has_expansion = bool(
                    ch.xpath(".//tei:ex | .//tei:expan", namespaces=NS)
                )
                t = safe_string(ch) if has_expansion else ""
                if v and t:
                    pieces.append(f" {v} {t} ")
                elif v:
                    pieces.append(f" {v} ")
                elif t:
                    pieces.append(f" {t} ")
                else:
                    t = safe_string(ch)
                    if t:
                        pieces.append(f" {t} ")
            elif ln == "lb":
                pieces.append(lb_mark)
            elif ln == "cb":
                pieces.append(cb_mark)
            elif ln == "pb":
                pieces.append(pb_mark)
            elif ln == "gap":
                pieces.append(gap_mark)
            else:
                walk(ch)
            if ch.tail:
                pieces.append(ch.tail)

    walk(elem)
    return " ".join("".join(pieces).split())


def iter_text_blocks(
    tree_or_root,
    div_types: Tuple[str, ...] = ("edition",),
) -> List[Tuple[str, str]]:
    """
    Return (block_tag, text) for every text-bearing block in the document body.

    Args:
        tree_or_root: lxml ElementTree or root Element
        div_types: which <div type="..."> sections to search
    """
    if isinstance(tree_or_root, etree._ElementTree):
        root = tree_or_root.getroot()
    else:
        root = tree_or_root

    results = []
    xpath_parts = [f"//tei:div[@type='{dt}']" for dt in div_types]
    xpath_str = " | ".join(xpath_parts)

    for div in root.xpath(xpath_str, namespaces=NS):
        for b in div.xpath(
            ".//tei:ab | .//tei:p | .//tei:seg | .//tei:l", namespaces=NS
        ):
            tag = etree.QName(b).localname
            text = extract_text_from_block(b)
            if text.strip():
                results.append((tag, text))
    return results


def extract_num_tags(block: etree._Element) -> List[Dict[str, str]]:
    """Extract all <num value="..."> tags from a block element."""
    nums = []
    for n in block.xpath(".//tei:num[@value]", namespaces=NS):
        v = (n.get("value") or "").strip()
        t = safe_string(n)
        if v:
            nums.append({"value": v, "text": t})
    return nums


# ──────────────────────────────────────────────
# High-level document parser
# ──────────────────────────────────────────────

@dataclass
class ParsedDocument:
    """Parsed representation of a DDB EpiDoc document."""
    ddb_id: str
    xml_path: str
    blocks: List[Tuple[str, str]] = field(default_factory=list)
    full_text: str = ""
    norm_text: str = ""
    num_tag_values: List[str] = field(default_factory=list)
    metadata: Optional[HgvMeta] = None


def parse_ddb_document(
    xml_path: Path,
    hgv_index: Optional[Dict[str, HgvMeta]] = None,
    div_types: Tuple[str, ...] = ("edition",),
) -> Optional[ParsedDocument]:
    """
    Parse a DDB EpiDoc XML file into a structured document.

    Returns None if the file cannot be parsed or has no text content.
    """
    try:
        tree = etree.parse(str(xml_path))
    except Exception as e:
        log.debug("XML parse error: %s (%s)", xml_path, e)
        return None

    root = tree.getroot()
    ddb_id = get_ddb_id(root)
    if not ddb_id:
        return None

    blocks = iter_text_blocks(tree, div_types=div_types)
    if not blocks:
        return None

    full_text = " ||| ".join(text for _, text in blocks)
    norm_text = normalize_for_search(full_text)

    # Collect all num tag values
    num_vals = []
    xpath_parts = [f"//tei:div[@type='{dt}']" for dt in div_types]
    for div in root.xpath(" | ".join(xpath_parts), namespaces=NS):
        for b in div.xpath(".//tei:ab | .//tei:p | .//tei:seg | .//tei:l", namespaces=NS):
            for n in b.xpath(".//tei:num[@value]", namespaces=NS):
                v = (n.get("value") or "").strip()
                if v:
                    num_vals.append(v)

    meta = None
    if hgv_index and ddb_id in hgv_index:
        meta = hgv_index[ddb_id]

    return ParsedDocument(
        ddb_id=ddb_id,
        xml_path=str(xml_path),
        blocks=blocks,
        full_text=full_text,
        norm_text=norm_text,
        num_tag_values=num_vals,
        metadata=meta,
    )

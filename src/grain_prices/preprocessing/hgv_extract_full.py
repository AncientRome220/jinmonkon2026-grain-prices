#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive HGV EpiDoc metadata extractor.

Extracts the full set of metadata fields from HGV TEI/EpiDoc XML files into
a normalized multi-table CSV output (one main table + five 1:many detail
tables) plus three data-quality reports.

This module is intentionally separate from `parse_xml.parse_hgv_meta()`,
which the existing rule-based pipeline uses to look up a minimal 6-field
metadata blob keyed by `ddb-hybrid`. The full extractor here is keyed by
`hgv_id` (the native HGV filename ID) because not every HGV file has a
`ddb-hybrid`.

Outputs (default `data/hgv_full/`):
    papyri.csv             — main table, 1 row per file
    languages.csv          — 1:many (translation languages)
    keywords.csv           — 1:many (HGV keyword scheme)
    ancient_places.csv     — 1:many (ancient placeName with subtype/ref)
    bibliography.csv       — 1:many (principal / corrections / other)
    mentioned_dates.csv    — 1:many (dates cited in commentary body)
    coverage_report.csv    — quality: main-table column coverage
    detail_table_stats.csv — quality: 1:many table summary
    parse_errors.csv       — quality: files that failed to parse

Usage:
    python -m grain_prices.preprocessing.hgv_extract_full \
        --hgv-dir C:\\research\\HGV_meta_EpiDoc \
        --out-dir data\\hgv_full \
        --workers 8 [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from grain_prices.pipeline_shared import NS
from grain_prices.preprocessing.parse_xml import iter_xml_files, safe_string

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────

PAPYRI_COLUMNS = [
    "hgv_id", "tm_id",
    "ddb_hybrid", "ddb_filename", "ddb_perseus",
    "title", "material",
    "origPlace_text", "settlement", "collection", "invNo",
    "date_text", "date_when", "date_notBefore", "date_notAfter", "date_precision",
    "dim_height", "dim_width", "dim_unit",
    "has_commentary", "has_archive_note",
]

DETAIL_TABLES = {
    "languages":       ["hgv_id", "language_code"],
    "keywords":        ["hgv_id", "keyword", "scheme"],
    "ancient_places":  ["hgv_id", "place_text", "subtype", "ref_uri", "key", "cert"],
    "bibliography":    ["hgv_id", "bibl_type", "title", "volume", "numbers",
                        "pages", "source", "target", "raw_text"],
    "mentioned_dates": ["hgv_id", "ref", "date_when", "date_notBefore",
                        "date_notAfter", "precision", "text"],
}

TABLE_NAMES = ["papyri"] + list(DETAIL_TABLES.keys())


@dataclass
class ParseError:
    file_path: str
    error_type: str
    message: str


@dataclass
class HgvFullRecord:
    papyri: Dict[str, Any]
    languages: List[Dict[str, Any]] = field(default_factory=list)
    keywords: List[Dict[str, Any]] = field(default_factory=list)
    ancient_places: List[Dict[str, Any]] = field(default_factory=list)
    bibliography: List[Dict[str, Any]] = field(default_factory=list)
    mentioned_dates: List[Dict[str, Any]] = field(default_factory=list)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _xp_str(root, xpath: str) -> str:
    """XPath string() evaluation, stripped, with NS bound."""
    try:
        return root.xpath(f"string({xpath})", namespaces=NS).strip()
    except Exception:
        return ""


def _xp_nodes(root, xpath: str):
    try:
        return root.xpath(xpath, namespaces=NS)
    except Exception:
        return []


# ──────────────────────────────────────────────
# Per-field parsers
# ──────────────────────────────────────────────

def parse_main(root, hgv_id: str) -> Dict[str, Any]:
    has_commentary = bool(_xp_nodes(root, '//tei:div[@type="commentary"]'))
    commentary_text = " ".join(
        safe_string(n) for n in _xp_nodes(root, '//tei:div[@type="commentary"]//tei:p')
    )
    has_archive_note = "Archiv" in commentary_text or "archive" in commentary_text.lower()

    return {
        "hgv_id":         hgv_id,
        "tm_id":          _xp_str(root, '//tei:idno[@type="TM"][1]'),
        "ddb_hybrid":     _xp_str(root, '//tei:idno[@type="ddb-hybrid"][1]'),
        "ddb_filename":   _xp_str(root, '//tei:idno[@type="ddb-filename"][1]'),
        "ddb_perseus":    _xp_str(root, '//tei:idno[@type="ddb-perseus-style"][1]'),
        "title":          _xp_str(root, '//tei:titleStmt/tei:title[1]'),
        "material":       _xp_str(root, '//tei:support/tei:material[1]'),
        "origPlace_text": _xp_str(root, '//tei:origPlace[1]'),
        "settlement":     _xp_str(root, '//tei:msIdentifier//tei:settlement[1]'),
        "collection":     _xp_str(root, '//tei:msIdentifier/tei:collection[1]'),
        "invNo":          _xp_str(root, '//tei:msIdentifier/tei:idno[@type="invNo"][1]'),
        "date_text":      _xp_str(root, '//tei:origDate[1]'),
        "date_when":      _xp_str(root, '//tei:origDate[1]/@when'),
        "date_notBefore": _xp_str(root, '//tei:origDate[1]/@notBefore'),
        "date_notAfter":  _xp_str(root, '//tei:origDate[1]/@notAfter'),
        "date_precision": _xp_str(root, '//tei:origDate[1]/@precision'),
        "dim_height":     _xp_str(root, '//tei:dimensions/tei:height[1]'),
        "dim_width":      _xp_str(root, '//tei:dimensions/tei:width[1]'),
        "dim_unit":       _xp_str(root, '//tei:dimensions/@unit'),
        "has_commentary": "1" if has_commentary else "0",
        "has_archive_note": "1" if has_archive_note else "0",
    }


def parse_languages(root, hgv_id: str) -> List[Dict[str, Any]]:
    rows = []
    for node in _xp_nodes(root, '//tei:langUsage/tei:language'):
        code = (node.get("ident") or "").strip()
        if code:
            rows.append({"hgv_id": hgv_id, "language_code": code})
    return rows


def parse_keywords(root, hgv_id: str) -> List[Dict[str, Any]]:
    rows = []
    for kw_block in _xp_nodes(root, '//tei:textClass/tei:keywords'):
        scheme = (kw_block.get("scheme") or "").strip()
        for term in kw_block.xpath('tei:term', namespaces=NS):
            text = safe_string(term)
            if text:
                rows.append({"hgv_id": hgv_id, "keyword": text, "scheme": scheme})
    return rows


def parse_ancient_places(root, hgv_id: str) -> List[Dict[str, Any]]:
    rows = []
    for pn in _xp_nodes(root, '//tei:placeName[@type="ancient"]'):
        rows.append({
            "hgv_id":     hgv_id,
            "place_text": safe_string(pn),
            "subtype":    (pn.get("subtype") or "").strip(),
            "ref_uri":    (pn.get("ref") or "").strip(),
            "key":        (pn.get("key") or "").strip(),
            "cert":       (pn.get("cert") or "").strip(),
        })
    return rows


def parse_bibliography(root, hgv_id: str) -> List[Dict[str, Any]]:
    rows = []
    for div in _xp_nodes(root, '//tei:div[@type="bibliography"]'):
        bibl_type = (div.get("subtype") or "").strip()
        for bibl in div.xpath('.//tei:bibl', namespaces=NS):
            title = ""
            for t in bibl.xpath('tei:title', namespaces=NS):
                title = safe_string(t)
                break
            volume = _xp_str(bibl, 'tei:biblScope[@type="volume"][1]')
            numbers = _xp_str(bibl, 'tei:biblScope[@type="numbers"][1]')
            pages = _xp_str(bibl, 'tei:biblScope[@type="pages"][1]')
            source = (bibl.get("type") or "").strip()
            target = ""
            ptr = bibl.xpath('tei:ptr', namespaces=NS)
            if ptr:
                target = (ptr[0].get("target") or "").strip()
            raw_text = safe_string(bibl)
            if not (title or volume or numbers or pages or raw_text or target):
                continue
            rows.append({
                "hgv_id":   hgv_id,
                "bibl_type": bibl_type,
                "title":    title,
                "volume":   volume,
                "numbers":  numbers,
                "pages":    pages,
                "source":   source,
                "target":   target,
                "raw_text": raw_text,
            })
    return rows


def parse_mentioned_dates(root, hgv_id: str) -> List[Dict[str, Any]]:
    rows = []
    for item in _xp_nodes(
        root, '//tei:div[@subtype="mentionedDates"]//tei:item'
    ):
        ref = _xp_str(item, 'tei:ref[1]')
        for date in item.xpath('tei:date', namespaces=NS):
            rows.append({
                "hgv_id":         hgv_id,
                "ref":            ref,
                "date_when":      (date.get("when") or "").strip(),
                "date_notBefore": (date.get("notBefore") or "").strip(),
                "date_notAfter":  (date.get("notAfter") or "").strip(),
                "precision":      (date.get("precision") or "").strip(),
                "text":           safe_string(date),
            })
    for date in _xp_nodes(
        root,
        '//tei:div[@subtype="mentionedDates"]/tei:note//tei:date '
        '| //tei:div[@subtype="mentionedDates"]/tei:date'
    ):
        rows.append({
            "hgv_id":         hgv_id,
            "ref":            "",
            "date_when":      (date.get("when") or "").strip(),
            "date_notBefore": (date.get("notBefore") or "").strip(),
            "date_notAfter":  (date.get("notAfter") or "").strip(),
            "precision":      (date.get("precision") or "").strip(),
            "text":           safe_string(date),
        })
    return rows


# ──────────────────────────────────────────────
# Per-file driver
# ──────────────────────────────────────────────

def parse_one_file(path: Path) -> Union[HgvFullRecord, ParseError]:
    try:
        tree = etree.parse(str(path))
    except etree.XMLSyntaxError as e:
        return ParseError(str(path), "XmlSyntaxError", str(e))
    except Exception as e:
        return ParseError(str(path), type(e).__name__, str(e))

    root = tree.getroot()
    hgv_id = _xp_str(root, '//tei:idno[@type="filename"][1]')
    if not hgv_id:
        return ParseError(str(path), "MissingHgvId",
                          "no <idno type='filename'> found")

    try:
        return HgvFullRecord(
            papyri          = parse_main(root, hgv_id),
            languages       = parse_languages(root, hgv_id),
            keywords        = parse_keywords(root, hgv_id),
            ancient_places  = parse_ancient_places(root, hgv_id),
            bibliography    = parse_bibliography(root, hgv_id),
            mentioned_dates = parse_mentioned_dates(root, hgv_id),
        )
    except Exception as e:
        return ParseError(str(path), type(e).__name__, str(e))


# Pool-friendly wrapper: returns (result, file_path) so failures keep file info
def _parse_worker(path_str: str):
    return parse_one_file(Path(path_str))


# ──────────────────────────────────────────────
# Writers
# ──────────────────────────────────────────────

def write_csv(out_path: Path, columns: List[str], rows: List[Dict[str, Any]]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_main_and_details(out_dir: Path, agg: Dict[str, List[Dict[str, Any]]]):
    write_csv(out_dir / "papyri.csv", PAPYRI_COLUMNS, agg["papyri"])
    for tbl, cols in DETAIL_TABLES.items():
        write_csv(out_dir / f"{tbl}.csv", cols, agg[tbl])


def write_coverage_report(out_dir: Path, agg: Dict[str, List[Dict[str, Any]]]):
    rows = agg["papyri"]
    total = len(rows)
    report = []
    for col in PAPYRI_COLUMNS:
        values = [r.get(col, "") for r in rows]
        non_null = [v for v in values if v not in ("", None)]
        counter = Counter(non_null)
        examples = " | ".join(f"{v}" for v, _ in counter.most_common(3))
        report.append({
            "column":          col,
            "non_null_count":  len(non_null),
            "coverage_pct":    f"{(len(non_null) / total * 100):.2f}" if total else "0.00",
            "distinct_values": len(counter),
            "top_examples":    examples,
        })
    write_csv(
        out_dir / "coverage_report.csv",
        ["column", "non_null_count", "coverage_pct", "distinct_values", "top_examples"],
        report,
    )


def write_detail_stats(out_dir: Path, agg: Dict[str, List[Dict[str, Any]]]):
    n_papyri = len(agg["papyri"])
    stats = []
    for tbl in DETAIL_TABLES:
        rows = agg[tbl]
        per_hgv = Counter(r["hgv_id"] for r in rows)
        stats.append({
            "table":             tbl,
            "total_rows":        len(rows),
            "distinct_hgv_ids":  len(per_hgv),
            "avg_rows_per_hgv":  f"{(len(rows) / n_papyri):.3f}" if n_papyri else "0.000",
            "max_rows_per_hgv":  max(per_hgv.values()) if per_hgv else 0,
        })
    write_csv(
        out_dir / "detail_table_stats.csv",
        ["table", "total_rows", "distinct_hgv_ids", "avg_rows_per_hgv", "max_rows_per_hgv"],
        stats,
    )


def write_parse_errors(out_dir: Path, errors: List[ParseError]):
    write_csv(
        out_dir / "parse_errors.csv",
        ["file_path", "error_type", "message"],
        [{"file_path": e.file_path, "error_type": e.error_type, "message": e.message}
         for e in errors],
    )


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def aggregate(results) -> Tuple[Dict[str, List[Dict[str, Any]]], List[ParseError]]:
    agg: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TABLE_NAMES}
    errors: List[ParseError] = []
    for result in results:
        if isinstance(result, ParseError):
            errors.append(result)
            continue
        agg["papyri"].append(result.papyri)
        for tbl in DETAIL_TABLES:
            agg[tbl].extend(getattr(result, tbl))
    return agg, errors


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--hgv-dir", type=Path, required=True,
                   help="Root directory of HGV EpiDoc XML files")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory for CSVs and reports")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of worker processes (default 1, recommend 4-8)")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N files (for testing)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    files = [str(p) for p in iter_xml_files(args.hgv_dir)]
    if args.limit:
        files = files[: args.limit]
    log.info("Found %d HGV XML files to parse", len(files))

    if args.workers and args.workers > 1:
        with Pool(args.workers) as pool:
            results = list(pool.imap_unordered(_parse_worker, files, chunksize=64))
    else:
        results = [_parse_worker(f) for f in files]

    agg, errors = aggregate(results)
    log.info("Parsed %d records, %d errors", len(agg["papyri"]), len(errors))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_main_and_details(args.out_dir, agg)
    write_coverage_report(args.out_dir, agg)
    write_detail_stats(args.out_dir, agg)
    write_parse_errors(args.out_dir, errors)

    print(f"\n[Summary]")
    print(f"  Files processed:   {len(files)}")
    print(f"  Records extracted: {len(agg['papyri'])}")
    print(f"  Parse errors:      {len(errors)}")
    for tbl in DETAIL_TABLES:
        print(f"  {tbl + ':':<18} {len(agg[tbl])} rows")
    print(f"\nOutput → {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

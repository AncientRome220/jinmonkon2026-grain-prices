#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the Digital Humanities class assignment submission set from
the full HGV extraction (data/hgv_full/).

Takes the grain-related slice of the full HGV extraction and produces
a normalized 6-table CSV bundle plus a Japanese README, suitable for
submission to the UTOL homework system.

Slice criterion: HGV keyword matches any of GRAIN_KEYWORD_PATTERNS
(Getreide / Weizen / Gerste / Saatdarlehen). Limited to the first
N rows by hgv_id ascending for determinism.

Usage:
    python -m grain_prices.preprocessing.build_assignment_subset \
        --hgv-full-dir data/hgv_full \
        --out-dir "<OUTPUT_DIR>" \
        --limit 200
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional


GRAIN_KEYWORD_PATTERN = re.compile(r"Getreide|Weizen|Gerste|Saatdarlehen", re.IGNORECASE)


# Materials seen most frequently in HGV. Anything not in this dict
# falls back to (name_de, "", "") so the lookup table never has
# untranslated NULL primary keys.
MATERIAL_JA_EN = {
    "Papyrus":          ("パピルス",     "Papyrus"),
    "Ostrakon":         ("オストラカ",   "Ostracon"),
    "Holz":             ("木板",         "Wood"),
    "Pergament":        ("羊皮紙",       "Parchment"),
    "Leder":            ("革",           "Leather"),
    "Stein":            ("石",           "Stone"),
    "Ton":              ("粘土",         "Clay"),
    "Wachs":            ("蝋板",         "Wax tablet"),
    "Wachstafel":       ("蝋板",         "Wax tablet"),
    "Metall":           ("金属",         "Metal"),
    "Knochen":          ("骨",           "Bone"),
    "Stoff":            ("布",           "Cloth"),
    "Mumienetikett":    ("ミイラ・ラベル", "Mummy label"),
}

NOMOS_JA_EN = {
    "Arsinoites":       ("アルシノエ",      "Arsinoite"),
    "Oxyrhynchites":    ("オクシュリュンコス", "Oxyrhynchite"),
    "Hermopolites":     ("ヘルモポリス",    "Hermopolite"),
    "Herakleopolites":  ("ヘラクレオポリス", "Herakleopolite"),
    "Memphites":        ("メンフィス",      "Memphite"),
    "Panopolites":      ("パノポリス",      "Panopolite"),
    "Pathyrites":       ("パテュリス",      "Pathyrite"),
    "Lykopolites":      ("リュコポリス",    "Lykopolite"),
    "Kynopolites":      ("キュノポリス",    "Kynopolite"),
    "Antaiopolites":    ("アンタイオポリス", "Antaiopolite"),
    "Apollonopolites":  ("アポロノポリス",  "Apollonopolite"),
    "Hermonthites":     ("ヘルモンティス",  "Hermonthite"),
    "Thebais":          ("テーバイス",      "Thebaid"),
    "Diospolites":      ("ディオスポリス",  "Diospolite"),
    "Latopolites":      ("ラトポリス",      "Latopolite"),
    "Aphroditopolites": ("アフロディトポリス", "Aphroditopolite"),
    "Saites":           ("サイス",          "Saite"),
    "Mendesios":        ("メンデス",        "Mendesian"),
}


# ──────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────

def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, columns: List[str], rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ──────────────────────────────────────────────
# Slicing
# ──────────────────────────────────────────────

def _sort_key(hgv_id: str):
    """Sort hgv_id numerically when possible, else lexicographically."""
    try:
        return (0, int(hgv_id))
    except ValueError:
        return (1, hgv_id)


def grain_hgv_ids(keywords_rows: List[Dict[str, str]], limit: int) -> List[str]:
    matched = {r["hgv_id"] for r in keywords_rows
               if GRAIN_KEYWORD_PATTERN.search(r.get("keyword", ""))}
    return sorted(matched, key=_sort_key)[:limit]


def subset_rows(rows: List[Dict[str, str]], hgv_ids: set) -> List[Dict[str, str]]:
    return [r for r in rows if r["hgv_id"] in hgv_ids]


# ──────────────────────────────────────────────
# Lookup table construction
# ──────────────────────────────────────────────

def build_material_lookup(papyri_rows: List[Dict[str, str]]):
    """Return (material_lookup_rows, name_de → material_id mapping).
       Empty material values are skipped (papyri.material_id remains '')."""
    names = sorted({r["material"] for r in papyri_rows if r.get("material")})
    lookup_rows = []
    id_map: Dict[str, str] = {}
    for i, name_de in enumerate(names, start=1):
        ja, en = MATERIAL_JA_EN.get(name_de, ("", ""))
        mid = str(i)
        id_map[name_de] = mid
        lookup_rows.append({
            "material_id": mid,
            "name_de":     name_de,
            "name_ja":     ja,
            "name_en":     en,
        })
    return lookup_rows, id_map


def build_nomos_lookup(places_rows: List[Dict[str, str]]):
    """Return (nomos_lookup_rows, (name_grc, tm_uri) → nomos_id mapping).
       Only nomos-typed rows participate; key is normalized to (text, uri)."""
    nomos_keys = set()
    for r in places_rows:
        if r.get("subtype") == "nome":
            text = (r.get("place_text") or "").strip()
            uri = (r.get("ref_uri") or "").strip()
            if text:
                nomos_keys.add((text, uri))
    sorted_keys = sorted(nomos_keys, key=lambda x: (x[0], x[1]))
    lookup_rows = []
    id_map: Dict[tuple, str] = {}
    for i, (text, uri) in enumerate(sorted_keys, start=1):
        ja, en = NOMOS_JA_EN.get(text, ("", ""))
        nid = str(i)
        id_map[(text, uri)] = nid
        lookup_rows.append({
            "nomos_id": nid,
            "name_grc": text,
            "name_ja":  ja,
            "name_en":  en,
            "tm_uri":   uri,
        })
    return lookup_rows, id_map


# ──────────────────────────────────────────────
# Table transformation
# ──────────────────────────────────────────────

PAPYRI_OUT_COLUMNS = [
    "hgv_id", "tm_id", "ddb_hybrid", "title", "material_id",
    "origPlace_text", "date_text", "date_when", "date_notBefore",
    "date_notAfter", "date_precision",
]

KEYWORDS_OUT_COLUMNS = ["hgv_id", "keyword", "scheme"]

PLACES_OUT_COLUMNS = [
    "hgv_id", "place_text", "subtype", "nomos_id", "ref_uri",
]

BIBLIOGRAPHY_OUT_COLUMNS = ["hgv_id", "title", "volume", "numbers"]

MATERIAL_LOOKUP_COLUMNS = ["material_id", "name_de", "name_ja", "name_en"]

NOMOS_LOOKUP_COLUMNS = ["nomos_id", "name_grc", "name_ja", "name_en", "tm_uri"]


def project_papyri(rows: List[Dict[str, str]], material_id_map: Dict[str, str]):
    out = []
    for r in rows:
        mat = r.get("material", "")
        out.append({
            "hgv_id":         r["hgv_id"],
            "tm_id":          r.get("tm_id", ""),
            "ddb_hybrid":     r.get("ddb_hybrid", ""),
            "title":          r.get("title", ""),
            "material_id":    material_id_map.get(mat, "") if mat else "",
            "origPlace_text": r.get("origPlace_text", ""),
            "date_text":      r.get("date_text", ""),
            "date_when":      r.get("date_when", ""),
            "date_notBefore": r.get("date_notBefore", ""),
            "date_notAfter":  r.get("date_notAfter", ""),
            "date_precision": r.get("date_precision", ""),
        })
    out.sort(key=lambda r: _sort_key(r["hgv_id"]))
    return out


def project_places(rows: List[Dict[str, str]], nomos_id_map: Dict[tuple, str]):
    out = []
    for r in rows:
        nid = ""
        if r.get("subtype") == "nome":
            key = ((r.get("place_text") or "").strip(),
                   (r.get("ref_uri") or "").strip())
            nid = nomos_id_map.get(key, "")
        out.append({
            "hgv_id":     r["hgv_id"],
            "place_text": r.get("place_text", ""),
            "subtype":    r.get("subtype", ""),
            "nomos_id":   nid,
            "ref_uri":    r.get("ref_uri", ""),
        })
    out.sort(key=lambda r: (_sort_key(r["hgv_id"]), r["place_text"]))
    return out


def project_keywords(rows: List[Dict[str, str]]):
    out = [{"hgv_id": r["hgv_id"],
            "keyword": r.get("keyword", ""),
            "scheme":  r.get("scheme", "")}
           for r in rows]
    out.sort(key=lambda r: (_sort_key(r["hgv_id"]), r["keyword"]))
    return out


def project_bibliography(rows: List[Dict[str, str]]):
    principal = [r for r in rows if r.get("bibl_type") == "principalEdition"]
    out = [{"hgv_id":  r["hgv_id"],
            "title":   r.get("title", ""),
            "volume":  r.get("volume", ""),
            "numbers": r.get("numbers", "")}
           for r in principal]
    out.sort(key=lambda r: (_sort_key(r["hgv_id"]), r["title"]))
    return out


# ──────────────────────────────────────────────
# README
# ──────────────────────────────────────────────

README_TEMPLATE = """\
# 第6回課題：複数の表で構成されるデータ（HGV実データ版）

## ファイル構成

```
第6回提出/
├── papyri.csv                 メイン表：パピルス文書（{n_papyri}行）
├── papyrus_keywords.csv       1:多 — HGVキーワード（{n_keywords}行）
├── papyrus_places.csv         1:多 — 古代地名（{n_places}行）
├── papyrus_bibliography.csv   1:多 — 主校訂書誌（{n_bibl}行）
├── material_lookup.csv        ルックアップ — 書写媒体（{n_material}行）
├── nomos_lookup.csv           ルックアップ — ノモス（{n_nomos}行）
└── README.md                  本ファイル
```

文字コードはすべて UTF-8 (BOM 付き)。Excel で直接開けます。

## データの説明

本データは、Heidelberger Gesamtverzeichnis der griechischen Papyrusurkunden Ägyptens（HGV）の EpiDoc XML 65,478 件から、私が自作した抽出スクリプトで全フィールドを取り出し、その中から穀物関連キーワード（Getreide／Weizen／Gerste／Saatdarlehen）を持つ文書 {n_papyri} 件をスライスしたものである。これは私の修士研究テーマであるローマ期エジプトの穀物価格史研究で扱う一次史料群そのものに対応する。

メイン表 `papyri.csv` は文書識別子（HGV番号・TM番号・DDbDP結合キー）、推定年代、出所地、書写媒体IDを持つ。これに3つの 1:多 テーブル — `papyrus_keywords.csv`（HGV分類語彙）、`papyrus_places.csv`（古代地名階層）、`papyrus_bibliography.csv`（主校訂集成）— が `hgv_id` を外部キーとして結合する。さらに2つのルックアップ表 — `material_lookup.csv`（書写媒体）と `nomos_lookup.csv`（ノモス＝古代行政区分）— があり、`papyri.material_id` と `papyrus_places.nomos_id`（subtype が nome の行のみ）がそれぞれの主キーを参照する。

ルックアップ表を分離した意義は、授業で扱われた「冗長性の排除」にある。例えば「Papyrus」という文字列を {n_papyri} 件のメイン表すべてに書き込む代わりに `material_id=1` という ID 一つで参照することで、媒体名の日本語訳を変更したい場合も `material_lookup.csv` の1行を編集するだけで全体に反映される。ノモスについても同様で、ギリシア語表記と日本語訳の対応を `nomos_lookup.csv` に一元管理している。

## 各表の列定義

### papyri.csv（主キー: hgv_id）

| 列 | 説明 |
|---|---|
| hgv_id | HGV独自ID（PK） |
| tm_id | Trismegistos 番号 |
| ddb_hybrid | DDbDP との結合キー |
| title | 文書のタイトル（多くは独/英） |
| material_id | 書写媒体ID（→ material_lookup） |
| origPlace_text | 出所地（テキスト） |
| date_text | 年代（人間可読、ドイツ語） |
| date_when | 厳密日付 (ISO) |
| date_notBefore | 年代下限 (ISO) |
| date_notAfter | 年代上限 (ISO) |
| date_precision | 年代の精度（low/medium） |

### papyrus_keywords.csv（外部キー: hgv_id）

HGVが付与した分類語彙。多くは独語、文書の内容ジャンル（Vertrag, Quittung 等）。

### papyrus_places.csv（外部キー: hgv_id, nomos_id）

文書に関連する古代地名。`subtype` が `nome` の行はノモス（行政区分）で、`nomos_id` 列で `nomos_lookup.csv` を参照可能。`ref_uri` は Pleiades / Trismegistos の典拠 URI。

### papyrus_bibliography.csv（外部キー: hgv_id）

主校訂（principalEdition）の集成名・巻・番号。1文書が複数の集成に収録される場合もある。

### material_lookup.csv（主キー: material_id）

スライス内に出現する書写媒体のドイツ語（HGV原語）・日本語・英語名対照表。

### nomos_lookup.csv（主キー: nomos_id）

スライス内に出現するノモスのギリシア語名（HGV原語）・日本語・英語名対照表。Trismegistos URI を典拠として併記。日本語訳が未収載の場合は空欄。

## 備考

- 本データは Heidelberger Akademie der Wissenschaften が公開している HGV EpiDoc XML（クリエイティブ・コモンズライセンス）に基づき、私が自作した抽出スクリプト `src/preprocessing/hgv_extract_full.py` および本提出物生成スクリプト `src/preprocessing/build_assignment_subset.py` で機械的に生成した。
- HGV の `langUsage` 要素は翻訳言語の固定セット（fr/en/de/it/es/la/el）であり原文言語ではないため、本提出物には言語表を含めていない。原文言語（多くは古代ギリシア語）を扱うには別途 DDbDP の本文 XML を参照する必要がある。
- スライス順序は `hgv_id` 昇順で決定的。同じ入力に対し同じ出力が得られる。
"""


def write_readme(out_dir: Path, counts: Dict[str, int]):
    text = README_TEMPLATE.format(
        n_papyri  = counts["papyri"],
        n_keywords= counts["keywords"],
        n_places  = counts["places"],
        n_bibl    = counts["bibliography"],
        n_material= counts["material"],
        n_nomos   = counts["nomos"],
    )
    (out_dir / "README.md").write_text(text, encoding="utf-8")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--hgv-full-dir", type=Path, required=True,
                   help="Directory containing the full HGV extraction CSVs")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Submission output directory (will be created)")
    p.add_argument("--limit", type=int, default=200,
                   help="Cap on the number of papyri rows (default 200)")
    args = p.parse_args(argv)

    src = args.hgv_full_dir
    papyri_all   = read_csv(src / "papyri.csv")
    keywords_all = read_csv(src / "keywords.csv")
    places_all   = read_csv(src / "ancient_places.csv")
    bibl_all     = read_csv(src / "bibliography.csv")

    print(f"Input rows -- papyri={len(papyri_all)}, keywords={len(keywords_all)}, "
          f"places={len(places_all)}, bibliography={len(bibl_all)}")

    selected_ids = grain_hgv_ids(keywords_all, args.limit)
    sel_set = set(selected_ids)
    print(f"Grain-keyword hgv_ids selected: {len(selected_ids)} "
          f"(cap={args.limit})")

    papyri_subset = subset_rows(papyri_all, sel_set)
    keywords_subset = subset_rows(keywords_all, sel_set)
    places_subset = subset_rows(places_all, sel_set)
    bibl_subset = subset_rows(bibl_all, sel_set)

    material_rows, material_id_map = build_material_lookup(papyri_subset)
    nomos_rows, nomos_id_map = build_nomos_lookup(places_subset)

    papyri_out = project_papyri(papyri_subset, material_id_map)
    keywords_out = project_keywords(keywords_subset)
    places_out = project_places(places_subset, nomos_id_map)
    bibl_out = project_bibliography(bibl_subset)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "papyri.csv",                PAPYRI_OUT_COLUMNS,       papyri_out)
    write_csv(args.out_dir / "papyrus_keywords.csv",      KEYWORDS_OUT_COLUMNS,     keywords_out)
    write_csv(args.out_dir / "papyrus_places.csv",        PLACES_OUT_COLUMNS,       places_out)
    write_csv(args.out_dir / "papyrus_bibliography.csv",  BIBLIOGRAPHY_OUT_COLUMNS, bibl_out)
    write_csv(args.out_dir / "material_lookup.csv",       MATERIAL_LOOKUP_COLUMNS,  material_rows)
    write_csv(args.out_dir / "nomos_lookup.csv",          NOMOS_LOOKUP_COLUMNS,     nomos_rows)

    counts = {
        "papyri":       len(papyri_out),
        "keywords":     len(keywords_out),
        "places":       len(places_out),
        "bibliography": len(bibl_out),
        "material":     len(material_rows),
        "nomos":        len(nomos_rows),
    }
    write_readme(args.out_dir, counts)

    print("\n[Output]")
    for k, v in counts.items():
        print(f"  {k:<14} {v:>6} rows")
    print(f"\nWrote → {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

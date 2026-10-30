"""HGV date metadata: @when, date alternatives, duplicate records, cache schema (#043)."""
from __future__ import annotations

import json
from pathlib import Path

from lxml import etree

from grain_prices.preprocessing.parse_xml import (
    HGV_INDEX_SCHEMA,
    HgvMeta,
    build_hgv_index,
    load_hgv_index,
    parse_hgv_meta,
)


def hgv_xml(ddb_id: str, origin: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>
<titleStmt><title>Test {ddb_id}</title></titleStmt>
<publicationStmt><idno type="ddb-hybrid">{ddb_id}</idno></publicationStmt>
<sourceDesc><msDesc><history><origin>{origin}</origin></history></msDesc></sourceDesc>
</fileDesc></teiHeader><text><body/></text></TEI>"""


def meta_of(origin: str) -> HgvMeta:
    root = etree.fromstring(hgv_xml("test;1;1", origin).encode())
    return parse_hgv_meta(root)[1]


def write_hgv(root: Path, relative: str, ddb_id: str, origin: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(hgv_xml(ddb_id, origin).encode("utf-8"))


def test_when_only_is_both_bounds():
    meta = meta_of('<origPlace>Arsinoites</origPlace><origDate when="0255-08-17">17. Aug. 255</origDate>')
    assert (meta.date_notbefore, meta.date_notafter) == ("0255-08-17", "0255-08-17")
    assert meta.date_when == "0255-08-17"
    assert meta.date_text == "17. Aug. 255"


def test_not_before_not_after_unchanged():
    meta = meta_of('<origDate notBefore="0101" notAfter="0200">II</origDate>')
    assert (meta.date_notbefore, meta.date_notafter) == ("0101", "0200")


def test_not_after_only_keeps_lower_bound_empty():
    meta = meta_of('<origDate notAfter="0300">vor 300</origDate>')
    assert (meta.date_notbefore, meta.date_notafter) == ("", "0300")


def test_date_alternatives_take_envelope_with_bce_order():
    meta = meta_of(
        '<origDate xml:id="dateAlternativeX" when="-0103-10">Okt. 103 v.Chr.</origDate>'
        '<origDate xml:id="dateAlternativeY" when="-0010-12-17">17. Dez. 10 v.Chr.</origDate>'
    )
    # Lexically "-0010" < "-0103"; chronologically 103 BCE comes first.
    assert (meta.date_notbefore, meta.date_notafter) == ("-0103-10", "-0010-12-17")
    assert meta.date_text == "Okt. 103 v.Chr."


def test_partial_dates_compare_as_spans():
    meta = meta_of(
        '<origDate xml:id="dateAlternativeX" when="0155">155</origDate>'
        '<origDate xml:id="dateAlternativeY" when="0155-02-04">4. Feb. 155</origDate>'
    )
    # "0155" covers the whole year, so it is both the earliest start and the latest end.
    assert (meta.date_notbefore, meta.date_notafter) == ("0155", "0155")


def test_open_alternative_leaves_envelope_open():
    meta = meta_of(
        '<origDate xml:id="dateAlternativeX" notBefore="0150" notAfter="0160">150-160</origDate>'
        '<origDate xml:id="dateAlternativeY" notAfter="0170">vor 170</origDate>'
    )
    assert (meta.date_notbefore, meta.date_notafter) == ("", "0170")


def test_duplicate_records_merge_independent_of_creation_order(tmp_path):
    origins = {
        "HGV2/20.xml": '<origPlace>Oxyrhynchos</origPlace><origDate when="0155-02-08">8. Feb. 155</origDate>',
        "HGV1/10.xml": '<origPlace>Arsinoites</origPlace><origDate when="0155-02-04">4. Feb. 155</origDate>',
        "HGV1/11.xml": '<origPlace>Arsinoites</origPlace><origDate when="0155-02-06">6. Feb. 155</origDate>',
    }
    indexes = []
    for name, order in (("a", list(origins.items())), ("b", list(reversed(origins.items())))):
        root = tmp_path / name
        for relative, origin in order:
            write_hgv(root, relative, "p.berl.frisk;;1", origin)
        indexes.append(build_hgv_index(root))
    assert indexes[0] == indexes[1]
    meta = indexes[0]["p.berl.frisk;;1"]
    assert (meta.date_notbefore, meta.date_notafter) == ("0155-02-04", "0155-02-08")
    assert meta.place == "Arsinoites / Oxyrhynchos"
    assert meta.date_when == "0155-02-04"  # first record in sorted path order


def test_unversioned_cache_is_rebuilt(tmp_path):
    hgv = tmp_path / "hgv"
    write_hgv(hgv, "HGV1/1.xml", "bgu;1;14", '<origDate when="0255-08-17">17. Aug. 255</origDate>')
    cache = tmp_path / "hgv_index_cache.json"
    stale = {"bgu;1;14": vars(HgvMeta(date_when="0255-08-17"))}
    cache.write_bytes(json.dumps(stale).encode("utf-8"))

    idx = load_hgv_index(hgv, cache_path=cache)
    assert idx["bgu;1;14"].date_notbefore == "0255-08-17"
    payload = json.loads(cache.read_text(encoding="utf-8"))
    assert payload["schema"] == HGV_INDEX_SCHEMA
    assert b"\r\n" not in cache.read_bytes()
    assert load_hgv_index(tmp_path / "missing", cache_path=cache) == idx


def test_spine_and_candidates_carry_when_dates(tmp_path):
    from grain_prices.canonical.candidates import generate_candidates
    from grain_prices.canonical.spine import build_spine

    ddb = tmp_path / "ddb"
    ddb.mkdir()
    (ddb / "sale.xml").write_bytes(Path("tests/fixtures/canonical_sale_multi.xml").read_bytes())
    hgv = tmp_path / "hgv"
    write_hgv(hgv, "HGV1/1.xml", "fixture;sale;1",
              '<origPlace>Arsinoites</origPlace><origDate when="0255-08-17">17. Aug. 255</origDate>')
    cache = tmp_path / "hgv_cache.json"

    build_spine(ddb, tmp_path / "spine", hgv_dir=hgv, hgv_cache=cache)
    generate_candidates(ddb, tmp_path / "candidates.jsonl", hgv_dir=hgv, hgv_cache=cache)
    for path in (tmp_path / "spine/candidate_mentions.jsonl", tmp_path / "candidates.jsonl"):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert rows
        for row in rows:
            assert (row["date_not_before"], row["date_not_after"]) == ("0255-08-17", "0255-08-17")
            assert row["place"] == "Arsinoites"

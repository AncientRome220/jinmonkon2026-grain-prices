"""Corpus-free regressions for #032, #033, #034 and #036."""
from copy import deepcopy
import json

import pytest

from grain_prices.canonical.options import scan_options, currency_name
from grain_prices.canonical.text import normalize_with_map
from grain_prices.legacy.stage2 import tokenize_with_positions
from grain_prices.canonical.observations import deduplicate_observations, locate_evidence
from grain_prices.canonical.provenance import candidate_fingerprint, validate_candidate_binding
from grain_prices.canonical.candidates import generate_candidates


def scan(text):
    norm = normalize_with_map(text).text
    return scan_options(norm, tokenize_with_positions(norm))


@pytest.mark.parametrize("word,name", [("νομίσματα", "nomisma"), ("κερατίων", "keration"),
                                      ("σολίδους", "solidus"), ("χρυσίνων", "chrysinon")])
def test_late_currencies(word, name):
    q, m = scan(f"πυροῦ ἀρτάβαι 2 {word} 3")
    assert q[0]["value"] == "2"
    assert m[0]["value"] == "3"
    assert currency_name(m[0]["currency"]) == name


def test_myriads_do_not_steal_artaba_number():
    q, m = scan("σίτου τῆς ἀρτάβης 1 δηναρίων μυριάδες 450")
    assert q[0]["value"] == "1"
    assert m[0]["value"] == "4500000"
    assert m[0]["written_value"] == "450" and m[0]["multiplier"] == 10000
    assert q[0]["value_char_end"] < m[0]["value_char_start"]


def test_loose_quantity_does_not_steal_money_digits():
    q, m = scan("Τεϊηοῦ κριθῆς μέτρα γ δραχμαὶ 12")
    assert q[0]["value"] is None  # bare Greek gamma is deliberately not parsed
    assert m[0]["value"] == "12"


def test_rejected_preceding_match_leaves_following_loose_match_available():
    q, m = scan("ἀρτάβας 8 δραχμὰς τέσσαρας 4")
    assert q[0]["value"] == "8"
    assert m[0]["value"] == "4"


def test_number_before_currency_keeps_expression_start_anchor():
    q, m = scan("πυροῦ λόγος 3 τριώβολον")
    assert m[0]["value"] == "3"
    assert m[0]["token_index"] == 2


def test_tied_ownership_is_not_decided_by_order():
    q, m = scan("ἀρτάβαι λόγος 2 λόγος δραχμαί")
    assert q[0]["value"] is None and m[0]["value"] is None
    assert q[0]["ownership_ambiguous"] and m[0]["ownership_ambiguous"]


def test_ellipses_and_fraction_sum():
    q, m = scan("δραχμαὶ 2811 αἳ πυρῶν 1405 1/2 1/12")
    assert float(q[0]["value"]) == pytest.approx(1405 + 7/12)
    assert q[0]["unit"] is None and q[0]["inference_rule"] == "grain_number"
    q, m = scan("τιμῆς πυροῦ ἀρταβῶν 5 ἀνὰ 1000 5000 4000")
    assert q[0]["value"] == "5"
    assert len(m) == 1 and m[0]["value"] == "1000" and m[0]["currency"] is None
    _, m = scan("πυροῦ τιμῆς 17")
    assert m[0]["inference_rule"] == "time_number"


def test_year_and_unrelated_numbers_are_not_elliptical_amounts():
    q, m = scan("τιμῆς πυροῦ 17 ἔτους")
    assert not q and not m


@pytest.mark.parametrize("text", [
    "ἀρταβῶν 1 ἔτους Γάλβα δραχμὰς 81",
    "ἀρταβῶν διοικήσεως 25 ἔτους κάτω τοπαρχίας",
    "ἐκ φόρου 1 ἔτους πυροῦ ἀρτάβαι",
    "10 ἔτους Φιλαδελφείας τάλαντα δύο γίνεται τάλαντα 2",
    "10 ἔτους Φιλαδελφίας τάλαντον ἕν γίνεται τάλαντον 1",
    "4 ἔτους ὡρίσθη δραχμῶν",
    "ἔτους 10 πυροῦ ἀρτάβαι",
])
def test_date_numbers_never_enter_explicit_or_loose_amounts(text):
    norm = normalize_with_map(text).text
    from grain_prices.canonical.options import NUM_RE, date_owned
    reserved = {(m.start(), m.end()) for m in NUM_RE.finditer(norm) if date_owned(norm, m)}
    q, m = scan(text)
    assert reserved
    assert not reserved & {(o['value_char_start'], o['value_char_end']) for o in q+m}


def test_rejected_following_year_leaves_valid_preceding_amount():
    _, m = scan("3 τριώβολον , 1 ἔτους Γάλβα")
    assert m[0]["value"] == "3"
    _, m = scan("δραχμαὶ 8, ἔτους 10")
    assert m[0]["value"] == "8"
    _, m = scan("ἔτους 10 δραχμαὶ τέσσαρες 4")
    assert m[0]["value"] == "4"
    q, m = scan("πυροῦ λόγος 17")
    assert not q and not m


def make_xml(path, block):
    path.write_text('<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>'
                    '<publicationStmt><idno type="ddb-hybrid">fixture;new;1</idno>'
                    '</publicationStmt></fileDesc></teiHeader><text><body><div type="edition">'
                    + block + '</div></body></text></TEI>', encoding="utf-8")


def test_enumeration_keeps_blocks_and_surface_value_spans(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    make_xml(corpus / "x.xml", "<ab>πυροῦ ἀρτάβαι 2 νομίσματα 3</ab>" * 2)
    target = tmp_path / "new.jsonl"
    generate_candidates(corpus, target)
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["source_block_id"] != rows[1]["source_block_id"]
    for row in rows:
        assert row["candidate_fingerprint"] == candidate_fingerprint(row)
        for option in row["qty_options"] + row["money_options"]:
            assert row["source_block_text"][option["value_char_start"]:option["value_char_end"]] == option["written_value"]
    original = target.read_bytes()
    generate_candidates(corpus, target)
    assert target.read_bytes() == original and b"\r\n" not in original


def observation(**kwargs):
    return {"window_id": "w1", "method": "model1", "repeat_index": 0,
            "doc_id": "doc", "source_block_id": "e0:b0", "source_block_sha256": "abc",
            "coordinate_system": "expanded-block-surface-v1", "quantity_value_span": [1, 2],
            "price_value_span": [3, 4], "context_tags": ["sale"], "rate_basis": "stated_rate",
            "analysis_eligible": True, "series_strict": True, **kwargs}


def test_dedup_preserves_sources_and_separates_blocks_models_repeats():
    a = observation()
    b = observation(window_id="w2")
    groups = deduplicate_observations([a, b])
    assert len(groups) == 1 and len(groups[0]["source_observations"]) == 2
    assert groups[0]["contributing_window_ids"] == ["w1", "w2"]
    for field, value in (("source_block_id", "e0:b1"), ("method", "human"), ("repeat_index", 1)):
        assert len(deduplicate_observations([a, observation(**{field: value})])) == 2


def test_conflicting_interpretations_are_excluded_not_unioned():
    groups = deduplicate_observations([observation(), observation(context_tags=["tax_commutation"])])
    assert groups[0]["grouping_status"] == "interpretation_conflict"
    assert not groups[0]["analysis_eligible"] and not groups[0]["series_strict"]
    assert groups[0]["context_tags"] is None


def test_analysis_writes_mixed_agreed_and_conflicting_groups(tmp_path):
    from grain_prices.canonical.analysis import analyze
    from grain_prices.canonical.io import write_jsonl
    rows = [observation(), observation(context_tags=["tax_commutation"]), observation(source_block_id="b1")]
    for r in rows:
        r.update(observation_schema_version="canonical-observation-v2", linkage_strength="explicit", commodity_class="raw_grain")
    source = tmp_path / "observations.jsonl"
    write_jsonl(source, rows)
    result = analyze(source, tmp_path / "analysis")
    assert result["interpretation_conflicts"] == 1
    assert result["series_counts"]["strict"] == 1


def test_null_options_require_unambiguous_evidence():
    assert locate_evidence({"source_block_text": "τιμῆς 2"}, "τιμης 2") == [0, 7]
    assert locate_evidence({"source_block_text": "τιμῆς 2 τιμῆς 2"}, "τιμης 2") is None
    a = observation(price_value_span=None)
    assert len(deduplicate_observations([a, a])) == 2
    a["evidence_char_span"] = [0, 7]
    assert len(deduplicate_observations([a, a])) == 1


def test_fingerprint_rejects_changed_options_and_missing_binding():
    c = {"window_id": "w1", "qty_options": [{"value": "1"}]}
    c["candidate_fingerprint"] = candidate_fingerprint(c)
    p = {"candidate_fingerprint": c["candidate_fingerprint"]}
    validate_candidate_binding(p, c)
    with pytest.raises(ValueError):
        validate_candidate_binding({}, c)
    changed = deepcopy(c)
    changed["qty_options"][0]["value"] = "2"
    with pytest.raises(ValueError):
        validate_candidate_binding(p, changed)


def test_fingerprinted_resume_and_evaluation(tmp_path, monkeypatch):
    from grain_prices.llm import canonical_runner as runner
    from grain_prices.canonical.io import write_jsonl
    from grain_prices.canonical.evaluation import evaluate_llm_window
    from test_canonical_pipeline import sample_candidate, sample_answer, PROMPT_PATH
    c = sample_candidate()
    c["candidate_fingerprint"] = candidate_fingerprint(c)
    source, target = tmp_path / "candidates.jsonl", tmp_path / "predictions.jsonl"
    write_jsonl(source, [c])
    calls = []
    class Client:
        def call(self, *args):
            calls.append(1)
            return sample_answer()
    monkeypatch.setattr(runner, "build_client", lambda *args: Client())
    runner.score_candidates(source, target, PROMPT_PATH, "openai", "mock")
    runner.score_candidates(source, target, PROMPT_PATH, "openai", "mock", resume=True)
    assert len(calls) == 1
    c["qty_options"][0]["value"] = "7"
    c["candidate_fingerprint"] = candidate_fingerprint(c)
    write_jsonl(source, [c])
    before = target.read_bytes()
    with pytest.raises(ValueError, match="fingerprint"):
        runner.score_candidates(source, target, PROMPT_PATH, "openai", "mock", resume=True)
    assert target.read_bytes() == before and len(calls) == 1
    gold = tmp_path / "gold.csv"
    gold.write_text("window_id,canonical_label\nw1,PRICE\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        evaluate_llm_window(target, gold, tmp_path / "evaluation.json", candidates_path=source)


def test_materialization_keeps_window_rows_and_analysis_groups(tmp_path):
    from grain_prices.canonical.io import write_jsonl, read_jsonl
    from grain_prices.canonical.materialize import materialize
    from grain_prices.canonical.analysis import analyze
    from test_canonical_pipeline import sample_answer
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    make_xml(corpus / "x.xml", "<ab>τιμῆς πυροῦ πυροῦ ἀρτάβαι 2 δραχμαὶ 8</ab>")
    source = tmp_path / "candidates.jsonl"
    generate_candidates(corpus, source)
    candidates = list(read_jsonl(source))
    rows = []
    for c in candidates:
        a = sample_answer()
        a["observations"][0]["evidence_span"] = c["evidence_text"]
        rows.append({"window_id": c["window_id"], "answer": a, "provider": "mock", "model_id": "mock",
                     "candidate_fingerprint": c["candidate_fingerprint"]})
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, rows)
    output = tmp_path / "observations.jsonl"
    result = materialize(source, predictions, output)
    assert result["final_observations"] == 2 and result["source_groups"] == 1
    assert len(list(read_jsonl(output))) == 2
    analysis = analyze(output, tmp_path / "analysis")
    assert analysis["emitted_window_observations"] == 2 and analysis["final_observations"] == 1


def test_unconfirmed_gold_cannot_be_frozen(tmp_path):
    from grain_prices.canonical.rebuild import freeze, write_csv, write_json
    from grain_prices.canonical.io import sha256_file
    source = tmp_path / "candidates.jsonl"
    source.write_text("", encoding="utf-8")
    gold = tmp_path / "gold.csv"
    row = {"sample_id": "c1", "window_id": "w1", "review_status": "", "reviewed_by": ""}
    write_csv(gold, [row])
    audit = tmp_path / "audit.json"
    write_json(audit, {"inputs": {"new_candidates": {"sha256": sha256_file(source)},
                                 "original_gold": {"path": str(gold)}}})
    with pytest.raises(ValueError, match="human review incomplete"):
        freeze(gold, source, audit, tmp_path / "frozen")

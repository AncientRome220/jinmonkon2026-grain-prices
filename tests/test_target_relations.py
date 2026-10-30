"""Source linkage is distinct from reading, ranking, and window geometry."""
from copy import deepcopy
import hashlib
import json

import pytest

from grain_prices.canonical.io import write_jsonl, read_jsonl
from grain_prices.canonical.provenance import candidate_fingerprint
from grain_prices.canonical.relations import (
    COORDINATES, GOLD_VERSION, OBJECTIVE_VERSION, PREDICTION_VERSION,
    target_id, validate_gold, evaluate_relations,
)
from grain_prices.canonical.relation_windows import (
    freeze_relations, search_windows, window_candidate, prepare_relations,
)
from grain_prices.canonical.rebuild import write_csv, write_json
from grain_prices.llm.canonical_runner import score_candidates


def target(block="e0:b0"):
    text = "πυροῦ ἀρτάβας 2 δραχμὰς 16 κριθῆς ἀρτάβας 2 δραχμὰς 16"
    span = lambda s: [text.index(s), text.index(s)+len(s)]
    sha = hashlib.sha256(text.encode()).hexdigest()
    grain = span("πυροῦ")
    relation = {"grain_span": grain, "commodity": "wheat", "rate_basis": "derived_from_total",
                "quantity": {"status": "present", "components": [
                    {"span": span("ἀρτάβας 2"), "value": "2", "denomination": "artaba"}]},
                "money": {"status": "present", "components": [
                    {"span": span("δραχμὰς 16"), "value": "16", "denomination": "drachma"}]},
                "evidence_span": [0, text.index(" κριθῆς")]}
    return {"schema_version": GOLD_VERSION, "objective_version": OBJECTIVE_VERSION,
            "target_id": target_id("fixture", block, sha, grain), "doc_id": "fixture",
            "source_block_id": block, "source_block_sha256": sha, "source_block_text": text,
            "coordinate_system": COORDINATES, "grain_span": grain, "grain_form": "πυροῦ",
            "qty_options": [], "money_options": [], "origin_sample_ids": ["c1"],
            "review_status": "confirmed", "reviewed_by": "fixture", "coverage": "exhaustive",
            "relations": [relation], "relation_proposals": [{"origin_sample_id": "c1",
                "anchor_resolution": "unique_commodity_compatible_grain_in_quote",
                "relation": deepcopy(relation)}]}


def evaluate(tmp_path, t, answers, *, omit_candidate=False, **prediction_fields):
    c = window_candidate(t, 80)
    p = {"schema_version": PREDICTION_VERSION, "target_id": t["target_id"],
         "candidate_fingerprint": c["candidate_fingerprint"], "model_id": "fixture",
         "provider": "fixture", "repeat_index": 0, "answer": {"relations": answers},
         **prediction_fields}
    gpath, cpath, ppath = [tmp_path / n for n in ("gold.jsonl", "input.jsonl", "pred.jsonl")]
    write_jsonl(gpath, [t])
    write_jsonl(cpath, [] if omit_candidate else [c])
    write_jsonl(ppath, [] if omit_candidate else [p])
    return evaluate_relations(ppath, gpath, cpath, tmp_path / "result.json")


def test_right_values_wrong_source_is_wrong_relationship(tmp_path):
    t = target()
    wrong = deepcopy(t["relations"][0])
    text = t["source_block_text"]
    for side, phrase in (("quantity", "ἀρτάβας 2"), ("money", "δραχμὰς 16")):
        start = text.rindex(phrase)
        wrong[side]["components"][0]["span"] = [start, start+len(phrase)]
    wrong["evidence_span"] = [0, len(text)]
    result = evaluate(tmp_path, t, [wrong])
    assert result["relationship"]["tp"] == 0
    assert result["relationship"]["fp"] == result["relationship"]["fn"] == 1


def test_reading_and_basis_are_separate(tmp_path):
    t = target()
    wrong = deepcopy(t["relations"][0])
    wrong["money"]["components"][0]["value"] = "1"
    result = evaluate(tmp_path, t, [wrong])
    assert result["relationship"]["f1"] == 1
    assert result["correct_readings_on_matched_relationships"] == 0
    assert result["correct_rate_basis_on_matched_relationships"] == 1
    assert result["complete_relation"]["f1"] == 0
    wrong = deepcopy(t["relations"][0])
    wrong["rate_basis"] = "unknown"
    result = evaluate(tmp_path, t, [wrong])
    assert result["correct_readings_on_matched_relationships"] == 1
    assert result["correct_rate_basis_on_matched_relationships"] == 0


def test_multiple_relations_order_and_duplicate_penalty(tmp_path):
    t = target()
    second = deepcopy(t["relations"][0])
    second["quantity"] = {"status": "unstated", "components": []}
    t["relations"].append(second)
    result = evaluate(tmp_path, t, list(reversed(t["relations"])))
    assert result["relationship"]["f1"] == 1
    result = evaluate(tmp_path, t, t["relations"] + [second])
    assert result["relationship"]["tp"] == 2
    assert result["relationship"]["fp"] == 1
    assert result["exact_target_relation_sets"] == 0


@pytest.mark.parametrize("fields", [{"error": "timeout"}, {"validation_problems": ["bad"]}])
def test_failures_stay_in_denominator(tmp_path, fields):
    t = target()
    result = evaluate(tmp_path, t, t["relations"], **fields)
    assert result["relationship"]["fn"] == 1
    result = evaluate(tmp_path, t, [], omit_candidate=True)
    assert result["relationship"]["fn"] == 1
    assert result["targets_without_candidate"] == [t["target_id"]]


def test_pending_gold_and_fingerprint_rejected(tmp_path):
    t = target()
    with pytest.raises(ValueError, match="fingerprint"):
        evaluate(tmp_path, t, t["relations"], candidate_fingerprint="stale")
    t["review_status"] = ""
    with pytest.raises(ValueError, match="human confirmation"):
        validate_gold([t])
    t["review_status"] = "confirmed"
    t["coverage"] = "primary_only"
    with pytest.raises(ValueError, match="exhaustive"):
        validate_gold([t])
    t["coverage"] = "exhaustive"
    t["relations"][0]["money"]["status"] = "unresolved"
    with pytest.raises(ValueError, match="status"):
        validate_gold([t])


def test_block_identity_and_full_expression_window(tmp_path):
    t = target()
    assert t["target_id"] != target("e0:b1")["target_id"]
    review = tmp_path / "review.jsonl"
    write_jsonl(review, [t])
    result = search_windows(review, tmp_path / "windows", [3, 4])
    # Radius 3 includes the currency word, but cuts off its numeral.
    assert result["summaries"][0]["eligible_both_sides_retained"] == 0
    assert result["summaries"][1]["eligible_both_sides_retained"] == 1
    assert not result['summaries'][0]['confirmed_gold_relation_retention'][0]['components_retained']
    assert result['summaries'][1]['confirmed_gold_relation_retention'][0]['components_retained']
    candidate = list(read_jsonl(tmp_path / "windows/relation_candidates_r4.jsonl"))[0]
    assert not set(candidate) & {"relations", "relation_proposals", "origin_annotations"}
    assert candidate["candidate_fingerprint"] == candidate_fingerprint(candidate)
    assert b"\r\n" not in (tmp_path / "windows/manifest_window_search.json").read_bytes()
    with pytest.raises(ValueError, match="1 to 80"):
        search_windows(review, tmp_path / "bad", [81])


def test_prepare_preserves_sample_and_freeze_rejects_dropped_origins(tmp_path):
    t = target()
    c = {"ddb_id": t["doc_id"], "window_id": "w1", "source_block_id": t["source_block_id"],
         "source_block_sha256": t["source_block_sha256"], "source_block_text": t["source_block_text"],
         "grain_form": "πυροῦ", "grain_surface_char_start_debug": 0,
         "grain_surface_char_end_debug": 5, "qty_options": [], "money_options": []}
    c["candidate_fingerprint"] = candidate_fingerprint(c)
    row = {"sample_id": "c1", "window_id": "w1", "candidate_fingerprint": c["candidate_fingerprint"],
           "canonical_label": "NO_PRICE", "quantity_option": "", "money_option": ""}
    csv, evidence, candidates = [tmp_path / n for n in ("old.csv", "evidence.jsonl", "candidates.jsonl")]
    write_csv(csv, [row])
    write_jsonl(evidence, [{"sample_id": "c1", "new_candidate": c, "old_candidate": c, "original_annotation": row}])
    write_jsonl(candidates, [c])
    before = csv.read_bytes()
    out = tmp_path / "new"
    manifest = prepare_relations(csv, evidence, candidates, out)
    assert manifest["historical_sample_rows"] == manifest["target_review_rows"] == 1
    assert csv.read_bytes() == before
    path = out / "relation_gold_review.jsonl"
    pending = list(read_jsonl(path))[0]
    assert pending["review_status"] == ""
    with pytest.raises(ValueError, match="human confirmation"):
        freeze_relations(path, out / "frozen.jsonl")
    pending.update(review_status="confirmed", reviewed_by="tester", coverage="exhaustive")
    pending["origin_sample_ids"] = []
    write_jsonl(path, [pending])
    with pytest.raises(ValueError, match="all historical"):
        freeze_relations(path, out / "frozen.jsonl")
    pending["origin_sample_ids"] = ["c1"]
    write_jsonl(path, [pending])
    freeze_relations(path, out / "frozen.jsonl")
    with pytest.raises(ValueError, match="refusing"):
        freeze_relations(path, out / "frozen.jsonl")


def test_historical_runner_rejects_new_objective_before_provider(tmp_path):
    path = tmp_path / "candidates.jsonl"
    write_jsonl(path, [window_candidate(target(), 20)])
    with pytest.raises(ValueError, match="incompatible"):
        score_candidates(path, tmp_path / "out.jsonl", tmp_path / "no-prompt.md", "openai", "fixture")

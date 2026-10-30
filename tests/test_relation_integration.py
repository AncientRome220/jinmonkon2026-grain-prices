from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_target_relations import target
from grain_prices.annotation.relation_review import save_review, reanchor, grain_choices
from grain_prices.canonical.io import write_jsonl, read_jsonl, sha256_file
from grain_prices.canonical.relation_windows import window_candidate
from grain_prices.canonical.relation_materialize import materialize_relations, group_relations
from grain_prices.canonical.provenance import candidate_fingerprint
from grain_prices.llm import relation_runner as runner


def setup_run(tmp_path):
    t = target()
    c = window_candidate(t, 80)
    inputs, output = tmp_path/'inputs.jsonl', tmp_path/'predictions.jsonl'
    write_jsonl(inputs, [c])
    prompt = Path('src/grain_prices/prompts/target_grain_relations_v1.md')
    answer = {"relations": t['relations'], "contexts": [{"relation_index": 0,
        "context_tags": ["unknown"], "linkage_strength": "explicit"}]}
    return t, c, inputs, output, prompt, answer


def test_prompt_whitelist_hides_block_gold_and_clipped_options(tmp_path):
    t, c, *_ = setup_run(tmp_path)
    t['source_block_text'] += ' SECRET_AFTER_CROP'
    # Local validation protects source identity; test the whitelist with valid
    # short input plus arbitrary gold/review keys.
    c = window_candidate(target(), 4)
    c.update(origin_annotations='HUMAN_SECRET', review_note='NOTE_SECRET', relations=['GOLD_SECRET'])
    c['candidate_fingerprint'] = candidate_fingerprint(c)
    rendered = runner.visible_input(c)
    assert 'HUMAN_SECRET' not in rendered and 'NOTE_SECRET' not in rendered and 'GOLD_SECRET' not in rendered
    assert 'κριθῆς' not in rendered
    assert 'source_block_text' not in rendered


def test_missing_prediction_file_keeps_targets_and_crop_diagnostics(tmp_path):
    from grain_prices.canonical.relations import evaluate_relations
    t, c, inputs, output, prompt, answer = setup_run(tmp_path)
    gold = tmp_path/'gold.jsonl'; write_jsonl(gold, [t])
    write_jsonl(inputs, [window_candidate(t, 3)])
    result = evaluate_relations(output, gold, inputs, tmp_path/'metrics.json')
    assert result['gold_targets'] == 1 and result['relationship']['fn'] == 1
    assert result['missing_prediction_targets'] == [t['target_id']]
    assert result['target_diagnostics'][0]['gold_evidence_outside_excerpt_relation_indexes'] == [0]


def test_full_pass_keeps_empty_options_and_timing_is_phase_specific(tmp_path):
    from grain_prices.canonical.relation_experiment import prepare_full_pass, summarize_checking
    t = target()
    c = {'ddb_id': t['doc_id'], 'window_id': 'fixture-window',
         'source_block_id': t['source_block_id'], 'source_block_sha256': t['source_block_sha256'],
         'source_block_text': t['source_block_text'], 'grain_form': t['grain_form'],
         'grain_surface_char_start_debug': t['grain_span'][0],
         'grain_surface_char_end_debug': t['grain_span'][1], 'qty_options': [], 'money_options': []}
    c['candidate_fingerprint'] = candidate_fingerprint(c)
    source = tmp_path/'population.jsonl'; write_jsonl(source, [c])
    prompt = Path('src/grain_prices/prompts/target_grain_relations_v1.md')
    spec = prepare_full_pass(source, tmp_path/'full', 25, prompt)
    assert spec['initial_call_limit'] == 1 and spec['maximum_call_limit'] == 3
    assert list(read_jsonl(Path(spec['input'])))[0]['money_options'] == []
    review = tmp_path/'review.jsonl'
    t.update(review_phase='post_llm_check', prediction_before_review={'answer': {'relations': []}},
             review_sessions=[{'phase': 'target_gold_review', 'active_seconds': 100},
                              {'phase': 'post_llm_check', 'active_seconds': 12}])
    write_jsonl(review, [t]); result = summarize_checking(review, tmp_path/'time.json')
    assert result['recorded_active_seconds'] == 12
    assert result['changed_relation_targets_among_confirmed'] == 1
    assert result['time_savings'] is None


def test_runner_retries_resume_and_materialization(tmp_path, monkeypatch):
    t, c, inputs, output, prompt, answer = setup_run(tmp_path)
    calls = []
    class Client:
        def call(self, system, user, schema):
            calls.append(1)
            assert 'Output JSON schema:' in system
            return {} if len(calls) < 3 else answer
    monkeypatch.setattr(runner, 'build_client', lambda *args: Client())
    result = runner.score_relations(inputs, output, prompt, 'openai', 'fixture')
    assert result['provider_calls'] == 3 and result['valid'] == 1
    result = runner.score_relations(inputs, output, prompt, 'openai', 'fixture', resume=True)
    assert result['resumed_valid'] == 1 and result['provider_calls'] == 0
    with pytest.raises(ValueError, match='incompatible'):
        runner.score_relations(inputs, output, prompt, 'openai', 'different', resume=True)
    assert b'\r\n' not in output.read_bytes()
    result = materialize_relations(inputs, output, tmp_path/'observations.jsonl')
    assert result['emitted_relations'] == result['deduplicated_source_groups'] == 1
    assert result['observations_eligible_for_historical_analysis'] == 0


def test_invalid_outputs_stop_after_two_retries_and_dry_run_no_provider(tmp_path, monkeypatch):
    t, c, inputs, output, prompt, answer = setup_run(tmp_path)
    class Client:
        def call(self, *args): return {'relations': [], 'contexts': [{'bad': 1}]}
    monkeypatch.setattr(runner, 'build_client', lambda *args: Client())
    r = runner.score_relations(inputs, output, prompt, 'anthropic', 'fixture')
    assert r['invalid'] == 1 and r['provider_calls'] == 3
    resumed = runner.score_relations(inputs, output, prompt, 'anthropic', 'fixture', resume=True)
    assert resumed['provider_calls'] == 0 and resumed['resumed_exhausted'] == 1
    monkeypatch.setattr(runner, 'build_client', lambda *args: pytest.fail('dry run called provider'))
    runner.score_relations(inputs, output, prompt, 'anthropic', 'fixture', dry_run=True)
    with pytest.raises(ValueError, match='at most two'):
        runner.score_relations(inputs, output, prompt, 'anthropic', 'fixture', max_validation_retries=3)


def test_compound_amounts_preserved_and_conflicts_separated(tmp_path, monkeypatch):
    t, c, inputs, output, prompt, answer = setup_run(tmp_path)
    second = deepcopy(answer['relations'][0]['money']['components'][0])
    second['span'] = [c['source_block_text'].rindex('δραχμὰς'), len(c['source_block_text'])]
    second['denomination'] = 'talent'
    answer['relations'][0]['money']['components'].append(second)
    answer['relations'][0]['evidence_span'] = [0, len(c['source_block_text'])]
    class Client:
        def call(self, *args): return answer
    monkeypatch.setattr(runner, 'build_client', lambda *args: Client())
    runner.score_relations(inputs, output, prompt, 'openai', 'fixture')
    materialize_relations(inputs, output, tmp_path/'obs.jsonl')
    row = list(read_jsonl(tmp_path/'obs.jsonl'))[0]
    assert len(row['relation']['money']['components']) == 2
    assert row['price_per_artaba_drachma'] is None
    other = deepcopy(row)
    other['relation']['money']['components'][0]['value'] = '999'
    assert group_relations([row, other])[0]['grouping_status'] == 'interpretation_conflict'
    other['model_id'] = 'different'
    assert len(group_relations([row, other])) == 2


def test_atomic_review_backup_stale_rejection_and_original_time(tmp_path):
    path = tmp_path/'review.jsonl'; row = target()
    row['origin_annotations'] = [{'Annotation_Seconds': '123'}]
    write_jsonl(path, [row]); digest = sha256_file(path); before = path.read_bytes()
    tampered = deepcopy(row); tampered['origin_annotations'][0]['Annotation_Seconds'] = '0'
    with pytest.raises(ValueError, match='original annotations'):
        save_review(path, digest, row['target_id'], tampered)
    edit = deepcopy(row); edit['review_note'] = 'source checked'
    save_review(path, digest, row['target_id'], edit, 12)
    saved = list(read_jsonl(path))[0]
    assert saved['origin_annotations'][0]['Annotation_Seconds'] == '123'
    assert saved['review_sessions'][0]['active_seconds'] == 12
    assert (tmp_path/'backups'/f'{digest}.jsonl').read_bytes() == before
    with pytest.raises(ValueError, match='changed on disk'):
        save_review(path, digest, row['target_id'], edit)
    assert b'\r\n' not in path.read_bytes()


def test_reanchor_and_merge_preserve_all_origins_without_approval(tmp_path):
    original = target(); second = reanchor(original, grain_choices(original)[1])
    second['origin_sample_ids'] = ['c2']
    path = tmp_path/'review.jsonl'; write_jsonl(path, [original, second])
    moved = reanchor(original, second['grain_span'])
    result = save_review(path, sha256_file(path), original['target_id'], moved)
    saved = list(read_jsonl(path))
    assert result['merged'] and len(saved) == 1
    assert saved[0]['origin_sample_ids'] == ['c1','c2']
    assert saved[0]['review_status'] == '' and saved[0]['coverage'] == 'unreviewed'


def test_confirmation_cannot_accept_unresolved_side(tmp_path):
    row = target(); path = tmp_path/'review.jsonl'; write_jsonl(path, [row])
    edit = deepcopy(row); edit['relations'][0]['money']['status'] = 'unresolved'
    with pytest.raises(ValueError, match='status'):
        save_review(path, sha256_file(path), row['target_id'], edit)


def test_cli_relation_dry_run(tmp_path):
    from grain_prices.cli import main
    t, c, inputs, output, prompt, answer = setup_run(tmp_path)
    assert main(['score-relations', '--runs-root', str(tmp_path/'runs'), '--run-id','fixture',
                 '--input',str(inputs),'--provider','openai','--model','fixture','--dry-run']) == 0


def test_checker_renders_both_radii_without_mutating_review(tmp_path, monkeypatch):
    import sys
    from streamlit.testing.v1 import AppTest
    path=tmp_path/'review.jsonl'; row=target(); row['review_status']=''
    write_jsonl(path,[row]); before=path.read_bytes()
    monkeypatch.setattr(sys,'argv',['relation_app.py','--review',str(path)])
    app=AppTest.from_file(Path('src/grain_prices/annotation/relation_app.py').resolve()).run(timeout=15)
    assert not app.exception
    assert path.read_bytes()==before
    assert len(app.dataframe)==2


def test_experiment_budget_and_unconfirmed_gate(tmp_path):
    from grain_prices.canonical.relation_experiment import prepare_experiment, analyze_experiment
    rows=[]
    for i in range(30):
        r=target('e0:b'+str(i)); r['origin_sample_ids']=['c'+str(i)]; r['review_status']=''; rows.append(r)
    path=tmp_path/'gold.jsonl'; write_jsonl(path,rows)
    prompt=Path('src/grain_prices/prompts/target_grain_relations_v1.md')
    with pytest.raises(ValueError,match='human confirmation'):
        prepare_experiment(path,tmp_path/'forbidden',prompt)
    result=prepare_experiment(path,tmp_path/'preview',prompt,preview=True)
    assert result['initial_call_limit']==6*30+540
    assert result['maximum_call_limit']==3*(6*30+540)
    assert len(result['stability_targets'])==30
    for r in rows: r['review_status']='confirmed'
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match='frozen gold manifest'):
        prepare_experiment(path, tmp_path/'not_frozen', prompt)
    from grain_prices.canonical.rebuild import write_json
    write_json(path.with_suffix('.manifest.json'), {'status': 'reviewed_awaiting_run_authorization',
        'frozen_gold_sha256': sha256_file(path), 'confirmed_targets': 30, 'historical_sample_rows': 30})
    prepare_experiment(path, tmp_path/'frozen_package', prompt)
    analysis = analyze_experiment(tmp_path/'frozen_package/experiment_package.json', tmp_path/'analysis')
    assert all(r['metrics']['gold_targets']==30 and len(r['metrics']['missing_prediction_targets'])==30 for r in analysis['results'])
    assert all(s['three_valid_identical_answers']==0 for s in analysis['stability'])

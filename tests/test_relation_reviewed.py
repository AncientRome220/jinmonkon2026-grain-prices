from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_target_relations import target
from grain_prices.canonical.io import write_jsonl, read_jsonl, sha256_file
from grain_prices.canonical.relation_reviewed import (
    select_reviewed, prepare_reviewed, load_reviewed, run_reviewed, analyze_reviewed,
)
from grain_prices.canonical.relation_provisional import BudgetStop
from grain_prices.llm import relation_runner, clients

PROMPT = Path('src/grain_prices/prompts/target_grain_relations_v2.md')


def population():
    rows = [target(f'e0:b{i}') for i in range(8)]
    for i, row in enumerate(rows):
        row['origin_sample_ids'] = [f's{i}']
        if i >= 3:
            row['relations'] = []
    return rows


def test_selection_preserves_labels_and_accounts_for_every_parent():
    rows = population()
    original = deepcopy(rows)
    selected, ledger = select_reviewed(rows, {'s0': 'unresolved amount'}, 5, 42)
    backwards, back_ledger = select_reviewed(rows[::-1], {'s0': 'unresolved amount'}, 5, 42)
    assert selected == backwards and ledger == back_ledger
    assert rows == original
    assert len(selected) == 5 and sum(bool(r['relations']) for r in selected) == 1
    assert len(ledger) == 8
    assert sum(r['status'] == 'deferred' for r in ledger) == 1
    assert all('s0' not in r['origin_sample_ids'] for r in selected)
    with pytest.raises(ValueError, match='unknown sample'):
        select_reviewed(rows, {'missing': 'reason'}, 5, 42)
    with pytest.raises(ValueError, match='insufficient'):
        select_reviewed(rows, {'s0': 'reason'}, 8, 42)


def package(tmp_path):
    source = tmp_path / 'source.jsonl'
    write_jsonl(source, population())
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'review_sha256': sha256_file(source), 'targets': 5,
        'seed': 42, 'deferred_samples': {'s0': 'unresolved'}, 'pricing_checked_date': '2026-10-28',
        'models': [{'provider': 'gemini', 'model': 'gemini-3.8-flash', 'temperature': None,
                    'input_usd_per_million': .75, 'output_usd_per_million': 3.75}]}))
    out = tmp_path / 'experiment'
    spec = prepare_reviewed(source, config, out, PROMPT)
    return source, out / 'reviewed_package.json', spec


def test_package_freezes_subset_and_never_exposes_gold(tmp_path):
    source, path, spec = package(tmp_path)
    assert len(list(read_jsonl(source))) == 8
    assert spec['targets'] == 5 and spec['initial_call_limit'] == 10 and spec['maximum_call_limit'] == 30
    for job in spec['jobs']:
        for row in read_jsonl(Path(job['input'])):
            assert 'review_note' not in relation_runner.visible_input(row)
            assert 'relations' not in json.loads(relation_runner.visible_input(row))
    load_reviewed(path)
    with pytest.raises(ValueError, match='not started'):
        analyze_reviewed(path, tmp_path / 'comparison')
    with Path(spec['gold_path']).open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='artifact changed'):
        load_reviewed(path)


def test_missing_approval_and_budget_stop_before_paid_call(tmp_path, monkeypatch):
    _, path, spec = package(tmp_path)
    approval = tmp_path / 'approval.json'
    approval.write_text(json.dumps({'package_sha256': sha256_file(path), 'explicit_user_approval': False,
        'maximum_calls': spec['maximum_call_limit'], 'ceiling_usd': 1}))
    with pytest.raises(ValueError, match='explicit approval'):
        run_reviewed(path, approval)
    approval.write_text(json.dumps({'package_sha256': sha256_file(path), 'explicit_user_approval': True,
        'maximum_calls': spec['maximum_call_limit'], 'ceiling_usd': .000001}))
    calls = []
    monkeypatch.setattr(relation_runner, 'build_client', lambda *a, **kw:
                        SimpleNamespace(call=lambda *a: calls.append(1)))
    import grain_prices.config
    monkeypatch.setattr(grain_prices.config, 'load_provider_keys', lambda *a: None)
    with pytest.raises(BudgetStop):
        run_reviewed(path, approval)
    assert calls == []
    assert not (path.parent / 'execution.lock').exists()


@pytest.mark.parametrize('level,temperature,expected', [(None, 0.0, 'thinking_budget'), ('medium', None, 'thinking_level')])
def test_gemini_level_config_and_legacy_budget_remain_distinct(monkeypatch, level, temperature, expected):
    genai = pytest.importorskip('google.genai')
    captured = {}
    def generate(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(text='{"relations": [], "contexts": []}')
    def make_client(**kwargs):
        captured['http_options'] = kwargs['http_options']
        return SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    monkeypatch.setenv('GEMINI_API_KEY', 'fixture')
    monkeypatch.setattr(genai, 'Client', make_client)
    client = clients.build_client('gemini', 'fixture', temperature, max_output_tokens=16000, thinking_level=level)
    client.call('system', 'user', relation_runner.PROVIDER_SCHEMA)
    config = captured['config'].model_dump(exclude_none=True)
    assert set(config['thinking_config']) == {expected}
    assert ('temperature' in config) == (temperature is not None)
    assert captured['http_options'].retry_options.attempts == 1


def test_gemini_level_rejects_deprecated_temperature():
    with pytest.raises(ValueError, match='unset temperature'):
        clients.build_client('gemini', 'gemini-3.8-flash', 0.0, thinking_level='medium')


def test_mock_execution_resume_and_analysis_keep_all_targets(tmp_path, monkeypatch):
    _, path, spec = package(tmp_path)
    approval = tmp_path / 'approval.json'
    approval.write_text(json.dumps({'package_sha256': sha256_file(path), 'explicit_user_approval': True,
        'maximum_calls': spec['maximum_call_limit'], 'ceiling_usd': 100}))
    calls = []
    def answer(*args):
        calls.append(1)
        return {'relations': [], 'contexts': []}
    monkeypatch.setattr(relation_runner, 'build_client', lambda *a, **kw: SimpleNamespace(call=answer))
    import grain_prices.config
    monkeypatch.setattr(grain_prices.config, 'load_provider_keys', lambda *a: None)
    run_reviewed(path, approval)
    assert len(calls) == 10
    run_reviewed(path, approval)
    assert len(calls) == 10
    report = analyze_reviewed(path, tmp_path / 'analysis')
    assert len(report['results']) == 2
    for row in report['results']:
        assert row['metrics']['gold_targets'] == 5
        assert row['metrics']['relationship']['fn'] == 1

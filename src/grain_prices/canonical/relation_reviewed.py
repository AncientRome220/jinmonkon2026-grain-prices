"""Reviewed subset selection, immutable comparison package and bounded execution."""
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import random

from grain_prices.canonical.io import read_jsonl, write_jsonl, sha256_file
from grain_prices.canonical.rebuild import write_json
from grain_prices.canonical.relations import validate_gold, evaluate_relations
from grain_prices.canonical.relation_windows import freeze_relations, window_candidate, within
from grain_prices.canonical.relation_provisional import ReservationLedger, source_code_hashes
from grain_prices.llm.relation_runner import (
    PROVIDER_SCHEMA, SCHEMA_SHA256, visible_input, score_relations, experiment_key, validate_answer,
)


def select_reviewed(rows, exclusions, count, seed):
    """Proportional positive/negative sampling, independent of any predictions."""
    validate_gold(rows)
    origins = [s for r in rows for s in r['origin_sample_ids']]
    if len(origins) != len(set(origins)):
        raise ValueError('duplicate sample origins')
    if set(exclusions) - set(origins):
        raise ValueError('exclusion names an unknown sample')
    if any(not str(reason).strip() for reason in exclusions.values()):
        raise ValueError('every exclusion needs a reason')
    eligible = [r for r in rows if not set(r['origin_sample_ids']) & set(exclusions)]
    if not 0 < count <= len(eligible):
        raise ValueError('insufficient eligible targets for requested sample')
    pools = {label: sorted([r for r in eligible if bool(r['relations']) == label],
                          key=lambda r: r['target_id']) for label in (False, True)}
    quotas = {label: count * len(pool) // len(eligible) for label, pool in pools.items()}
    order = sorted(pools, key=lambda label: (-(count * len(pools[label]) % len(eligible)), label))
    for label in order[:count - sum(quotas.values())]:
        quotas[label] += 1
    rng = random.Random(seed)
    selected = sorted([r for label in (False, True) for r in rng.sample(pools[label], quotas[label])],
                      key=lambda r: r['target_id'])
    selected_ids = {r['target_id'] for r in selected}
    ledger = []
    for row in sorted(rows, key=lambda r: r['target_id']):
        reasons = {s: exclusions[s] for s in row['origin_sample_ids'] if s in exclusions}
        ledger.append({'target_id': row['target_id'], 'origin_sample_ids': row['origin_sample_ids'],
                       'status': 'deferred' if reasons else 'selected' if row['target_id'] in selected_ids else 'eligible_not_drawn',
                       'reasons': reasons, 'has_relations': bool(row['relations'])})
    return selected, ledger


def reserve_amount(system, user, job):
    input_bound = len((system + user + json.dumps(PROVIDER_SCHEMA)).encode('utf-8')) + 8192
    output_bound = job['execution_settings']['max_output_tokens']
    return (Decimal(input_bound) * Decimal(str(job['input_usd_per_million']))
            + Decimal(output_bound) * Decimal(str(job['output_usd_per_million']))) / Decimal(1000000)


def prepare_reviewed(review: Path, config_path: Path, out: Path, prompt: Path):
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if sha256_file(review) != config['review_sha256']:
        raise ValueError('selection decisions refer to another review snapshot')
    rows = list(read_jsonl(review))
    selected, ledger = select_reviewed(rows, config['deferred_samples'], config['targets'], config['seed'])
    models = config['models']
    if len({(m['provider'], m['model']) for m in models}) != len(models):
        raise ValueError('duplicate comparison models')
    for model in models:
        if model['provider'] not in {'openai', 'anthropic', 'gemini'} or model['model'].endswith('-latest'):
            raise ValueError('use a supported provider and explicit model ID')
        if model.get('temperature') is not None:
            raise ValueError('this comparison uses provider-default sampling')
        if model['input_usd_per_million'] <= 0 or model['output_usd_per_million'] <= 0:
            raise ValueError('positive pricing required')
    if out.exists() and any(out.iterdir()):
        raise ValueError('use a new reviewed experiment directory')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'review_snapshot.jsonl').write_bytes(review.read_bytes())
    (out / 'selection_config.json').write_bytes(config_path.read_bytes())
    (out / prompt.name).write_bytes(prompt.read_bytes())
    prompt_copy = out / prompt.name
    write_json(out / 'selection_ledger.json', ledger)
    subset = out / 'selected_review.jsonl'
    write_jsonl(subset, selected)
    write_json(out / 'manifest_prepare_relations.json', {
        'historical_sample_rows': sum(len(r['origin_sample_ids']) for r in selected),
        'sample_sources': {s: {k: r[k] for k in ('doc_id', 'source_block_id', 'source_block_sha256')}
                           for r in selected for s in r['origin_sample_ids']},
        'parent_review_sha256': config['review_sha256'],
        'selection_ledger_sha256': sha256_file(out / 'selection_ledger.json'),
        'selection_scope': 'Explicit selected subset; all parent origins retained in selection ledger.',
    })
    freeze = freeze_relations(subset, out / 'gold.jsonl')
    system = prompt_copy.read_text(encoding='utf-8') + '\nOutput JSON schema:\n' + json.dumps(PROVIDER_SCHEMA, sort_keys=True)
    jobs, geometry = [], {}
    for radius in (25, 80):
        candidates = [window_candidate(r, radius) for r in selected]
        path = out / f'main_r{radius}.jsonl'
        write_jsonl(path, candidates)
        misses = []
        for row, candidate in zip(selected, candidates):
            bounds = [candidate['evidence_char_start'], candidate['evidence_char_end']]
            for index, rel in enumerate(row['relations']):
                if not within(rel['evidence_span'], bounds) or not all(within(c['span'], bounds)
                        for side in ('quantity', 'money') for c in rel[side]['components']):
                    misses.append({'target_id': row['target_id'], 'samples': row['origin_sample_ids'], 'relation_index': index})
        geometry[str(radius)] = misses
        for model in models:
            settings = {'max_output_tokens': 16000, 'prompt_version': 'target-grain-prompt-v2'}
            if model['provider'] == 'gemini':
                settings['gemini_thinking_level'] = 'medium'
            job = {**model, 'radius': radius, 'repeat': 1, 'max_validation_retries': 2,
                   'execution_settings': settings, 'targets': len(selected),
                   'input': str(path), 'input_sha256': sha256_file(path),
                   'initial_calls': len(selected), 'maximum_calls': 3 * len(selected),
                   'prediction_output': str(out / f"predictions_{model['model']}_r{radius}.jsonl")}
            job['reservation_main_usd'] = str(sum(reserve_amount(system, visible_input(c), job) for c in candidates))
            input_est = sum((len(system) + len(visible_input(c))) / 3 for c in candidates)
            job['scenario_usd_no_retries'] = {str(n): round((input_est * model['input_usd_per_million']
                + len(selected) * n * model['output_usd_per_million']) / 1e6, 2) for n in (2000, 8000)}
            # Offline runner validation: no client is constructed in dry-run mode.
            dry = score_relations(path, Path(job['prediction_output']), prompt_copy, model['provider'], model['model'],
                                  model.get('temperature'), dry_run=True, execution_settings=settings)
            if dry['candidate_count'] != len(selected):
                raise ValueError('dry-run population mismatch')
            jobs.append(job)
    artifacts = {str(p): sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()}
    spec = {'status': 'reviewed_subset_awaiting_explicit_paid_approval', 'targets': len(selected),
            'positive_targets': sum(bool(r['relations']) for r in selected), 'relations': freeze['relations'],
            'selection_counts': dict(Counter(r['status'] for r in ledger)), 'seed': config['seed'],
            'parent_review_sha256': config['review_sha256'], 'artifacts': artifacts,
            'gold_path': str(out / 'gold.jsonl'), 'prompt_path': str(prompt_copy),
            'prompt_sha256': sha256_file(prompt_copy), 'schema_sha256': SCHEMA_SHA256,
            'code_sha256': source_code_hashes(), 'jobs': jobs, 'cropped_relations': geometry,
            'initial_call_limit': sum(j['initial_calls'] for j in jobs),
            'maximum_call_limit': sum(j['maximum_calls'] for j in jobs),
            'reservation_main_usd': str(sum(Decimal(j['reservation_main_usd']) for j in jobs)),
            'reservation_with_retries_usd': str(3 * sum(Decimal(j['reservation_main_usd']) for j in jobs)),
            'scenario_usd_no_retries': {str(n): round(sum(j['scenario_usd_no_retries'][str(n)] for j in jobs), 2) for n in (2000, 8000)},
            'pricing_checked_date': config['pricing_checked_date'], 'freeze': freeze,
            'cost_note': 'Scenarios estimate input as characters/3. Reservations use UTF-8 bytes plus 8192 overhead and full output cap; no refunds. Taxes and FX excluded.',
            'scope': 'Single-reviewer AI-assisted development subset, conditional on adjudicability; not independent corpus-wide accuracy.',
            'stability': 'No additional repetitions in this package; do not claim stability.',
            'full_pass': 'Model/radius selected after comparison; separate population package and spending limit.',
            'dry_run_validated_jobs': len(jobs)}
    write_json(out / 'reviewed_package.json', spec)
    return spec


def load_reviewed(package: Path, check_code=False):
    spec = json.loads(package.read_text(encoding='utf-8'))
    if spec.get('status') != 'reviewed_subset_awaiting_explicit_paid_approval':
        raise ValueError('not a reviewed-subset package')
    for name, digest in spec['artifacts'].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError('frozen reviewed artifact changed')
    if sha256_file(Path(spec['prompt_path'])) != spec['prompt_sha256'] or SCHEMA_SHA256 != spec['schema_sha256']:
        raise ValueError('prompt/schema changed')
    if check_code and source_code_hashes() != spec['code_sha256']:
        raise ValueError('code changed; regenerate package before approval')
    validate_gold(list(read_jsonl(Path(spec['gold_path']))))
    return spec


def run_reviewed(package: Path, approval: Path):
    spec = load_reviewed(package, check_code=True)
    approved = json.loads(approval.read_text(encoding='utf-8'))
    digest = sha256_file(package)
    if approved.get('package_sha256') != digest or approved.get('explicit_user_approval') is not True:
        raise ValueError('explicit approval for this exact reviewed package required')
    if approved.get('maximum_calls') != spec['maximum_call_limit']:
        raise ValueError('approval call limit differs')
    lock = package.parent / 'execution.lock'
    with lock.open('x'):
        pass
    try:
        from grain_prices.config import load_provider_keys
        load_provider_keys(Path('.env'))
        ledger = ReservationLedger(package.parent / 'budget_ledger.json', digest, approved['ceiling_usd'], spec['maximum_call_limit'])
        for job in spec['jobs']:
            def reserve(key, system, user):
                ledger.reserve(key, reserve_amount(system, user, job))
            result = score_relations(Path(job['input']), Path(job['prediction_output']), Path(spec['prompt_path']),
                job['provider'], job['model'], job.get('temperature'), repeat=1,
                resume=Path(job['prediction_output']).exists(), execution_settings=job['execution_settings'],
                before_call=reserve, max_validation_retries=2)
            write_json(package.parent / f"manifest_{job['model']}_r{job['radius']}.json", result)
    finally:
        lock.unlink()


def analyze_reviewed(package: Path, output: Path):
    spec = load_reviewed(package)
    if not any(Path(j['prediction_output']).exists() for j in spec['jobs']):
        raise ValueError('no model output; comparison has not started')
    results = []
    for job in spec['jobs']:
        path = Path(job['prediction_output'])
        if not path.exists():
            results.append({'model': job['model'], 'radius': job['radius'], 'status': 'not_started'})
            continue
        candidates = {r['target_id']: r for r in read_jsonl(Path(job['input']))}
        seen = set()
        for row in read_jsonl(path):
            tid = row['target_id']
            if tid not in candidates or tid in seen:
                raise ValueError('duplicate or out-of-sample prediction')
            seen.add(tid)
            expected = experiment_key(candidates[tid], 0, job['provider'], job['model'], job.get('temperature'),
                                      spec['prompt_sha256'], job['execution_settings'])
            if (row.get('experiment_key') != expected or row.get('execution_settings') != job['execution_settings']
                    or row.get('repeat_index') != 0 or row.get('provider') != job['provider']
                    or row.get('model_id') != job['model'] or row.get('temperature') != job.get('temperature')):
                raise ValueError('prediction settings differ from package')
            if not row.get('error') and not row.get('validation_problems'):
                if validate_answer({**(row.get('answer') or {}), 'contexts': row.get('contexts', [])}, candidates[tid]):
                    raise ValueError('prediction claims validity but fails contract')
        metrics = evaluate_relations(path, Path(spec['gold_path']), Path(job['input']),
                                    output / f"{job['model']}_r{job['radius']}.json")
        results.append({'model': job['model'], 'radius': job['radius'], 'metrics': metrics})
    result = {'status': 'reviewed_development_subset', 'scope': spec['scope'], 'results': results,
              'package_sha256': sha256_file(package), 'cropped_relations': spec['cropped_relations']}
    write_json(output / 'comparison.json', result)
    return result

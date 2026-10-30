"""Offline input/call-budget preparation and paired analysis; no provider calls."""
from collections import Counter
import json
from pathlib import Path
import random

from grain_prices.canonical.io import read_jsonl, write_jsonl, sha256_file, base_manifest
from grain_prices.canonical.rebuild import write_json
from grain_prices.canonical.relations import validate_gold, linkage_key, reading_key, evaluate_relations
from grain_prices.canonical.relation_windows import window_candidate, target_from_candidate
from grain_prices.canonical.relations import GOLD_VERSION
from grain_prices.llm.relation_runner import validate_answer, experiment_key
from grain_prices.llm.relation_runner import visible_input, PROVIDER_SCHEMA, SCHEMA_SHA256

MODELS=[('openai','gpt-6.1-sol',None,2.0,10.0),('anthropic','claude-fable-5-1',None,10.0,50.0),
        ('gemini','gemini-2.5-flash',0.0,0.30,2.50)]


def prepare_experiment(review: Path, out: Path, prompt: Path, preview=False):
    rows=list(read_jsonl(review))
    freeze_record = None
    if not preview:
        validate_gold(rows)
        freeze_path = review.with_suffix('.manifest.json')
        if not freeze_path.exists():
            raise ValueError('experiment requires the sample-accounted frozen gold manifest')
        freeze_record = json.loads(freeze_path.read_text(encoding='utf8'))
        if (freeze_record.get('status') != 'reviewed_awaiting_run_authorization'
                or freeze_record.get('frozen_gold_sha256') != sha256_file(review)
                or freeze_record.get('confirmed_targets') != len(rows)
                or freeze_record.get('historical_sample_rows') != sum(len(r['origin_sample_ids']) for r in rows)):
            raise ValueError('frozen gold manifest differs from experiment gold')
    if len(rows)<30: raise ValueError('comparison requires at least 30 reviewed targets')
    if out.exists() and any(out.iterdir()): raise ValueError('use a new experiment package directory')
    out.mkdir(parents=True,exist_ok=True)
    selected=random.Random(20261023).sample(sorted(rows,key=lambda r:r['target_id']),30)
    selected_ids={r['target_id'] for r in selected}
    write_jsonl(out/'gold_all.jsonl',rows)
    write_jsonl(out/'gold_stability30.jsonl',selected)
    system=prompt.read_text(encoding='utf8')+'\nOutput JSON schema:\n'+json.dumps(PROVIDER_SCHEMA,sort_keys=True)
    jobs=[]
    for radius in (25,80):
        candidates=[window_candidate(r,radius) for r in rows]
        stability=[c for c in candidates if c['target_id'] in selected_ids]
        for label, inputs, repeat in (('main',candidates,1),('stability',stability,3)):
            path=out/f'{label}_r{radius}.jsonl'; write_jsonl(path,inputs)
            # Scenario estimate only: chars/3 is NOT a tokenizer or billed usage.
            input_est=sum((len(system)+len(visible_input(c)))/3 for c in inputs)*repeat
            for provider,model,temp,inprice,outprice in MODELS:
                jobs.append({'provider':provider,'model':model,'temperature':temp,'radius':radius,'phase':label,
                             'timeout_seconds':180.0, 'max_validation_retries':2,
                             'requested_max_output_tokens':16000 if provider=='anthropic' else None,
                             'schema_delivery':'shared prompt only' if provider=='anthropic' else 'shared prompt and provider JSON schema',
                             'input':str(path),'input_sha256':sha256_file(path),'repeat':repeat,
                             'targets':len(inputs),'initial_calls':len(inputs)*repeat,'maximum_calls':len(inputs)*repeat*3,
                             'estimated_input_tokens_chars_div_3':round(input_est),
                             'cost_scenarios_usd_no_retries':{str(output_tokens):round((input_est*inprice+len(inputs)*repeat*output_tokens*outprice)/1e6,2)
                                 for output_tokens in (2000,8000)},
                             'prediction_output':str(out/f'predictions_{model}_{label}_r{radius}.jsonl')})
    result={'status':'preview_not_scoreable' if preview else 'frozen_inputs_awaiting_explicit_approval',
            'historical_origins':sum(len(r['origin_sample_ids']) for r in rows),'targets':len(rows),
            'confirmed_targets':sum(r.get('review_status')=='confirmed' for r in rows),
            'seed':20261023,'stability_targets':sorted(selected_ids),'jobs':jobs,
            'initial_call_limit':sum(j['initial_calls'] for j in jobs),'maximum_call_limit':sum(j['maximum_calls'] for j in jobs),
            'prompt_path':str(prompt),'prompt_sha256':sha256_file(prompt),'schema_sha256':SCHEMA_SHA256,
            'review_sha256':sha256_file(review),'code':base_manifest(Path.cwd(),out.parent.name,'relation-experiment',{}),
            'freeze_record':freeze_record,
            'gold_sha256':{name:sha256_file(out/name) for name in ('gold_all.jsonl','gold_stability30.jsonl')},
            'pricing_checked_date':'2026-10-23',
            'pricing_sources':['https://developers.openai.com/api/docs/models/gpt-6.1-sol',
                'https://platform.claude.com/docs/en/models/fable-5-1/overview','https://ai.google.dev/gemini-api/docs/pricing#gemini-2.5-flash'],
            'cost_limitations':'Scenarios use characters/3 input estimate and 2000 or 8000 output/thinking tokens per call, standard uncached prices. Not measured token counts, a quote, or a spend cap. Retries can triple these scenarios. Refresh after gold freeze.',
            'full_pass':'separate package and approval after radius decision; excluded from these limits'}
    result['cost_scenarios_total_usd_no_retries']={str(n):round(sum(j['cost_scenarios_usd_no_retries'][str(n)] for j in jobs),2) for n in (2000,8000)}
    write_json(out/'experiment_package.json',result)
    return result


def analyze_experiment(package: Path, output: Path):
    spec=json.loads(package.read_text(encoding='utf8'))
    if spec.get('status', '').startswith('provisional_'):
        raise ValueError('provisional experiment requires provisional-analyze, not formal evaluation')
    if spec['status']=='preview_not_scoreable': raise ValueError('cannot score preview gold')
    if sha256_file(Path(spec['prompt_path'])) != spec['prompt_sha256'] or spec['schema_sha256'] != SCHEMA_SHA256:
        raise ValueError('experiment prompt or schema changed')
    results=[]; signatures={}
    for job in spec['jobs']:
        predictions=Path(job['prediction_output']); inputs=Path(job['input'])
        if sha256_file(inputs)!=job['input_sha256']: raise ValueError('experiment input changed')
        by_id = {c['target_id']: c for c in read_jsonl(inputs)}
        for p in (read_jsonl(predictions) if predictions.exists() else []):
            ri = p.get('repeat_index')
            if p['target_id'] not in by_id or type(ri) is not int or not 0 <= ri < job['repeat']:
                raise ValueError('prediction outside approved experiment job')
            expected = experiment_key(by_id[p['target_id']], ri, job['provider'], job['model'], job['temperature'], spec['prompt_sha256'])
            if (p.get('experiment_key') != expected or p.get('model_id') != job['model']
                    or p.get('provider') != job['provider'] or p.get('temperature') != job['temperature']):
                raise ValueError('prediction conditions differ from approved experiment job')
        gold=package.parent/('gold_all.jsonl' if job['phase']=='main' else 'gold_stability30.jsonl')
        if sha256_file(gold)!=spec['gold_sha256'][gold.name]: raise ValueError('frozen experiment gold changed')
        for ri in range(job['repeat']):
            dest=output/f"{job['model']}_{job['phase']}_r{job['radius']}_repeat{ri}.json"
            metrics=evaluate_relations(predictions,gold,inputs,dest,ri)
            results.append({'model':job['model'],'radius':job['radius'],'phase':job['phase'],'repetition':ri,'metrics':metrics})
        if job['phase']=='stability':
            for row in (read_jsonl(predictions) if predictions.exists() else []):
                key=(job['model'],job['radius'],row['target_id'])
                answer=row.get('answer')
                if row.get('error') or row.get('validation_problems') or not answer:
                    signature=None
                else:
                    signature=repr(sorted([(linkage_key(r),reading_key(r),r['rate_basis']) for r in answer['relations']],key=repr))
                signatures.setdefault(key,{})[row['repeat_index']]=signature
    stability=[]
    for _, model, *_ in MODELS:
        for radius in (25,80):
            relevant=[values for (m,r,t),values in signatures.items() if m==model and r==radius]
            stability.append({'model':model,'radius':radius,'denominator':30,
                'three_valid_identical_answers':sum(len(v)==3 and all(s is not None for s in v.values()) and len(set(v.values()))==1 for v in relevant)})
    comparisons=[]
    for _,model,*_ in MODELS:
        pair={r['radius']:r['metrics'] for r in results if r['model']==model and r['phase']=='main'}
        comparisons.append({'model':model,'relationship_f1_delta_25_minus_80':pair[25]['relationship']['f1']-pair[80]['relationship']['f1'],
                            'complete_f1_delta_25_minus_80':pair[25]['complete_relation']['f1']-pair[80]['complete_relation']['f1'],
                            'paired_targets': [{'target_id': a['target_id'], 'radius25': a, 'radius80': b}
                                for a, b in zip(pair[25]['target_diagnostics'], pair[80]['target_diagnostics'])]})
    result={'results':results,'stability':stability,'paired_conditions':comparisons,
            'interpretation':'Development on reused historical targets, not independent population accuracy or a causal model-quality comparison.'}
    write_json(output/'comparison.json',result)
    return result


def prepare_post_llm_review(candidates_path: Path, predictions_path: Path, out: Path):
    """Real predictions are required; never substitute a mock full pass."""
    candidates={c['target_id']:c for c in read_jsonl(candidates_path)}
    predictions=list(read_jsonl(predictions_path))
    if len({(p.get('provider'),p.get('model_id'),p.get('repeat_index')) for p in predictions}) != 1:
        raise ValueError('post-LLM sample must use exactly one model and repetition')
    if len({p['target_id'] for p in predictions})!=len(predictions): raise ValueError('duplicate predictions')
    from grain_prices.canonical.provenance import validate_candidate_binding
    pools={True:[],False:[]}; failures=[]
    for p in predictions:
        c=candidates[p['target_id']]; validate_candidate_binding(p,c)
        answer=p.get('answer')
        if p.get('error') or p.get('validation_problems') or not answer or validate_answer({**answer,'contexts':p.get('contexts',[])},c):
            failures.append(p['target_id']); continue
        pools[bool(answer['relations'])].append(p)
    rng=random.Random(20261023); chosen=[]
    for label in (True,False):
        pool=sorted(pools[label],key=lambda p:p['target_id'])
        chosen+=rng.sample(pool,min(15,len(pool)))
    chosen_ids={p['target_id'] for p in chosen}
    rest=sorted([p for pool in pools.values() for p in pool if p['target_id'] not in chosen_ids],key=lambda p:p['target_id'])
    if len(chosen)<30:
        chosen+=rng.sample(rest,min(30-len(chosen),len(rest)))
    if len(chosen)!=30: raise ValueError('fewer than 30 valid full-pass targets')
    if out.exists() and any(out.iterdir()): raise ValueError('post-LLM review directory already exists')
    rows=[]
    for p in chosen:
        c=candidates[p['target_id']]
        row={**c,'schema_version':GOLD_VERSION,'origin_sample_ids':[c['target_id']],
             'origin_annotations':[p], 'relation_proposals':[], 'relations':p['answer']['relations'],
             'prediction_before_review':p,'review_phase':'post_llm_check','review_status':'','reviewed_by':'',
             'coverage':'unreviewed','review_note':'','preparation_flags':['model_prediction_visible_for_correction']}
        rows.append(row)
    write_jsonl(out/'relation_gold_review.jsonl',rows)
    result={'seed':20261023,'targets':30,'predicted_positive':sum(bool(p['answer']['relations']) for p in chosen),
            'predicted_negative':sum(not p['answer']['relations'] for p in chosen),'invalid_or_errored_targets':failures,
            'missing_prediction_targets':sorted(set(candidates)-{p['target_id'] for p in predictions}),
            'inputs':{str(x):sha256_file(x) for x in (candidates_path,predictions_path)},
            'status':'awaiting_actual_human_checking','population_correction_rate':None,
            'limitation':'Balanced diagnostic sample. No population rate without sampling weights, no time savings without baseline.'}
    write_json(out/'post_llm_sample_manifest.json',result)
    return result


def prepare_full_pass(candidates_path: Path, out: Path, radius: int, prompt: Path):
    """Prepare one chosen population; this does not authorize generation."""
    if radius not in (25, 80):
        raise ValueError('choose radius 25 or 80 after the paired comparison')
    if out.exists() and any(out.iterdir()):
        raise ValueError('use a new full-pass package directory')
    targets = {}
    origins = {}
    for c in read_jsonl(candidates_path):
        from grain_prices.canonical.provenance import validate_candidate_binding
        validate_candidate_binding(c, c)
        t = target_from_candidate(c)
        targets.setdefault(t['target_id'], t)
        origins.setdefault(t['target_id'], []).append(c['window_id'])
    path = out/'full_pass_inputs.jsonl'
    inputs = [window_candidate(t, radius) for t in targets.values()]
    write_jsonl(path, inputs)
    system = prompt.read_text(encoding='utf8')+'\nOutput JSON schema:\n'+json.dumps(PROVIDER_SCHEMA,sort_keys=True)
    input_est = sum((len(system)+len(visible_input(c)))/3 for c in inputs)
    result = {'status': 'awaiting_separate_explicit_approval', 'radius': radius,
              'model': 'claude-fable-5-1', 'provider': 'anthropic', 'temperature': None,
              'repeat': 1, 'initial_call_limit': len(targets), 'maximum_call_limit': 3*len(targets),
              'timeout_seconds': 180.0, 'max_validation_retries': 2, 'requested_max_output_tokens': 16000,
              'estimated_input_tokens_chars_div_3': round(input_est),
              'cost_scenarios_usd_no_retries': {str(n): round((input_est*10+len(targets)*n*50)/1e6,2) for n in (2000,8000)},
              'cost_limitations': 'Standard uncached $10/$50 per million input/output tokens; chars/3 input estimate, scenarios not spend caps; retries can triple costs. Refresh pricing before approval.',
              'input': str(path), 'input_sha256': sha256_file(path),
              'population_sha256': sha256_file(candidates_path), 'prompt_sha256': sha256_file(prompt),
              'schema_sha256': SCHEMA_SHA256, 'source_window_ids_by_target': origins,
              'code': base_manifest(Path.cwd(), out.parent.name, 'relation-full-pass', {}),
              'limitation': 'Population is the corrected candidate population, not all corpus grain mentions. Targets without options after cropping are retained.'}
    write_json(out/'full_pass_package.json', result)
    return result


def summarize_checking(review: Path, output: Path):
    rows = list(read_jsonl(review))
    if any(r.get('review_phase') != 'post_llm_check' for r in rows):
        raise ValueError('only actual post-LLM review sessions can measure checking effort')
    confirmed = [r for r in rows if r.get('review_status') == 'confirmed']
    sessions = [s for r in rows for s in r.get('review_sessions', []) if s.get('phase') == 'post_llm_check']
    result = {'sample_targets': len(rows), 'confirmed_targets': len(confirmed),
              'changed_relation_targets_among_confirmed': sum(r['relations'] != r['prediction_before_review']['answer']['relations'] for r in confirmed),
              'review_sessions': sessions,
              'recorded_active_seconds': sum(s.get('active_seconds', 0) for s in sessions),
              'population_correction_rate': None, 'time_savings': None,
              'review_sha256': sha256_file(review),
              'limitation': 'Incomplete sessions remain visible; balanced diagnostic sample is not a population rate or a comparable time-saving baseline.'}
    write_json(output, result)
    return result

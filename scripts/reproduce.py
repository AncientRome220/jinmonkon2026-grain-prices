"""Recompute the paper's six evaluations without network or provider calls."""
from pathlib import Path
import csv
import json
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from grain_prices.canonical.relations import evaluate_relations

def main():
    # Fail closed if a future change attempts a network connection.
    def forbidden(*args,**kwargs):
        raise RuntimeError('Network access is forbidden during saved-output evaluation')
    socket.create_connection=forbidden
    socket.socket.connect=forbidden
    expected=json.loads((ROOT/'results/evaluation_expected.json').read_text(encoding='utf-8'))
    out=ROOT/'reproduced'; out.mkdir(exist_ok=True)
    summary=[]
    for job in expected['jobs']:
        name=f"{job['model']}_r{job['radius']}"
        result=evaluate_relations(ROOT/job['predictions'],ROOT/'data/evaluation/gold.jsonl',
                                  ROOT/job['candidates'],out/(name+'.json'))
        for key,value in job['expected'].items():
            if result[key]!=value:
                raise AssertionError(f'{name}: {key} differs from the frozen experiment')
        assert not result['targets_without_candidate']
        summary.append(dict(model=job['model'],radius=job['radius'],
                            position_p=result['relationship']['precision'],position_r=result['relationship']['recall'],
                            position_f1=result['relationship']['f1'],complete_p=result['complete_relation']['precision'],
                            complete_r=result['complete_relation']['recall'],complete_f1=result['complete_relation']['f1'],
                            positive_targets_with_output=result['target_has_valuation']['tp'],
                            negative_targets_with_output=result['target_has_valuation']['fp']))
    with (out/'table4.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    full=json.loads((ROOT/'results/full_population_summary.json').read_text(encoding='utf-8'))
    assert sum(full['commodity_relation_counts'].values())==full['emitted_relations']==2080
    assert full['targets']==full['valid_targets']+full['errored_targets']+full['invalid_targets']==9222
    assert full['valid_targets_without_pairs']==7328
    assert full['human_verified'] is False and full['individual_outputs_distributed'] is False
    print('PASS: 150 targets, 36 reference pairs, 6 conditions / 900 outputs; all evaluation fields match.')
    print('PASS: full-population aggregate consistency (individual full-population outputs are not distributed).')
    for r in summary: print(f"{r['model']} r{r['radius']}: position F1={r['position_f1']:.3f}; complete F1={r['complete_f1']:.3f}")

if __name__=='__main__':main()

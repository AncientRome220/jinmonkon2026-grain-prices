"""Verify the exact release inventory, private-field exclusion and source binding."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from grain_prices.canonical.provenance import validate_candidate_binding
from grain_prices.canonical.relations import validate_gold
from grain_prices.llm.relation_runner import visible_input

def main():
    manifest=json.loads((ROOT/'provenance/export_manifest.json').read_text(encoding='utf-8'))
    expected={r['path'] for r in manifest['files']}|{'provenance/export_manifest.json'}
    if (ROOT/'.git').exists():
        tracked=set(subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z']).decode().strip('\0').split('\0'))
        assert tracked==expected,('inventory mismatch',tracked-expected,expected-tracked)
    for row in manifest['files']:
        path=ROOT/row['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],row['path']
        assert path.stat().st_size==row['bytes'],row['path']
        if path.suffix in {'.py','.md','.json','.jsonl','.toml','.yml','.cff','.lock'}:
            text=path.read_text(encoding='utf-8-sig')
            assert not re.search(r'[A-Za-z]:[\\/](?:Users|research)[\\/]',text),row['path']
            assert not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}|sk-[A-Za-z0-9_-]{30,}|AIza[A-Za-z0-9_-]{30,})',text),row['path']
            assert ('Obsidian'+'/') not in text,row['path']
    readrows=lambda name:[json.loads(s) for s in (ROOT/'data/evaluation'/name).read_text(encoding='utf-8').splitlines()]
    gold=readrows('gold.jsonl');validate_gold(gold)
    assert len(gold)==150 and sum(len(r['relations']) for r in gold)==36
    assert {r['reviewed_by'] for r in gold}=={'reviewer-1'}
    denied={'ai_draft','ai_review_recommendation','review_sessions','review_note','relation_proposals','origin_annotations'}
    assert all(not denied.intersection(r) for r in gold)
    for radius in (25,80):
        candidates=readrows(f'main_r{radius}.jsonl')
        assert len(candidates)==150
        for row in candidates:validate_candidate_binding(row,row);visible_input(row)
    print(f'PASS: {len(expected)} allowlisted files; hashes, privacy fields and input fingerprints verified.')

if __name__=='__main__':main()

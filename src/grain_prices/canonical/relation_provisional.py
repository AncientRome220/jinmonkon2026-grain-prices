"""Explicitly non-gold development comparison with immutable reference snapshots."""
from collections import Counter
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path

from grain_prices.canonical.io import read_jsonl, write_jsonl, sha256_file
from grain_prices.canonical.rebuild import write_json
from grain_prices.canonical.relation_experiment import MODELS
from grain_prices.canonical.relation_windows import window_candidate
from grain_prices.canonical.relations import (
    validate_target, validate_relation, linkage_key, _evaluate_relation_rows,
)
from grain_prices.llm.relation_runner import (
    PROVIDER_SCHEMA, SCHEMA_SHA256, visible_input, validate_answer, experiment_key,
    score_relations,
)

WARNING = "暫定：人手確認未完了。下書きとの一致度であり、正式な抽出精度ではない。"
SETTINGS = {"max_output_tokens": 16000, "gemini_thinking_budget": 8192,
            "input_reservation": "UTF-8 bytes of prompt, schema and input plus 8192 overhead tokens",
            "budget_policy": "reserve before request; no refunds, including failed requests"}
PRICING = {
    "openai": [2.0, 10.0, "https://developers.openai.com/api/docs/models/gpt-6.1-sol"],
    "anthropic": [10.0, 50.0, "https://platform.claude.com/docs/en/models/fable-5-1/overview"],
    "gemini": [0.30, 2.50, "https://ai.google.dev/gemini-api/docs/pricing#gemini-2.5-flash"],
}


def source_code_hashes():
    root = Path(__file__).resolve().parents[1]
    return {str(p): sha256_file(p) for p in sorted(root.rglob("*.py"))}


def reservation(system, user, provider):
    # Deliberately over-reserve rather than treat characters/3 as a spending cap.
    input_bound = len((system + user + json.dumps(PROVIDER_SCHEMA)).encode("utf8")) + 8192
    output_bound = SETTINGS["max_output_tokens"] + (8192 if provider == "gemini" else 0)
    i, o, _ = PRICING[provider]
    return (Decimal(input_bound)*Decimal(str(i)) + Decimal(output_bound)*Decimal(str(o)))/Decimal(1000000)


def validate_draft(rows):
    if not rows or len({r["target_id"] for r in rows}) != len(rows):
        raise ValueError("empty or duplicate provisional references")
    for r in rows:
        validate_target(r)
        if not r.get("provisional_reference"):
            raise ValueError("explicit provisional reference marker required")
        if not r["provisional_reference"]["scoreable"]:
            continue
        keys = []
        for rel in r["relations"]:
            errors = validate_relation(rel, r)
            if errors:
                raise ValueError("; ".join(errors))
            keys.append(linkage_key(rel))
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate draft relations")


def prepare_provisional(review: Path, out: Path, prompt: Path, corrections: Path):
    if out.exists() and any(out.iterdir()):
        raise ValueError("use a new provisional directory")
    original = list(read_jsonl(review))
    decisions = json.loads(corrections.read_text(encoding="utf8"))
    if decisions["review_sha256"] != sha256_file(review):
        raise ValueError("three-case audit is bound to another review snapshot")
    by_id = {d["target_id"]: d for d in decisions["decisions"]}
    if len(by_id) != len(decisions["decisions"]) or set(by_id)-{r["target_id"] for r in original}:
        raise ValueError("duplicate or out-of-sample corrections")
    rows = deepcopy(original)
    for row in rows:
        human = row.get("review_status") == "confirmed"
        group = "human_confirmed" if human else "ai_draft"
        hold = row.get("ai_draft", {}).get("disposition") == "needs_review"
        reason = row.get("ai_draft", {}).get("reason", "") if hold else ""
        decision = by_id.get(row["target_id"])
        if decision:
            if decision["source_block_sha256"] != row["source_block_sha256"]:
                raise ValueError("correction source mismatch")
            hold, reason = decision["hold"], decision["reason"]
            if decision.get("relations") is not None:
                row["relations"] = decision["relations"]
                group = "ai_corrected"
                # Original human metadata lives only in the untouched snapshot.
                row.update(review_status="", reviewed_by="", coverage="unreviewed")
                row.pop("review_sessions", None)
        row["provisional_reference"] = {"scoreable": not hold, "reason": reason,
                                         "provenance": group, "human_review_incomplete": True}
    validate_draft(rows)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out/"review_snapshot.jsonl", original)
    write_jsonl(out/"draft_reference.jsonl", rows)
    write_json(out/"three_case_audit.json", decisions)
    jobs=[]
    system=prompt.read_text(encoding="utf8")+'\nOutput JSON schema:\n'+json.dumps(PROVIDER_SCHEMA,sort_keys=True)
    for radius in (25,80):
        inputs=[window_candidate(r,radius) for r in original]
        path=out/f"main_r{radius}.jsonl"
        write_jsonl(path,inputs)
        for provider,model,temp,*_ in MODELS:
            reserves={c["target_id"]:str(reservation(system,visible_input(c),provider)) for c in inputs}
            chars=sum(len(system)+len(visible_input(c)) for c in inputs)
            ip,op,_=PRICING[provider]
            jobs.append({"provider":provider,"model":model,"temperature":temp,"radius":radius,
                "repeat":1,"max_validation_retries":2,"execution_settings":SETTINGS,
                "input":str(path),"input_sha256":sha256_file(path),"targets":len(inputs),
                "initial_calls":len(inputs),"maximum_calls":3*len(inputs),
                "prediction_output":str(out/f"predictions_{model}_r{radius}.jsonl"),
                "reservation_usd_by_target":reserves,
                "scenario_usd_no_retries":{str(n):round((chars/3*ip+len(inputs)*n*op)/1e6,2) for n in (2000,8000)}})
    spec={"status":"provisional_awaiting_explicit_paid_approval","warning":WARNING,
          "targets":len(rows),"reviewed_in_source":sum(r.get("review_status")=="confirmed" for r in original),
          "scoreable_targets":sum(r["provisional_reference"]["scoreable"] for r in rows),
          "scoreable_by_provenance":dict(Counter(r["provisional_reference"]["provenance"] for r in rows if r["provisional_reference"]["scoreable"])),
          "holds":[{"target_id":r["target_id"],"sample_ids":r["origin_sample_ids"],"reason":r["provisional_reference"]["reason"]} for r in rows if not r["provisional_reference"]["scoreable"]],
          "review_source":str(review),"review_source_sha256":sha256_file(review),
          "artifacts":{str(out/n):sha256_file(out/n) for n in ("review_snapshot.jsonl","draft_reference.jsonl","three_case_audit.json")},
          "prompt_path":str(prompt),"prompt_sha256":sha256_file(prompt),"schema_sha256":SCHEMA_SHA256,
          "code_sha256":source_code_hashes(),"jobs":jobs,"pricing":PRICING,"pricing_checked_date":"2026-10-23",
          "initial_call_limit":sum(j["initial_calls"] for j in jobs),"maximum_call_limit":sum(j["maximum_calls"] for j in jobs),
          "reservation_main_usd":str(sum(Decimal(v) for j in jobs for v in j["reservation_usd_by_target"].values())),
          "reservation_with_retries_usd":str(3*sum(Decimal(v) for j in jobs for v in j["reservation_usd_by_target"].values())),
          "cost_note":"Scenario estimates are not quotes. Reservations conservatively charge the full potential request; no refunds. Gemini separately reserves thinking. Taxes/FX excluded.",
          "repetitions":"additional stability repetitions deferred","formal_gold_frozen":False}
    write_json(out/"provisional_package.json",spec)
    return spec


def load_package(path, check_code=False):
    s=json.loads(path.read_text(encoding="utf8"))
    if s.get("status") != "provisional_awaiting_explicit_paid_approval":
        raise ValueError("not a provisional experiment package")
    for name,digest in s["artifacts"].items():
        if sha256_file(Path(name)) != digest: raise ValueError("reference artifact changed")
    for j in s["jobs"]:
        if sha256_file(Path(j["input"])) != j["input_sha256"]: raise ValueError("input changed")
    if sha256_file(Path(s["prompt_path"]))!=s["prompt_sha256"] or SCHEMA_SHA256!=s["schema_sha256"]:
        raise ValueError("prompt/schema changed")
    if check_code and source_code_hashes()!=s["code_sha256"]:
        raise ValueError("code changed; regenerate package before approval")
    return s


def analyze_provisional(package: Path, output: Path):
    spec=load_package(package)
    rows=list(read_jsonl(package.parent/"draft_reference.jsonl")); validate_draft(rows)
    selected=[r for r in rows if r["provisional_reference"]["scoreable"]]
    selected_ids={r["target_id"] for r in selected}
    results=[]
    output.mkdir(parents=True,exist_ok=True)
    if not any(Path(j['prediction_output']).exists() for j in spec['jobs']):
        raise ValueError("no model output; do not manufacture a zero-score experiment")
    for j in spec["jobs"]:
        ppath=Path(j["prediction_output"])
        if not ppath.exists():
            results.append({"model":j["model"],"radius":j["radius"],"status":"not_started","population":len(rows)})
            continue
        candidates=list(read_jsonl(Path(j["input"]))); by_id={r["target_id"]:r for r in candidates}
        preds=list(read_jsonl(ppath)); seen=set(); valid=[]; invalid=[]; errors=[]
        for p in preds:
            tid=p["target_id"]
            if tid in seen or tid not in by_id: raise ValueError("duplicate or out-of-sample prediction")
            seen.add(tid)
            expected=experiment_key(by_id[tid],0,j["provider"],j["model"],j["temperature"],spec["prompt_sha256"],j["execution_settings"])
            if (p.get("experiment_key")!=expected or p.get("repeat_index")!=0
                    or p.get("model_id")!=j["model"] or p.get("provider")!=j["provider"]
                    or p.get("temperature")!=j["temperature"] or p.get("execution_settings")!=j["execution_settings"]):
                raise ValueError("prediction settings differ from package")
            if p.get("error"): errors.append(tid)
            elif p.get("validation_problems") or validate_answer({**(p.get("answer") or {}),"contexts":p.get("contexts",[])},by_id[tid]):
                invalid.append(tid)
                p={**p,"validation_problems":["provisional analysis validation failed"]}
            else: valid.append(tid)
            if p.get("candidate_fingerprint")!=by_id[tid]["candidate_fingerprint"]:
                raise ValueError("prediction source fingerprint mismatch")
        metrics={}
        for group in ("all_scoreable","human_confirmed","ai_draft","ai_corrected"):
            subset=[r for r in selected if group=="all_scoreable" or r["provisional_reference"]["provenance"]==group]
            ids={r["target_id"] for r in subset}
            folder=output/f'{j["model"]}_r{j["radius"]}'/group; folder.mkdir(parents=True,exist_ok=True)
            ip=folder/"inputs.jsonl"; pp=folder/"predictions.jsonl"
            write_jsonl(ip,[r for r in candidates if r["target_id"] in ids])
            filtered=[{**p,"validation_problems":["invalid contexts or relation"]} if p["target_id"] in invalid else p
                      for p in preds if p["target_id"] in ids]
            write_jsonl(pp,filtered)
            dest=folder/"agreement.json"
            result=_evaluate_relation_rows(pp,subset,ip,dest,0,package.parent/"draft_reference.jsonl")
            result.update(status="provisional_draft_agreement",warning=WARNING,note=WARNING,
                          reference_targets=result.pop("gold_targets"),reference_relations=result.pop("gold_relations"))
            write_json(dest,result); metrics[group]=result
        results.append({"model":j["model"],"radius":j["radius"],"population":len(rows),
            "valid":len(valid),"errors":errors,"invalid":invalid,"missing":sorted(set(by_id)-seen),
            "agreement":metrics,"held_predictions":[p for p in preds if p["target_id"] not in selected_ids]})
    paired=[]
    for _,model,*_ in MODELS:
        pair={r['radius']:r['agreement']['all_scoreable'] for r in results if r['model']==model and 'agreement' in r}
        if set(pair)=={25,80}:
            wider={r['target_id']:r for r in pair[80]['target_diagnostics']}
            paired.append({'model':model,'relationship_agreement_f1_delta_25_minus_80':pair[25]['relationship']['f1']-pair[80]['relationship']['f1'],
                'complete_agreement_f1_delta_25_minus_80':pair[25]['complete_relation']['f1']-pair[80]['complete_relation']['f1'],
                'targets':[{'target_id':r['target_id'],'radius25':r,'radius80':wider[r['target_id']]} for r in pair[25]['target_diagnostics']]})
    result={"status":"provisional_draft_agreement","warning":WARNING,"results":results,"holds":spec["holds"],"paired_conditions":paired,
            "package_sha256":sha256_file(package),"predictions_sha256":{j["prediction_output"]:sha256_file(Path(j["prediction_output"])) if Path(j['prediction_output']).exists() else None for j in spec["jobs"]}}
    write_json(output/"comparison.json",result)
    return result


class BudgetStop(SystemExit):
    """Must escape runner's recoverable per-target exception handler."""


class ReservationLedger:
    def __init__(self, path, package_hash, ceiling, maximum_calls):
        self.path=Path(path); self.ceiling=Decimal(str(ceiling)); self.maximum_calls=maximum_calls
        if not self.ceiling.is_finite() or self.ceiling<=0: raise ValueError("positive finite budget required")
        self.state=json.loads(self.path.read_text()) if self.path.exists() else {
            "package_sha256":package_hash,"ceiling_usd":str(self.ceiling),"reserved_usd":"0","requests":[]}
        if self.state["package_sha256"]!=package_hash or Decimal(self.state["ceiling_usd"])!=self.ceiling:
            raise ValueError("budget resume differs from approval")

    def reserve(self, key, amount):
        cost=Decimal(str(amount)); used=Decimal(self.state["reserved_usd"])
        if cost<=0 or not cost.is_finite(): raise ValueError("invalid reservation")
        if used+cost>self.ceiling or len(self.state["requests"])>=self.maximum_calls:
            raise BudgetStop("Stopped before provider call: approved reservation budget/call limit reached")
        if sum(r["key"]==key for r in self.state["requests"])>=3:
            raise BudgetStop("Stopped: this target exhausted its three reserved calls")
        self.state["requests"].append({"key":key,"reserved_usd":str(cost)})
        self.state["reserved_usd"]=str(used+cost)
        temp=self.path.with_suffix('.tmp'); write_json(temp,self.state); temp.replace(self.path)


def run_provisional(package: Path, approval: Path):
    spec=load_package(package,check_code=True)
    approved=json.loads(approval.read_text(encoding="utf8"))
    digest=sha256_file(package)
    if approved.get("package_sha256")!=digest or not approved.get("explicit_user_approval"):
        raise ValueError("explicit approval for this exact package required")
    if approved.get("maximum_calls")!=spec["maximum_call_limit"]:
        raise ValueError("approval call limit differs")
    lock=package.parent/"execution.lock"
    with lock.open('x'):
        pass
    try:
        from grain_prices.config import load_provider_keys
        load_provider_keys(Path('.env'))
        ledger=ReservationLedger(package.parent/"budget_ledger.json",digest,approved["ceiling_usd"],spec["maximum_call_limit"])
        for j in spec["jobs"]:
            def reserve(key,system,user):
                ledger.reserve(key,reservation(system,user,j["provider"]))
            result=score_relations(Path(j["input"]),Path(j["prediction_output"]),Path(spec["prompt_path"]),
                j["provider"],j["model"],j["temperature"],repeat=1,resume=Path(j["prediction_output"]).exists(),
                execution_settings=j["execution_settings"],before_call=reserve,max_validation_retries=2)
            write_json(package.parent/f'manifest_{j["model"]}_r{j["radius"]}.json',result)
    finally:
        lock.unlink()

"""Provider runner for source-anchored relations; historical scoring is separate."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time

from grain_prices.canonical.contracts import CONTEXT_VALUES, LINKAGE_VALUES
from grain_prices.canonical.io import read_jsonl, sha256_file, write_jsonl
from grain_prices.canonical.provenance import candidate_fingerprint, validate_candidate_binding
from grain_prices.canonical.relations import (
    OBJECTIVE_VERSION, PREDICTION_VERSION, PREDICTION_SCHEMA, validate_relation, validate_target,
)
from grain_prices.llm.canonical_runner import ScoringStopped
from grain_prices.llm.clients import ProviderError, build_client

PROMPT_VERSION = "target-grain-prompt-v1"
PROVIDER_SCHEMA = deepcopy(PREDICTION_SCHEMA)
PROVIDER_SCHEMA["required"].append("contexts")
PROVIDER_SCHEMA["properties"]["contexts"] = {
    "type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["relation_index", "context_tags", "linkage_strength"],
        "properties": {
            "relation_index": {"type": "integer"},
            "context_tags": {"type": "array", "items": {"type": "string", "enum": sorted(CONTEXT_VALUES)}, "minItems": 1},
            "linkage_strength": {"type": "string", "enum": sorted(LINKAGE_VALUES)},
        },
    },
}
SCHEMA_SHA256 = hashlib.sha256(json.dumps(PROVIDER_SCHEMA, sort_keys=True).encode()).hexdigest()


def validate_input(candidate):
    validate_target(candidate)
    if candidate.get("objective_version") != OBJECTIVE_VERSION or not candidate.get("candidate_fingerprint"):
        raise ValueError("relation input requires its objective and fingerprint")
    validate_candidate_binding(candidate, candidate)
    lo, hi = candidate["evidence_char_start"], candidate["evidence_char_end"]
    if not (0 <= lo <= candidate["grain_span"][0] < candidate["grain_span"][1] <= hi <= len(candidate["source_block_text"])):
        raise ValueError("target outside excerpt")
    if candidate["evidence_text"] != candidate["source_block_text"][lo:hi]:
        raise ValueError("excerpt differs from source")


def visible_input(candidate):
    """Whitelist model input; full blocks, historical notes and gold never pass."""
    validate_input(candidate)
    lo, hi = candidate["evidence_char_start"], candidate["evidence_char_end"]
    result = {k: candidate[k] for k in ("target_id", "doc_id", "grain_span", "grain_form",
                                      "coordinate_system", "evidence_char_start", "evidence_char_end", "evidence_text")}
    result["tokens"] = [{"span": t["span"], "text": t["text"]}
                        for t in candidate["token_inventory"] if lo <= t["span"][0] < t["span"][1] <= hi]
    fields = {"option_id", "value", "unit", "currency", "written_value", "multiplier",
              "inference_rule", "ownership_ambiguous", "char_start", "char_end",
              "value_char_start", "value_char_end", "denomination_char_start", "denomination_char_end"}
    for kind in ("qty_options", "money_options"):
        result[kind] = [{k: v for k, v in o.items() if k in fields} for o in candidate[kind]
                        if lo <= o["char_start"] < o["char_end"] <= hi]
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


def validate_answer(answer, candidate):
    if not isinstance(answer, dict) or set(answer) != {"relations", "contexts"}:
        return ["answer must contain relations and contexts"]
    if not isinstance(answer["relations"], list) or not isinstance(answer["contexts"], list):
        return ["relations and contexts must be lists"]
    problems = [e for r in answer["relations"] for e in validate_relation(
        r, candidate, [candidate["evidence_char_start"], candidate["evidence_char_end"]])]
    indexes = []
    for c in answer["contexts"]:
        if not isinstance(c, dict) or set(c) != {"relation_index", "context_tags", "linkage_strength"}:
            problems.append("invalid context fields")
            continue
        index = c["relation_index"]
        if type(index) is not int or not 0 <= index < len(answer["relations"]):
            problems.append("invalid context relation index")
        else:
            indexes.append(index)
        if (not isinstance(c["context_tags"], list) or not c["context_tags"]
                or any(not isinstance(t, str) or t not in CONTEXT_VALUES for t in c["context_tags"])):
            problems.append("invalid context tags")
        if c["linkage_strength"] not in LINKAGE_VALUES:
            problems.append("invalid linkage strength")
    if sorted(indexes) != list(range(len(answer["relations"]))):
        problems.append("one context annotation is required per relation")
    return problems


def experiment_key(candidate, repeat, provider, model, temperature, prompt_hash, execution_settings=None):
    prompt_version = (execution_settings or {}).get("prompt_version", PROMPT_VERSION)
    payload = [OBJECTIVE_VERSION, prompt_version, PREDICTION_VERSION, SCHEMA_SHA256,
               candidate_fingerprint(candidate), repeat, provider, model, temperature, prompt_hash]
    if execution_settings is not None:
        payload.append(execution_settings)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def score_relations(input_path: Path, output_path: Path, prompt_path: Path,
                    provider: str, model: str, temperature=None, repeat=1, resume=False,
                    limit=0, dry_run=False, base_url="", api_key_env="OPENAI_API_KEY",
                    max_validation_retries=2, timeout=180.0, execution_settings=None, before_call=None):
    if not 0 <= max_validation_retries <= 2 or repeat < 1:
        raise ValueError("positive repetitions and at most two validation retries required")
    candidates = list(read_jsonl(input_path))
    if limit:
        candidates = candidates[:limit]
    by_id = {c["target_id"]: c for c in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("duplicate relation targets")
    for c in candidates:
        validate_input(c)
    system = prompt_path.read_text(encoding="utf-8") + "\nOutput JSON schema:\n" + json.dumps(PROVIDER_SCHEMA, sort_keys=True)
    prompt_hash = sha256_file(prompt_path)
    counts = {"candidate_count": len(candidates), "repeat": repeat, "initial_call_limit": len(candidates)*repeat,
              "maximum_call_limit": len(candidates)*repeat*(1+max_validation_retries),
              "input_sha256": sha256_file(input_path), "prompt_sha256": prompt_hash,
              "schema_sha256": SCHEMA_SHA256, "attempted": 0, "valid": 0, "invalid": 0, "errors": 0,
              "provider_calls": 0, "resumed_valid": 0}
    if dry_run:
        return {**counts, "sample_prompt": visible_input(candidates[0]) if candidates else "",
                "system_prompt": system, "schema": PROVIDER_SCHEMA}
    retained = []
    prior_calls = {}
    if output_path.exists():
        if not resume:
            raise ValueError("output exists; use compatible resume or a new path")
        seen = set()
        for row in read_jsonl(output_path):
            c = by_id.get(row.get("target_id"))
            ri = row.get("repeat_index")
            if c is None or type(ri) is not int or not 0 <= ri < repeat:
                raise ValueError("resume contains a target/repetition outside the run")
            validate_candidate_binding(row, c)
            expected = experiment_key(c, ri, provider, model, temperature, prompt_hash, execution_settings)
            if (row.get("experiment_key") != expected or expected in seen
                    or row.get("schema_version") != PREDICTION_VERSION
                    or row.get("objective_version") != OBJECTIVE_VERSION):
                raise ValueError("incompatible or duplicate resume record")
            seen.add(expected)
            used = row.get('provider_calls', row.get('validation_attempts', 1))
            if type(used) is not int or not 0 <= used <= 1+max_validation_retries:
                raise ValueError('incompatible resume call accounting')
            prior_calls[expected] = used
            if row.get("answer") is not None and not row.get("error") and not row.get("validation_problems"):
                check = {**row["answer"], "contexts": row.get("contexts", [])}
                if validate_answer(check, c):
                    raise ValueError("resume record claimed valid but fails contract")
                retained.append(row)
            elif used >= 1+max_validation_retries:
                # An exhausted target stays failed; resume cannot buy it more attempts.
                retained.append(row)
        shutil.copy2(output_path, output_path.with_suffix(".before-resume-"+sha256_file(output_path)[:12]+".jsonl"))
    counts["resumed_valid"] = sum(not r.get('error') and not r.get('validation_problems') for r in retained)
    counts['resumed_exhausted'] = len(retained)-counts['resumed_valid']
    counts['previous_provider_calls'] = sum(prior_calls.values())
    done = {r["experiment_key"] for r in retained}
    extra = {} if execution_settings is None else {"max_output_tokens": execution_settings["max_output_tokens"]}
    if execution_settings and execution_settings.get("gemini_thinking_level"):
        extra["thinking_level"] = execution_settings["gemini_thinking_level"]
    client = build_client(provider, model, temperature, base_url, api_key_env, timeout, **extra)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, retained)
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        for ri in range(repeat):
            for c in candidates:
                key = experiment_key(c, ri, provider, model, temperature, prompt_hash, execution_settings)
                if key in done:
                    continue
                row = {"schema_version": PREDICTION_VERSION, "objective_version": OBJECTIVE_VERSION,
                       "target_id": c["target_id"], "window_id": c["window_id"], "doc_id": c["doc_id"],
                       "candidate_fingerprint": c["candidate_fingerprint"], "repeat_index": ri,
                       "provider": provider, "model_id": model, "temperature": temperature,
                       "prompt_version": (execution_settings or {}).get("prompt_version", PROMPT_VERSION), "prompt_sha256": prompt_hash,
                       "schema_sha256": SCHEMA_SHA256, "experiment_key": key}
                if execution_settings is not None:
                    row["execution_settings"] = execution_settings
                counts["attempted"] += 1
                started = time.perf_counter()
                used = prior_calls.get(key, 0)
                try:
                    for attempt in range(used, max_validation_retries+1):
                        if before_call is not None:
                            before_call(key, system, visible_input(c))
                        counts["provider_calls"] += 1
                        used += 1
                        answer = client.call(system, visible_input(c), PROVIDER_SCHEMA)
                        problems = validate_answer(answer, c)
                        if not problems:
                            break
                    row.update(answer={"relations": answer["relations"]} if isinstance(answer, dict) and isinstance(answer.get("relations"), list) else None,
                               contexts=answer.get("contexts", []) if isinstance(answer, dict) else [],
                               validation_problems=problems, validation_attempts=attempt+1, error=None)
                    counts["invalid" if problems else "valid"] += 1
                except Exception as exc:
                    if isinstance(exc, ProviderError) and exc.fatal:
                        raise ScoringStopped(exc, counts) from exc
                    row.update(answer=None, contexts=[], validation_problems=[], error=str(exc))
                    counts["errors"] += 1
                row["latency_ms"] = round((time.perf_counter()-started)*1000, 1)
                row['provider_calls'] = used
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+"\n")
                handle.flush()
    return counts

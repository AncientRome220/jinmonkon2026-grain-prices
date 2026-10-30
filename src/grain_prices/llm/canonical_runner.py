from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from grain_prices.canonical import DEFINITION_VERSION
from grain_prices.canonical.contracts import CANONICAL_PREDICTION_SCHEMA, validate_prediction
from grain_prices.canonical.io import read_jsonl, sha256_file
from grain_prices.canonical.provenance import candidate_fingerprint, validate_candidate_binding
from grain_prices.llm.clients import ProviderError, build_client
from grain_prices.llm.prompt_loader import load_prompts

SCHEMA_VERSION = "canonical-prediction-v1"
PROMPT_VERSION = "canonical-v13.1-v4-options"
SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(CANONICAL_PREDICTION_SCHEMA, sort_keys=True).encode("utf-8")
).hexdigest()


class ScoringStopped(RuntimeError):
    """An account-level provider failure ended the run; ``counts`` covers the rows written."""

    def __init__(self, cause: ProviderError, counts: dict):
        self.counts = counts
        super().__init__(f"stopped after {counts['attempted']} new windows: {cause}")


def render_options(options: list[dict], kind: str) -> str:
    if not options:
        return "(none)"
    field = "unit" if kind == "quantity" else "currency"
    return "\n".join(
        f"{o['option_id']}: value={o.get('value')} {field}={o.get(field)}, "
        f"written_value={o.get('written_value')}, multiplier={o.get('multiplier', 1)}, "
        f"inference={o.get('inference_rule')}, ambiguous={o.get('ownership_ambiguous', False)}, "
        f"distance={o.get('dist_from_grain')}, "
        f"boundary={o.get('crosses_relation_boundary', o.get('crosses_block_boundary', False))}"
        for o in options
    )


def experiment_key(window_id: str, repeat: int, provider: str, model: str,
                   temperature: float | None = 0.0, prompt_sha256: str = "",
                   candidate_sha256: str = "") -> str:
    raw = "|".join((window_id, str(repeat), provider, model, PROMPT_VERSION,
                    prompt_sha256, SCHEMA_VERSION, SCHEMA_SHA256,
                    DEFINITION_VERSION, str(temperature), candidate_sha256))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_prompt(template: str, candidate: dict) -> str:
    return template.format(
        doc_id=candidate["ddb_id"], window_id=candidate["window_id"],
        grain_form=candidate["grain_form"], evidence_text=candidate["evidence_text"],
        qty_options=render_options(candidate.get("qty_options", []), "quantity"),
        money_options=render_options(candidate.get("money_options", []), "money"),
    )


def score_candidates(input_path: Path, output_path: Path, prompt_path: Path,
                     provider: str, model: str, temperature: float | None = 0.0,
                     repeat: int = 1, resume: bool = False, limit: int = 0,
                     dry_run: bool = False, base_url: str = "",
                     api_key_env: str = "OPENAI_API_KEY",
                     max_validation_retries: int = 2, timeout: float = 180.0) -> dict:
    candidates = list(read_jsonl(input_path))
    if any(c.get("objective_version") == "target-grain-relation-v1" for c in candidates):
        raise ValueError("target-relation candidates need a separately versioned relation prompt/runner; historical score-llm is incompatible")
    if limit:
        candidates = candidates[:limit]
    by_id = {c["window_id"]: c for c in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("duplicate candidate window IDs")
    for candidate in candidates:
        if candidate.get("candidate_fingerprint"):
            validate_candidate_binding(candidate, candidate)
    system, template = load_prompts(prompt_path)
    prompt_sha256 = sha256_file(prompt_path)
    if not system or not template:
        raise ValueError(f"prompt sections missing: {prompt_path}")
    if dry_run:
        return {"candidate_count": len(candidates), "sample_prompt": build_prompt(template, candidates[0]) if candidates else ""}
    client = build_client(provider, model, temperature, base_url, api_key_env, timeout)
    completed_rows: dict[str, dict] = {}
    if resume and output_path.exists():
        for row in read_jsonl(output_path):
            candidate = by_id.get(row.get("window_id"))
            if candidate is None:
                raise ValueError("resume file contains a window outside the input")
            validate_candidate_binding(row, candidate)
            expected = experiment_key(candidate["window_id"], int(row.get("repeat_index", 0)),
                                      provider, model, temperature, prompt_sha256,
                                      candidate_fingerprint(candidate))
            if row.get("experiment_key") != expected:
                raise ValueError("resume experiment differs; use a new output file")
            key = row.get("experiment_key")
            if key and row.get("answer") is not None and not row.get("error"):
                completed_rows[key] = row
    done = set(completed_rows)
    resumed_invalid = sum(
        bool(row.get("validation_problems")) for row in completed_rows.values()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {
        "candidate_count": len(candidates),
        "resumed_valid": len(done) - resumed_invalid,
        "resumed_invalid": resumed_invalid,
        "attempted": 0, "valid": 0, "invalid": 0, "errors": 0,
    }
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in completed_rows.values():
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        for repeat_index in range(repeat):
            for candidate in candidates:
                key = experiment_key(candidate["window_id"], repeat_index, provider, model,
                                     temperature, prompt_sha256, candidate_fingerprint(candidate))
                if key in done:
                    continue
                counts["attempted"] += 1
                started = time.perf_counter()
                row = {
                    "experiment_key": key, "window_id": candidate["window_id"],
                    "doc_id": candidate["ddb_id"], "repeat_index": repeat_index,
                    "provider": provider, "model_id": model,
                    "temperature": temperature, "prompt_version": PROMPT_VERSION,
                    "prompt_sha256": prompt_sha256,
                    "schema_version": SCHEMA_VERSION, "definition_version": DEFINITION_VERSION,
                    "schema_sha256": SCHEMA_SHA256,
                    "candidate_fingerprint": candidate_fingerprint(candidate),
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                }
                try:
                    answer = None
                    problems = []
                    for validation_attempt in range(max_validation_retries + 1):
                        answer = client.call(
                            system, build_prompt(template, candidate),
                            CANONICAL_PREDICTION_SCHEMA,
                        )
                        problems = validate_prediction(answer, candidate)
                        if not problems:
                            break
                    row.update(answer=answer, validation_problems=problems, error=None)
                    row["validation_attempts"] = validation_attempt + 1
                    counts["invalid" if problems else "valid"] += 1
                except Exception as exc:
                    if isinstance(exc, ProviderError) and exc.fatal:
                        # Every remaining window would fail the same way; recording one
                        # error row each hid the cause behind 1,407 copies (#040). The
                        # window is left unwritten, so --resume sends it again.
                        counts["attempted"] -= 1
                        raise ScoringStopped(exc, counts) from exc
                    row.update(answer=None, validation_problems=[], error=str(exc))
                    counts["errors"] += 1
                row["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
    return counts

from __future__ import annotations

import argparse
from importlib.resources import files
import json
import logging
import sys
from pathlib import Path

from grain_prices.canonical.analysis import analyze
from grain_prices.canonical.annotation import (
    merge_relabel_worklist, prepare_gold, prepare_relabel_worklist, sample_canonical,
    validate_annotations,
)
from grain_prices.canonical.candidates import generate_candidates
from grain_prices.canonical.evaluation import (
    evaluate_canonical, evaluate_legacy, evaluate_llm_window,
)
from grain_prices.canonical.io import (
    base_manifest, corpus_fingerprint, make_run_id, sha256_file, write_manifest,
)
from grain_prices.canonical.materialize import materialize
from grain_prices.canonical.relation_windows import prepare_relations, search_windows, freeze_relations
from grain_prices.canonical.relations import evaluate_relations
from grain_prices.canonical.relation_experiment import prepare_experiment, analyze_experiment, prepare_post_llm_review, prepare_full_pass, summarize_checking
from grain_prices.canonical.relation_provisional import prepare_provisional, analyze_provisional, run_provisional
from grain_prices.canonical.relation_reviewed import prepare_reviewed, analyze_reviewed, run_reviewed
from grain_prices.canonical.spine import build_spine
from grain_prices.config import (
    config_value, find_project_root, load_config, load_provider_keys, resolve_corpus_path,
)
from grain_prices.llm.canonical_runner import ScoringStopped, score_candidates
from grain_prices.llm.relation_runner import score_relations
from grain_prices.canonical.relation_materialize import materialize_relations
from grain_prices.snapshot import snapshot_run

ROOT = find_project_root()


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", default="", help="Stable run directory name; generated if omitted")
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--config", type=Path, help="Optional TOML configuration override")


def _artifact(path: Path | None) -> dict | None:
    if path is None:
        return None
    return {"path": str(path.resolve()), "sha256": sha256_file(path)} if path.exists() else {
        "path": str(path.resolve()), "missing": True,
    }


def _temperature(value: str) -> float | None:
    # Current Claude and other reasoning models reject temperature outright (#038).
    return None if value == "default" else float(value)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Canonical v13.1 grain-price pipeline")
    sub = root.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-spine", help="Write lexical hits and filtered candidate mentions")
    _common(p)
    p.add_argument("--ddb-dir", type=Path)
    p.add_argument("--hgv-dir", type=Path)
    p.add_argument("--no-hgv", action="store_true")
    p.add_argument("--hgv-cache", type=Path)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--window-chars", type=int)

    p = sub.add_parser("generate-candidates", help="Enumerate quantity and money options")
    _common(p)
    p.add_argument("--ddb-dir", type=Path)
    p.add_argument("--hgv-dir", type=Path)
    p.add_argument("--no-hgv", action="store_true")
    p.add_argument("--hgv-cache", type=Path)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--window-tokens", type=int)

    for command in ("score-llm", "score-relations"):
        p = sub.add_parser(command, help="Score candidates with a versioned provider contract")
        _common(p)
        p.add_argument("--input", type=Path)
        p.add_argument("--output", type=Path)
        p.add_argument("--provider", choices=["gemini", "openai", "anthropic", "openai-compatible"], required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--base-url", default="")
        p.add_argument("--api-key-env", default="OPENAI_API_KEY")
        p.add_argument("--temperature", type=_temperature, default=0.0,
                       help="a number, or 'default' to leave it unset for models that reject it")
        p.add_argument("--timeout", type=float, default=180.0,
                       help="seconds per request (anthropic, openai, openai-compatible; not gemini)")
        p.add_argument("--repeat", type=int, default=1)
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--resume", action="store_true")
        p.add_argument("--max-validation-retries", type=int, default=2)
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--prompt", type=Path)

    p = sub.add_parser("materialize", help="Create final observations from valid judgments")
    _common(p)
    p.add_argument("--mode", choices=["window", "target_relation"], default="window")
    p.add_argument("--candidates", type=Path)
    source_group = p.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--predictions", type=Path)
    source_group.add_argument("--annotations", type=Path,
                              help="Validated canonical human annotation CSV")
    p.add_argument("--repeat-index", type=int, default=0,
                   help="Which LLM repetition to materialize (default: 0)")
    p.add_argument("--output", type=Path)

    p = sub.add_parser("annotate", help="Prepare or validate canonical human annotation")
    _common(p)
    p.add_argument("action", choices=["sample-canonical", "prepare-gold", "prepare-relabel",
                                      "merge-relabel", "validate"])
    p.add_argument("--gold", type=Path, default=Path("data/annotations/gold_standard_annotation.csv"))
    p.add_argument("--spine", type=Path)
    p.add_argument("--candidates", type=Path)
    p.add_argument("--annotations", type=Path)
    p.add_argument("--legacy-worklist", type=Path, default=Path("data/annotations/gs_relabel_worklist_v13.csv"))
    p.add_argument("--relabel-worklist", type=Path)
    p.add_argument("--size", type=int, default=150, help="sample-canonical: rows to draw")
    p.add_argument("--seed", type=int, default=20260818, help="sample-canonical: RNG seed")
    p.add_argument("--per-doc-cap", type=int, default=2,
                   help="sample-canonical: max windows drawn from one document")
    p.add_argument("--stratify-by", default="linkage_strength",
                   help="sample-canonical: candidate field used for proportional strata")
    p.add_argument("--pass1-only", action="store_true",
                   help="validate: check window-level labels only, not observation fields")

    p = sub.add_parser("evaluate", help="Evaluate frozen legacy or canonical labels")
    _common(p)
    p.add_argument("--mode", choices=["legacy_v12", "canonical_v13_1", "canonical_llm", "target_relation"], required=True)
    p.add_argument("--legacy-report", type=Path, default=Path("outputs/reports/method_comparison_gold_standard.csv"))
    p.add_argument("--legacy-gold", type=Path,
                   default=Path("data/annotations/gold_standard_annotation.csv"))
    p.add_argument("--annotations", type=Path)
    p.add_argument("--candidates", type=Path)
    p.add_argument("--predictions", type=Path,
                   help="canonical_llm: predictions JSONL written by score-llm")
    p.add_argument("--repeat-index", type=int, default=0,
                   help="canonical_llm: which repetition to score (default: 0)")

    p = sub.add_parser("relations", help="Offline target-relation review and window-length search")
    _common(p)
    p.add_argument("action", choices=["prepare", "search", "freeze", "experiment", "analyze", "provisional-experiment", "provisional-analyze", "provisional-run", "reviewed-experiment", "reviewed-analyze", "reviewed-run", "post-llm-sample", "full-pass", "checking-summary"])
    p.add_argument("--radius", type=int, choices=[25, 80])
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--evidence", type=Path)
    p.add_argument("--candidates", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--radii", type=int, nargs="+", default=[10, 15, 20, 25, 30, 40, 50, 60, 80])
    p.add_argument("--preview", action="store_true", help="experiment: prepare non-scoreable inputs before human review")
    p.add_argument("--prompt", type=Path, default=Path(str(files("grain_prices.prompts").joinpath("target_grain_relations_v1.md"))))

    p = sub.add_parser("analyze", help="Export strict, broad and exploratory final series")
    _common(p)
    p.add_argument("--observations", type=Path)

    p = sub.add_parser("snapshot-run", help="Publish a portable, small provenance snapshot")
    p.add_argument("--run-id", required=True)
    p.add_argument("--source-runs-root", type=Path, default=Path("outputs/runs"))
    p.add_argument("--snapshots-root", type=Path, default=Path("runs"))
    return root


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _record_stem(predictions: Path) -> str:
    """``predictions_<provider>_<model>[_tag].jsonl`` -> ``<provider>_<model>[_tag]``."""
    return predictions.stem.removeprefix("predictions_")


def _prepare_runtime(args: argparse.Namespace) -> dict:
    config = load_config(ROOT, getattr(args, "config", None))
    if getattr(args, "runs_root", None) is None:
        args.runs_root = _project_path(config_value(config, "runs", "root") or "outputs/runs")
    if args.command in {"build-spine", "generate-candidates"}:
        args.ddb_dir = resolve_corpus_path(
            args.ddb_dir, "GRAIN_PRICES_DDB_DIR", config, "ddb_dir",
            ("research", "DDB_EpiDoc_XML"),
        )
        args.hgv_dir = None if args.no_hgv else resolve_corpus_path(
            args.hgv_dir, "GRAIN_PRICES_HGV_DIR", config, "hgv_dir",
            ("research", "HGV_meta_EpiDoc"),
        )
        if args.ddb_dir is None:
            raise ValueError(
                "DDbDP corpus path is required: pass --ddb-dir, set "
                "GRAIN_PRICES_DDB_DIR, or configure corpus.ddb_dir"
            )
        if not args.no_hgv and args.hgv_dir is None:
            raise ValueError(
                "HGV corpus path is required unless --no-hgv is used: pass --hgv-dir, "
                "set GRAIN_PRICES_HGV_DIR, or configure corpus.hgv_dir"
            )
        if args.hgv_cache is None:
            args.hgv_cache = _project_path(
                config_value(config, "cache", "hgv_index") or "data/interim/hgv_index_cache_v2.json"
            )
        if args.command == "build-spine" and args.window_chars is None:
            args.window_chars = int(config_value(config, "pipeline", "window_chars") or 200)
        if args.command == "generate-candidates" and args.window_tokens is None:
            args.window_tokens = int(config_value(config, "pipeline", "window_tokens") or 80)
    if args.command in {"score-llm", "score-relations"} and args.prompt is None:
        args.prompt = Path(str(files("grain_prices.prompts").joinpath(
            "target_grain_relations_v1.md" if args.command == "score-relations" else "price_adjudication_canonical_v13_1_options_v2.md"
        )))
    return config


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = parser().parse_args(argv)
    if args.command in {"score-llm", "score-relations"} and not args.dry_run:
        load_provider_keys(ROOT / ".env")  # only the command that calls a provider
    if args.command == "snapshot-run":
        result = snapshot_run(
            args.run_id,
            _project_path(args.source_runs_root),
            _project_path(args.snapshots_root),
            ROOT,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    _prepare_runtime(args)
    run_id = args.run_id or make_run_id()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_dir = args.runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest = base_manifest(ROOT, run_id, args.command, config)
    manifest_suffix = args.command
    # score-llm and evaluate records are named after the predictions file, so a
    # stability rerun or a second model gets its own record instead of
    # overwriting the first (#039).
    if args.command in {"score-llm", "score-relations"}:
        manifest_suffix += "_" + (_record_stem(args.output) if args.output
                                  else f"{args.provider}_{args.model.replace('/', '_')}")
    elif args.command == "evaluate":
        manifest_suffix += f"_{args.mode}"
        if args.mode in {"canonical_llm", "target_relation"} and args.predictions:
            manifest_suffix += "_" + _record_stem(args.predictions)
            if args.repeat_index:
                manifest_suffix += f"_r{args.repeat_index}"
    elif args.command in {"annotate", "relations"}:
        manifest_suffix += f"_{args.action}"
    manifest_path = run_dir / f"manifest_{manifest_suffix}.json"
    write_manifest(manifest_path, manifest)
    try:
        if args.command == "build-spine":
            manifest["corpus_fingerprint"] = corpus_fingerprint(args.ddb_dir, args.limit)
            manifest["metadata_source"] = None if args.no_hgv else {
                "root": str(args.hgv_dir.resolve()), "cache": str(args.hgv_cache.resolve())
            }
            write_manifest(manifest_path, manifest)
            result = build_spine(args.ddb_dir, run_dir, None if args.no_hgv else args.hgv_dir,
                                 args.hgv_cache, args.limit, args.window_chars)
        elif args.command == "generate-candidates":
            manifest["corpus_fingerprint"] = corpus_fingerprint(args.ddb_dir, args.limit)
            manifest["metadata_source"] = None if args.no_hgv else {
                "root": str(args.hgv_dir.resolve()), "cache": str(args.hgv_cache.resolve())
            }
            write_manifest(manifest_path, manifest)
            result = generate_candidates(args.ddb_dir, run_dir / "candidate_options.jsonl",
                                         None if args.no_hgv else args.hgv_dir,
                                         args.hgv_cache, args.limit, args.window_tokens)
        elif args.command in {"score-llm", "score-relations"}:
            source = args.input or run_dir / "candidate_options.jsonl"
            target = args.output or run_dir / f"predictions_{args.provider}_{args.model.replace('/', '_')}.jsonl"
            scorer = score_relations if args.command == "score-relations" else score_candidates
            result = scorer(source, target, args.prompt, args.provider, args.model,
                                      args.temperature, args.repeat, args.resume, args.limit,
                                      args.dry_run, args.base_url, args.api_key_env,
                                      args.max_validation_retries, args.timeout)
            manifest.update(prompt_sha256=sha256_file(args.prompt), model_id=args.model,
                            provider=args.provider, temperature=args.temperature,
                            input_artifact=_artifact(source), output_path=str(target.resolve()))
        elif args.command == "materialize":
            source = args.candidates or run_dir / "candidate_options.jsonl"
            target = args.output or run_dir / "final_observations.jsonl"
            if args.mode == "target_relation":
                if not args.predictions or args.annotations:
                    raise ValueError("relation materialization requires predictions")
                result = materialize_relations(source, args.predictions, target, args.repeat_index)
            else:
                result = materialize(source, args.predictions, target, args.annotations, args.repeat_index)
            manifest.update(candidate_artifact=_artifact(source),
                            judgment_artifact=_artifact(args.predictions or args.annotations),
                            output_path=str(target.resolve()))
        elif args.command == "annotate":
            annotations = args.annotations or run_dir / "gold_canonical.csv"
            if args.action == "sample-canonical":
                candidates = args.candidates or run_dir / "candidate_options.jsonl"
                target = args.annotations or Path("data/annotations/gold_canonical_v13_1.csv")
                subset = run_dir / "gold_sample_candidates.jsonl"
                result = sample_canonical(candidates, target,
                                          run_dir / "manifest_gold_sample.json",
                                          args.size, args.seed, args.per_doc_cap,
                                          args.stratify_by, subset)
                manifest.update(candidate_artifact=_artifact(candidates),
                                output_path=str(target.resolve()), seed=args.seed,
                                per_doc_cap=args.per_doc_cap, stratify_by=args.stratify_by)
            elif args.action == "prepare-gold":
                spine = args.spine or run_dir / "candidate_mentions.jsonl"
                result = prepare_gold(args.gold, spine, annotations,
                                      run_dir / "gold_mapping_worklist.jsonl",
                                      args.legacy_worklist)
                manifest.update(gold_artifact=_artifact(args.gold), spine_artifact=_artifact(spine))
            elif args.action == "prepare-relabel":
                candidates = args.candidates or run_dir / "candidate_options.jsonl"
                result = prepare_relabel_worklist(annotations, args.legacy_worklist,
                                                  candidates, run_dir / "canonical_relabel_worklist.csv")
                manifest.update(annotation_artifact=_artifact(annotations),
                                legacy_worklist_artifact=_artifact(args.legacy_worklist),
                                candidate_artifact=_artifact(candidates))
            elif args.action == "merge-relabel":
                relabel = args.relabel_worklist or run_dir / "canonical_relabel_worklist.csv"
                result = merge_relabel_worklist(annotations, relabel,
                                                run_dir / "gold_canonical_merged.csv")
                manifest.update(annotation_artifact=_artifact(annotations),
                                relabel_artifact=_artifact(relabel))
            else:
                candidates = args.candidates or run_dir / "candidate_options.jsonl"
                errors = validate_annotations(annotations, candidates,
                                              require_observation_fields=not args.pass1_only)
                result = {"valid": not errors, "errors": errors}
                manifest.update(annotation_artifact=_artifact(annotations),
                                candidate_artifact=_artifact(candidates))
        elif args.command == "relations":
            if args.action == "reviewed-experiment":
                if not args.evidence:
                    raise ValueError("--evidence must name source-bound subset decisions and model settings")
                result = prepare_reviewed(args.review, args.evidence, args.output or run_dir / "experiment", args.prompt)
            elif args.action == "reviewed-analyze":
                result = analyze_reviewed(args.review, args.output or run_dir / "comparison")
            elif args.action == "reviewed-run":
                if not args.evidence:
                    raise ValueError("--evidence must name explicit paid-run approval")
                run_reviewed(args.review, args.evidence)
                result = {"status": "execution_finished"}
            elif args.action == "prepare":
                if not args.evidence or not args.candidates:
                    raise ValueError("relations prepare requires --evidence and --candidates")
                result = prepare_relations(args.review, args.evidence, args.candidates,
                                           args.output or run_dir / "review")
            elif args.action == "search":
                result = search_windows(args.review, args.output or run_dir / "windows", args.radii)
            elif args.action == "experiment":
                result = prepare_experiment(args.review, args.output or run_dir / "experiment", args.prompt, args.preview)
            elif args.action == "analyze":
                result = analyze_experiment(args.review, args.output or run_dir / "comparison")
            elif args.action == "provisional-experiment":
                if not args.evidence:
                    raise ValueError("--evidence must name the source-bound three-case audit")
                result = prepare_provisional(args.review, args.output or run_dir / "experiment", args.prompt, args.evidence)
            elif args.action == "provisional-analyze":
                result = analyze_provisional(args.review, args.output or run_dir / "comparison")
            elif args.action == "provisional-run":
                if not args.evidence:
                    raise ValueError("--evidence must name explicit paid-run approval")
                run_provisional(args.review, args.evidence)
                result = {"status": "execution_finished"}
            elif args.action == "post-llm-sample":
                if not args.candidates:
                    raise ValueError("post-llm-sample requires --candidates; --review names predictions")
                result = prepare_post_llm_review(args.candidates, args.review, args.output or run_dir / "post_llm_review")
            elif args.action == "full-pass":
                if args.radius is None:
                    raise ValueError("full-pass requires the chosen --radius; --review names corrected candidates")
                result = prepare_full_pass(args.review, args.output or run_dir / "full_pass", args.radius, args.prompt)
            elif args.action == "checking-summary":
                result = summarize_checking(args.review, args.output or run_dir / "checking_summary.json")
            else:
                result = freeze_relations(args.review, args.output or run_dir / "gold_relations.jsonl")
        elif args.command == "evaluate":
            output = run_dir / f"evaluation_{manifest_suffix.removeprefix('evaluate_')}.json"
            if args.mode == "legacy_v12":
                result = evaluate_legacy(args.legacy_report, output, args.legacy_gold)
                manifest.update(report_artifact=_artifact(args.legacy_report),
                                gold_artifact=_artifact(args.legacy_gold))
            elif args.mode == "target_relation":
                if not all((args.annotations, args.predictions, args.candidates)):
                    raise ValueError("target_relation requires --annotations, --predictions and --candidates")
                result = evaluate_relations(args.predictions, args.annotations, args.candidates,
                                            output, args.repeat_index)
            elif args.mode == "canonical_llm":
                annotations = args.annotations or Path("data/annotations/gold_canonical_v13_1.csv")
                if not args.predictions:
                    raise SystemExit("evaluate --mode canonical_llm needs --predictions")
                extra = {"candidates_path": args.candidates} if args.candidates else {}
                result = evaluate_llm_window(args.predictions, annotations, output,
                                             args.repeat_index, **extra)
                manifest.update(annotation_artifact=_artifact(annotations),
                                prediction_artifact=_artifact(args.predictions),
                                repeat_index=args.repeat_index)
            else:
                annotations = args.annotations or run_dir / "gold_canonical.csv"
                candidates = args.candidates or run_dir / "candidate_options.jsonl"
                result = evaluate_canonical(annotations, output, candidates)
                manifest.update(annotation_artifact=_artifact(annotations),
                                candidate_artifact=_artifact(candidates))
        elif args.command == "analyze":
            observations = args.observations or run_dir / "final_observations.jsonl"
            result = analyze(observations, run_dir / "analysis")
            manifest["observation_artifact"] = _artifact(observations)
        else:
            raise AssertionError(args.command)
        manifest.update(status="complete", result=result)
        write_manifest(manifest_path, manifest)
        print(json.dumps({"run_id": run_id, **result}, ensure_ascii=False, indent=2))
        return 0
    except ScoringStopped as exc:
        # The provider refused the account (#040): say why and keep what was written.
        manifest.update(status="failed", error=str(exc), result=exc.counts)
        write_manifest(manifest_path, manifest)
        print(f"score-llm: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        write_manifest(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

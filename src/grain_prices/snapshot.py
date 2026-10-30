from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from grain_prices.canonical.io import sha256_file


_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_SENSITIVE_KEYS = {"api_key", "access_token", "password", "secret", "token"}


def _portable_path(value: str, project_root: Path) -> str:
    candidate = Path(value)
    if not (candidate.is_absolute() or _WINDOWS_ABSOLUTE.match(value)):
        return value.replace("\\", "/")
    try:
        return candidate.resolve().relative_to(project_root.resolve()).as_posix()
    except (OSError, ValueError):
        return f"<external>/{candidate.name}"


def sanitize(value: Any, project_root: Path, key: str = "") -> Any:
    """Remove secrets and machine-specific absolute paths from snapshot data."""
    if key.lower() in _SENSITIVE_KEYS:
        return "<redacted>"
    if isinstance(value, dict):
        return {name: sanitize(item, project_root, name) for name, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item, project_root, key) for item in value]
    if isinstance(value, str):
        return _portable_path(value, project_root)
    return value


def snapshot_run(
    run_id: str,
    source_runs_root: Path,
    snapshots_root: Path,
    project_root: Path,
) -> dict[str, Any]:
    run_dir = source_runs_root / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    manifest_paths = sorted(run_dir.glob("manifest_*.json"))
    if not manifest_paths:
        raise ValueError(f"No manifests found for run: {run_id}")

    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths]
    incomplete = [m.get("command", "unknown") for m in manifests if m.get("status") != "complete"]
    if incomplete:
        raise ValueError(f"Cannot snapshot incomplete or failed commands: {', '.join(incomplete)}")

    artifacts = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        if path.name.startswith("manifest_"):
            continue
        artifacts.append({
            "logical_path": path.relative_to(run_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    # Keyed by manifest name, not command: a run that evaluates several models has
    # one evaluate manifest per model, and keying by command kept only the last (#039).
    metrics = {
        path.stem.removeprefix("manifest_"): sanitize(manifest.get("result", {}), project_root)
        for path, manifest in zip(manifest_paths, manifests)
        if manifest.get("command") in {"evaluate", "analyze"}
    }
    snapshot = {
        "snapshot_schema_version": 2,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "commands": [sanitize(manifest, project_root) for manifest in manifests],
        "artifacts": artifacts,
        "metrics": metrics,
    }
    destination = snapshots_root / run_id
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "manifest.json"
    # LF on every platform; write_text() writes CRLF on Windows (#026, #039).
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return {"snapshot": str(output), "commands": len(manifests), "artifacts": len(artifacts)}

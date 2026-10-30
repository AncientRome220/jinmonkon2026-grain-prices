from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from grain_prices.canonical import BUILD_VERSION, DEFINITION_VERSION, LEXICON_VERSION, NORM_VERSION


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_fingerprint(root: Path, limit: int = 0) -> dict:
    """Hash the ordered XML inventory without reading the multi-GB corpus twice."""
    files = sorted(root.rglob("*.xml"))
    if limit:
        files = files[:limit]
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        stat = path.stat()
        total_bytes += stat.st_size
        relative = path.relative_to(root).as_posix()
        digest.update(f"{relative}\0{stat.st_size}\0".encode("utf-8"))
        # A complete content hash would add a second full-corpus scan. Instead,
        # hash every small file and deterministic 64 KiB samples from large
        # files; this catches content replacement even when size is unchanged.
        with path.open("rb") as handle:
            if stat.st_size <= 196608:
                digest.update(handle.read())
            else:
                digest.update(handle.read(65536))
                handle.seek(max(0, stat.st_size // 2 - 32768))
                digest.update(handle.read(65536))
                handle.seek(max(0, stat.st_size - 65536))
                digest.update(handle.read(65536))
        digest.update(b"\n")
    return {
        "root": str(root.resolve()),
        "xml_files": len(files),
        "total_bytes": total_bytes,
        "inventory_sha256": digest.hexdigest(),
        "fingerprint_method": "relative_path+size+content_sample_3x64KiB",
    }


def git_sha(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def make_run_id(prefix: str = "canonical") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}"


def base_manifest(root: Path, run_id: str, command: str, config: dict) -> dict:
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    source_files = sorted((root / "src/grain_prices").rglob("*.py")) + sorted((root / "src/grain_prices/prompts").glob("*.md"))
    source_hashes = {p.relative_to(root).as_posix(): sha256_file(p) for p in source_files}
    return {
        "implementation_sha256": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
        "implementation_files": source_hashes,
        "run_id": run_id,
        "command": command,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "definition_version": DEFINITION_VERSION,
        "build_version": BUILD_VERSION,
        "norm_version": NORM_VERSION,
        "lexicon_version": LEXICON_VERSION,
        "git_sha": git_sha(root),
        "config": config,
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "status": "started",
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )

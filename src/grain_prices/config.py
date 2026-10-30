from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from typing import Any


def find_project_root(start: Path | None = None) -> Path:
    """Return the nearest parent containing pyproject.toml, or the current directory."""
    origin = (start or Path.cwd()).resolve()
    for candidate in (origin, *origin.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return origin


def load_provider_keys(path: Path) -> list[str]:
    """Copy ``*_API_KEY`` variables from a dotenv file into the environment.

    Only provider keys are read: corpus paths in the same file would otherwise
    outrank configs/local.toml and silently change which corpus a run reads.
    The file wins over the process environment, so a stale key left in the OS
    environment cannot shadow the one kept in ``.env``; an empty value in the file
    changes nothing. Returns the names loaded, never the values (#038).
    """
    if not path.is_file():
        return []
    loaded = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.removeprefix("export ").split("=", 1)
        name, value = name.strip(), value.strip()
        if value[:1] in {'"', "'"} and value[0] in value[1:]:
            value = value[1:value.index(value[0], 1)]  # quoted: drop anything after
        else:
            value = value.split(" #", 1)[0].split("\t#", 1)[0].strip()  # inline comment
        if name.endswith("_API_KEY") and value:
            os.environ[name] = value
            loaded.append(name)
    return loaded


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_config(project_root: Path, explicit: Path | None = None) -> dict[str, Any]:
    """Load repository defaults, local overrides, then an explicit config file."""
    config = _read_toml(project_root / "configs" / "default.toml")
    config = _merge(config, _read_toml(project_root / "configs" / "local.toml"))
    if explicit is not None:
        path = explicit if explicit.is_absolute() else project_root / explicit
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        config = _merge(config, _read_toml(path))
    return config


def config_value(config: dict[str, Any], section: str, key: str) -> Any:
    value = config.get(section, {})
    return value.get(key) if isinstance(value, dict) else None


def resolve_corpus_path(
    cli_value: Path | None,
    env_name: str,
    config: dict[str, Any],
    config_key: str,
    legacy_parts: tuple[str, ...],
) -> Path | None:
    """Resolve CLI > environment > config > existing legacy location."""
    if cli_value is not None:
        return cli_value
    if os.environ.get(env_name):
        return Path(os.environ[env_name])
    configured = config_value(config, "corpus", config_key)
    if configured:
        return Path(str(configured))
    legacy = Path("C:/")
    for part in legacy_parts:
        legacy /= part
    if legacy.exists():
        print(
            f"warning: using deprecated implicit corpus path {legacy}; "
            f"set {env_name} or pass the CLI option",
            file=sys.stderr,
        )
        return legacy
    return None

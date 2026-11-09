"""Project config file: ``dubber.toml``.

A dubbing project repeats the same flags on every command — target
language, voice library, bed level, enhancement targets. This module
loads those defaults from a TOML file so a project can pin them once:

.. code-block:: toml

   [dubber]
   target_language = "hi"
   voice_library = "voices.json"
   bed_level = 0.3

   [dubber.enhance]
   target = -16.0
   ceiling = -1.0

Explicit CLI flags always win over the file; the file only fills in
what the command line leaves out. Unknown keys and out-of-range
values fail loudly at load time instead of silently doing nothing.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from video_dubber.enhance import (
    DEFAULT_CEILING_DBFS,
    DEFAULT_GATE_THRESHOLD_DBFS,
    DEFAULT_HIGH_PASS_HZ,
    DEFAULT_TARGET_DBFS,
)

__all__ = [
    "ConfigError",
    "DubberConfig",
    "EnhanceConfig",
    "find_config",
    "load_config",
]

_CONFIG_FILE_NAME = "dubber.toml"
_USER_CONFIG_DIR = Path.home() / ".config" / "video-dubber" / _CONFIG_FILE_NAME


class ConfigError(Exception):
    """Raised when a config file is missing, malformed, or invalid."""


@dataclass
class EnhanceConfig:
    """Enhancement defaults (see :mod:`video_dubber.enhance`)."""

    target: float = DEFAULT_TARGET_DBFS
    ceiling: float = DEFAULT_CEILING_DBFS
    high_pass: float = DEFAULT_HIGH_PASS_HZ
    gate: float = DEFAULT_GATE_THRESHOLD_DBFS

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "ceiling": self.ceiling,
            "high_pass": self.high_pass,
            "gate": self.gate,
        }


@dataclass
class DubberConfig:
    """Effective configuration: file values over built-in defaults."""

    target_language: str | None = None
    source_language: str | None = None
    backend: str = "dictionary"
    voice_library: str | None = None
    bed_level: float | None = None
    enhance: EnhanceConfig = field(default_factory=EnhanceConfig)
    path: Path | None = None

    def to_dict(self) -> dict:
        return {
            "target_language": self.target_language,
            "source_language": self.source_language,
            "backend": self.backend,
            "voice_library": self.voice_library,
            "bed_level": self.bed_level,
            "enhance": self.enhance.to_dict(),
            "path": str(self.path) if self.path else None,
        }


def load_config(path: str | Path | None = None) -> DubberConfig:
    """Load the config file, or defaults when no file exists.

    An explicit ``path`` that does not exist is an error (a typo
    should not silently fall back to defaults); a missing file on the
    search path just means defaults. See :func:`find_config` for the
    search order.
    """
    found = find_config(path)
    if path is not None and found is None:
        raise ConfigError(f"config file not found: {path}")
    if found is None:
        return DubberConfig()
    try:
        with open(found, "rb") as fh:
            raw = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {found}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read config file {found}: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) - {"dubber"}:
        raise ConfigError(
            f"{found}: top level must contain only a [dubber] table, "
            f"got keys {sorted(raw)}"
        )
    config = _parse_dubber(raw.get("dubber") or {})
    config.path = found
    return config


def _parse_dubber(table: dict) -> DubberConfig:
    if not isinstance(table, dict):
        raise ConfigError("[dubber] must be a table")
    known = {
        "target_language",
        "source_language",
        "backend",
        "voice_library",
        "bed_level",
        "enhance",
    }
    unknown = sorted(set(table) - known)
    if unknown:
        raise ConfigError(f"unknown [dubber] key(s): {', '.join(unknown)}")
    config = DubberConfig()
    if "target_language" in table:
        config.target_language = _require_str(table, "target_language")
    if "source_language" in table:
        config.source_language = _require_str(table, "source_language")
    if "backend" in table:
        config.backend = _require_str(table, "backend")
        if config.backend not in ("dictionary", "argos"):
            raise ConfigError(
                f"backend must be 'dictionary' or 'argos', got {config.backend!r}"
            )
    if "voice_library" in table:
        config.voice_library = _require_str(table, "voice_library")
    if "bed_level" in table:
        config.bed_level = _require_number(table, "bed_level")
        if not 0.0 <= config.bed_level <= 1.0:
            raise ConfigError(
                f"bed_level must be between 0 and 1, got {config.bed_level}"
            )
    if "enhance" in table:
        config.enhance = _parse_enhance(table["enhance"])
    return config


def _parse_enhance(table: dict) -> EnhanceConfig:
    if not isinstance(table, dict):
        raise ConfigError("[dubber.enhance] must be a table")
    known = {"target", "ceiling", "high_pass", "gate"}
    unknown = sorted(set(table) - known)
    if unknown:
        raise ConfigError(f"unknown [dubber.enhance] key(s): {', '.join(unknown)}")
    enhance = EnhanceConfig()
    for key in known:
        if key in table:
            setattr(enhance, key, _require_number(table, f"enhance.{key}"))
    if enhance.ceiling < enhance.target:
        raise ConfigError(
            "enhance.ceiling "
            f"({enhance.ceiling}) must be >= enhance.target ({enhance.target})"
        )
    return enhance


def _require_str(table: dict, key: str) -> str:
    value = table[key]
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a non-empty string, got {value!r}")
    return value


def _require_number(table: dict, key: str) -> float:
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number, got {value!r}")
    return float(value)


def find_config(explicit: str | Path | None = None) -> Path | None:
    """Locate the config file: explicit path, then search order.

    Search order after an explicit path:

    1. ``./dubber.toml`` (project directory — the usual case),
    2. ``~/.config/video-dubber/dubber.toml`` (user defaults).

    Returns ``None`` when no file exists anywhere on the path.
    """
    if explicit is not None:
        candidate = Path(explicit)
        return candidate if candidate.is_file() else None
    local = Path.cwd() / _CONFIG_FILE_NAME
    if local.is_file():
        return local
    if _USER_CONFIG_DIR.is_file():
        return _USER_CONFIG_DIR
    return None

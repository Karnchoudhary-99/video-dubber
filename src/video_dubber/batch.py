"""Batch dubbing: dub many videos from one manifest file.

A manifest is JSON with optional ``defaults`` (shared job settings) and
a ``jobs`` list. Each job dubs one video (extract → transcribe →
translate → dub) or one pre-made transcript (translate → dub), and
writes its outputs under ``<out-dir>/<job-name>/``. A per-job
``reference`` transcript (ground truth for the source audio) enables a
WER score for that job's transcription.

Relative paths in a manifest are resolved against the manifest's own
directory, so a manifest can be moved with its files. Jobs whose
``dubbed.wav`` already exists are skipped unless ``force`` is set, so a
re-run resumes where the last one stopped instead of redoing the work.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

__all__ = ["BatchError", "BatchJob", "BatchManifest"]

_PATH_FIELDS = ("video", "transcript", "script", "profile", "glossary", "reference")
_JOB_KEYS = frozenset(_PATH_FIELDS) | {
    "name",
    "target",
    "source",
    "backend",
    "max_chars",
    "mux",
    "model",
}
_DEFAULT_KEYS = _JOB_KEYS - {"name"}


class BatchError(Exception):
    """Raised when a batch manifest is invalid or a batch job fails."""


@dataclass(frozen=True)
class BatchJob:
    """One dubbing job: one video or transcript, one target language."""

    name: str
    target: str
    profile: str
    video: str | None = None
    transcript: str | None = None
    script: str | None = None
    source: str | None = None
    backend: str = "whisper"
    model: str = "tiny"
    glossary: str | None = None
    reference: str | None = None
    max_chars: int | None = None
    mux: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise BatchError("job needs a non-empty name")
        if not self.target:
            raise BatchError(f"job {self.name!r} needs a target language")
        if not self.profile:
            raise BatchError(f"job {self.name!r} needs a speaker profile")
        if self.backend not in ("whisper", "scripted"):
            raise BatchError(
                f"job {self.name!r}: unknown backend {self.backend!r} (whisper or scripted)"
            )
        if bool(self.video) == bool(self.transcript):
            raise BatchError(f"job {self.name!r} needs exactly one of 'video' or 'transcript'")
        if self.video and self.backend == "scripted" and not self.script:
            raise BatchError(f"job {self.name!r}: backend scripted needs a 'script' word list")
        if self.max_chars is not None and self.max_chars <= 0:
            raise BatchError(f"job {self.name!r}: max_chars must be positive")


@dataclass
class BatchManifest:
    """A parsed manifest: defaults already merged into every job."""

    jobs: list[BatchJob]
    defaults: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: str, base_dir: str = ".") -> BatchManifest:
        """Parse manifest JSON; relative paths resolve against ``base_dir``."""
        try:
            raw = json.loads(data)
        except json.JSONDecodeError as exc:
            raise BatchError(f"manifest is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise BatchError("manifest must be a JSON object")
        unknown = set(raw) - {"defaults", "jobs"}
        if unknown:
            raise BatchError(f"manifest has unknown top-level keys: {sorted(unknown)}")
        defaults = raw.get("defaults") or {}
        if not isinstance(defaults, dict):
            raise BatchError("'defaults' must be a JSON object")
        bad_keys = set(defaults) - _DEFAULT_KEYS
        if bad_keys:
            raise BatchError(f"defaults has unknown keys: {sorted(bad_keys)}")
        raw_jobs = raw.get("jobs")
        if not isinstance(raw_jobs, list) or not raw_jobs:
            raise BatchError("manifest needs a non-empty 'jobs' list")
        jobs: list[BatchJob] = []
        seen: set[str] = set()
        for entry in raw_jobs:
            if not isinstance(entry, dict):
                raise BatchError("each job must be a JSON object")
            bad_keys = set(entry) - _JOB_KEYS
            if bad_keys:
                raise BatchError(f"job has unknown keys: {sorted(bad_keys)}")
            merged = {**defaults, **entry}
            merged = {
                key: (_resolve(value, base_dir) if key in _PATH_FIELDS else value)
                for key, value in merged.items()
            }
            try:
                job = BatchJob(**merged)
            except TypeError as exc:
                raise BatchError(f"invalid job {entry.get('name', '?')!r}: {exc}") from exc
            if job.name in seen:
                raise BatchError(f"duplicate job name: {job.name!r}")
            seen.add(job.name)
            _check_files(job)
            jobs.append(job)
        return cls(jobs=jobs, defaults=defaults)

    @classmethod
    def from_json_file(cls, path: str) -> BatchManifest:
        """Load a manifest file; paths resolve against its directory."""
        try:
            with open(path, encoding="utf-8") as fh:
                data = fh.read()
        except OSError as exc:
            raise BatchError(f"cannot read manifest {path!r}: {exc}") from exc
        return cls.from_json(data, base_dir=os.path.dirname(os.path.abspath(path)))


def _resolve(value, base_dir: str):
    if isinstance(value, str) and not os.path.isabs(value):
        return os.path.join(base_dir, value)
    return value


def _check_files(job: BatchJob) -> None:
    for key in _PATH_FIELDS:
        path = getattr(job, key)
        if path is not None and not os.path.exists(path):
            raise BatchError(f"job {job.name!r}: {key} file not found: {path}")

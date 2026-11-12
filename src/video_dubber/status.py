"""Project status: what is done, what is stale, what is broken.

A dubbing project accumulates state across commands — ``dubber.toml``,
``manifest.json``, a voice library, and ``out/<job>/`` outputs — and
after a few sessions it is hard to tell at a glance which jobs ran,
which need re-running, and whether the project files themselves are
still valid. :func:`project_status` answers that without running
anything: it loads the config and the manifest (reporting load errors
instead of raising), then grades every job from its outputs.

A job is **not started** when its output directory does not exist,
**partial** when the directory exists but no ``dubbed.wav`` was written
(a batch that died mid-job), **done** when the dub exists, and
**stale** when the dub exists but its source file is newer than the
dub — the source changed since the dub was made, so the dub no longer
matches it. The report verdict and MOS proxy come from the job's
``report.json`` when one exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from video_dubber.assemble import DubReport
from video_dubber.batch import BatchError, BatchManifest
from video_dubber.config import ConfigError, load_config

__all__ = [
    "JobStatus",
    "ProjectStatus",
    "StatusError",
    "project_status",
]

_DUBBED_FILE = "dubbed.wav"
_REPORT_FILE = "report.json"


class StatusError(Exception):
    """Raised when a status cannot be computed (not a project directory)."""


@dataclass
class JobStatus:
    """One manifest job, graded from its outputs on disk."""

    name: str
    target: str
    state: str  # "not started" | "partial" | "done" | "stale"
    verdict: str | None = None
    mos_proxy: float | None = None
    source: str | None = None
    note: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "target": self.target,
            "state": self.state,
            "verdict": self.verdict,
            "mos_proxy": self.mos_proxy,
            "source": self.source,
            "note": self.note,
        }


@dataclass
class ProjectStatus:
    """The whole project at a glance."""

    root: str
    has_config: bool
    config_error: str | None
    target_language: str | None
    has_manifest: bool
    manifest_error: str | None
    has_library: bool
    jobs: list[JobStatus] = field(default_factory=list)

    @property
    def done(self) -> int:
        return sum(1 for job in self.jobs if job.state in ("done", "stale"))

    @property
    def summary(self) -> str:
        total = len(self.jobs)
        states = {
            state: sum(1 for job in self.jobs if job.state == state)
            for state in ("not started", "partial", "done", "stale")
        }
        parts = [f"{count} {state}" for state, count in states.items() if count]
        return f"{self.done}/{total} jobs done" + (f" ({', '.join(parts)})" if parts else "")

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "has_config": self.has_config,
            "config_error": self.config_error,
            "target_language": self.target_language,
            "has_manifest": self.has_manifest,
            "manifest_error": self.manifest_error,
            "has_library": self.has_library,
            "summary": self.summary,
            "jobs": [job.to_dict() for job in self.jobs],
        }

    def to_markdown(self) -> str:
        lines = [f"# Project status: {self.root}", ""]
        if self.has_config:
            target = self.target_language or "(no default target)"
            lines.append(f"- Config: `dubber.toml` (target {target})")
        elif self.config_error:
            lines.append(f"- Config: BROKEN — {self.config_error}")
        else:
            lines.append("- Config: no `dubber.toml` (using built-in defaults)")
        if self.has_manifest:
            lines.append(f"- Manifest: {len(self.jobs)} job(s)")
        elif self.manifest_error:
            lines.append(f"- Manifest: BROKEN — {self.manifest_error}")
        else:
            lines.append("- Manifest: no `manifest.json`")
        lines.append(f"- Voice library: {'present' if self.has_library else 'missing'}")
        lines.append(f"- {self.summary}")
        for job in self.jobs:
            detail = f"{job.name} ({job.target}): {job.state}"
            if job.verdict is not None:
                detail += f", verdict {job.verdict}"
            if job.mos_proxy is not None:
                detail += f", MOS proxy {job.mos_proxy:.2f}"
            if job.note:
                detail += f" — {job.note}"
            lines.append(f"  - {detail}")
        return "\n".join(lines) + "\n"


def _grade_job(job, job_dir: Path) -> JobStatus:
    """Grade one job from its output directory."""
    source = job.video or job.transcript
    dubbed = job_dir / _DUBBED_FILE
    if not dubbed.is_file():
        if not job_dir.is_dir():
            return JobStatus(
                name=job.name,
                target=job.target,
                state="not started",
                source=source,
            )
        return JobStatus(
            name=job.name,
            target=job.target,
            state="partial",
            source=source,
            note="output directory exists but no dubbed.wav was written",
        )
    status = JobStatus(name=job.name, target=job.target, state="done", source=source)
    report_path = job_dir / _REPORT_FILE
    if report_path.is_file():
        try:
            report = DubReport.from_dict(json.loads(report_path.read_text(encoding="utf-8")))
        except (ValueError, json.JSONDecodeError, KeyError) as exc:
            status.note = f"report.json unreadable: {exc}"
        else:
            status.verdict = report.verdict
            status.mos_proxy = report.mos_proxy
    else:
        status.note = "no report.json alongside the dub"
    if source:
        source_path = Path(source)
        if source_path.is_file() and source_path.stat().st_mtime > dubbed.stat().st_mtime:
            status.state = "stale"
            status.note = "source changed since the dub was made" + (
                f"; {status.note}" if status.note else ""
            )
    return status


def project_status(root: str | Path, *, out_dir: str = "out") -> ProjectStatus:
    """Grade a dubbing project directory without running anything."""
    root = Path(root)
    if not root.is_dir():
        raise StatusError(f"not a directory: {root}")

    config_path = root / "dubber.toml"
    config_error: str | None = None
    target_language: str | None = None
    has_config = config_path.is_file()
    has_library = False
    if has_config:
        try:
            config = load_config(config_path)
        except ConfigError as exc:
            config_error = str(exc)
            has_config = False
        else:
            target_language = config.target_language
            if config.voice_library:
                has_library = (root / config.voice_library).is_file()
    if not has_config and (root / "voices.json").is_file():
        has_library = True

    manifest_path = root / "manifest.json"
    manifest_error: str | None = None
    jobs: list[JobStatus] = []
    has_manifest = manifest_path.is_file()
    if has_manifest:
        try:
            manifest = BatchManifest.from_json_file(str(manifest_path))
        except BatchError as exc:
            manifest_error = str(exc)
            has_manifest = False
        else:
            jobs = [_grade_job(job, root / out_dir / job.name) for job in manifest.jobs]
    return ProjectStatus(
        root=str(root),
        has_config=has_config,
        config_error=config_error,
        target_language=target_language,
        has_manifest=has_manifest,
        manifest_error=manifest_error,
        has_library=has_library,
        jobs=jobs,
    )

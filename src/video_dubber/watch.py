"""Watch-folder mode: dub every new video that lands in a directory.

Point ``dubber watch`` at a drop folder and it dubs each new media
file as it appears — extract → transcribe → translate → dub — writing
outputs under ``<out-dir>/<job-name>/`` like a batch job. A JSON state
file records which files were already dubbed (by size + mtime
fingerprint), so a restart never re-dubs finished work, a file edited
in place is re-dubbed once, and a file whose dub failed is retried on
the next scan instead of being marked done.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from video_dubber.batch import BatchJob, BatchJobResult, run_job

__all__ = [
    "MEDIA_EXTENSIONS",
    "WatchConfig",
    "WatchError",
    "WatchScan",
    "WatchState",
    "fingerprint_file",
    "sanitize_job_name",
    "scan_new_files",
    "watch_loop",
    "watch_once",
]

MEDIA_EXTENSIONS = (
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
    ".avi",
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
)
_DEFAULT_STATE_NAME = ".dubber-watch.json"


class WatchError(Exception):
    """Raised when the watch folder or its configuration is invalid."""


@dataclass(frozen=True)
class WatchConfig:
    """One watch folder and the dub settings applied to every file in it."""

    watch_dir: str
    out_dir: str
    target: str
    profile: str
    source: str | None = None
    backend: str = "whisper"
    model: str = "tiny"
    script: str | None = None  # scripted-backend word list (JSON)
    glossary: str | None = None
    max_chars: int | None = None
    mux: bool = True
    profiles: dict[str, str] | None = None  # speaker name -> profile JSON path
    bed_level: float | None = None  # keep the video's music bed under the dub (0-1)
    enhance: bool = False  # enhance the finished dub track (no trimming)
    state_path: str | None = None  # defaults to <out-dir>/.dubber-watch.json
    extensions: tuple[str, ...] = MEDIA_EXTENSIONS

    def __post_init__(self) -> None:
        if not os.path.isdir(self.watch_dir):
            raise WatchError(f"watch directory not found: {self.watch_dir}")
        if not self.target:
            raise WatchError("watch needs a target language")
        if not self.profile:
            raise WatchError("watch needs a speaker profile")
        if not os.path.exists(self.profile):
            raise WatchError(f"speaker profile not found: {self.profile}")
        if self.backend not in ("whisper", "scripted"):
            raise WatchError(f"unknown backend {self.backend!r} (whisper or scripted)")
        if self.backend == "scripted" and not self.script:
            raise WatchError("backend scripted needs a 'script' word list")
        if self.script is not None and not os.path.exists(self.script):
            raise WatchError(f"script file not found: {self.script}")
        if self.glossary is not None and not os.path.exists(self.glossary):
            raise WatchError(f"glossary file not found: {self.glossary}")
        if self.max_chars is not None and self.max_chars <= 0:
            raise WatchError("max_chars must be positive")
        if self.bed_level is not None and not 0.0 <= self.bed_level <= 1.0:
            raise WatchError("'bed_level' must be between 0 and 1")
        if not self.extensions:
            raise WatchError("watch needs at least one file extension")
        if any(not ext.startswith(".") for ext in self.extensions):
            raise WatchError(f"extensions must start with '.': {self.extensions}")
        if self.profiles:
            for speaker, path in self.profiles.items():
                if not os.path.exists(path):
                    raise WatchError(
                        f"profile for speaker {speaker!r} not found: {path}"
                    )

    @property
    def resolved_state_path(self) -> str:
        """Where the processed-files state lives (explicit or the default)."""
        return self.state_path or os.path.join(self.out_dir, _DEFAULT_STATE_NAME)


def fingerprint_file(path: str) -> dict[str, int]:
    """Cheap identity for a file: its size and mtime.

    Hashing whole videos on every scan would cost more than the scan
    itself; size + nanosecond mtime catches edits, renames and
    replacements in practice. A file rewritten with identical size and
    mtime is the one blind spot — re-dub it with ``--force`` via batch.
    """
    stat = os.stat(path)
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


@dataclass
class WatchState:
    """Processed files: relative path -> fingerprint at dub time."""

    files: dict[str, dict[str, int]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> WatchState:
        """Load a state file; a missing or corrupt file starts empty.

        A corrupt state file is not fatal — the worst case is re-dubbing
        files whose outputs already exist (run_job skips those), never
        losing work. The corruption is reported on stderr so it is not
        silently swallowed.
        """
        try:
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except FileNotFoundError:
            return cls()
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"warning: ignoring corrupt watch state {path!r}: {exc}")
            return cls()
        if not isinstance(raw, dict) or not isinstance(raw.get("files"), dict):
            print(f"warning: ignoring corrupt watch state {path!r}: bad shape")
            return cls()
        files = {
            name: fp
            for name, fp in raw["files"].items()
            if isinstance(fp, dict)
            and isinstance(fp.get("size"), int)
            and isinstance(fp.get("mtime_ns"), int)
        }
        return cls(files=files)

    def save(self, path: str) -> None:
        """Write the state file, creating its directory if needed."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"files": self.files}, indent=2, sort_keys=True))

    def is_processed(self, rel_path: str, fingerprint: dict[str, int]) -> bool:
        """True when this exact file version was already dubbed."""
        return self.files.get(rel_path) == fingerprint

    def mark_processed(self, rel_path: str, fingerprint: dict[str, int]) -> None:
        self.files[rel_path] = fingerprint


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def sanitize_job_name(stem: str, taken: set[str]) -> str:
    """Turn a file stem into a safe batch-job name, de-duplicated.

    Lower-cased, runs of non-alphanumerics become one dash, edge dashes
    stripped; an empty result becomes ``clip``. A second file that
    sanitizes to the same name gets ``-2``, ``-3`` … so two files can
    never collide in the same output directory.
    """
    name = _NON_ALNUM.sub("-", stem.lower()).strip("-") or "clip"
    if name not in taken:
        return name
    suffix = 2
    while f"{name}-{suffix}" in taken:
        suffix += 1
    return f"{name}-{suffix}"


def scan_new_files(
    watch_dir: str,
    state: WatchState,
    extensions: tuple[str, ...] = MEDIA_EXTENSIONS,
) -> list[tuple[str, dict[str, int], bool]]:
    """List media files in ``watch_dir`` that still need dubbing.

    Returns ``(path, fingerprint, changed)`` triples sorted by file
    name: brand-new files (``changed=False``) and files whose
    fingerprint differs from the state (``changed=True`` — edited in
    place since their last dub). Dotfiles and non-media files are
    ignored; subdirectories are not scanned.
    """
    wanted = {ext.lower() for ext in extensions}
    found: list[tuple[str, dict[str, int], bool]] = []
    for entry in sorted(os.listdir(watch_dir)):
        if entry.startswith("."):
            continue
        path = os.path.join(watch_dir, entry)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(entry)[1].lower() not in wanted:
            continue
        fingerprint = fingerprint_file(path)
        rel_path = os.path.relpath(path, watch_dir)
        changed = rel_path in state.files and not state.is_processed(
            rel_path, fingerprint
        )
        if state.is_processed(rel_path, fingerprint):
            continue
        found.append((path, fingerprint, changed))
    return found


# --- Running ------------------------------------------------------------------


@dataclass
class WatchScan:
    """Outcome of one scan of the watch folder."""

    results: list[BatchJobResult]
    new_files: int = 0  # files found that needed dubbing (before running)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.results if r.status == "ok")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "failed")

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == "skipped")

    def to_json(self) -> str:
        return json.dumps(
            {
                "new_files": self.new_files,
                "succeeded": self.succeeded,
                "failed": self.failed,
                "skipped": self.skipped,
                "jobs": [
                    {
                        "name": r.name,
                        "status": r.status,
                        "error": r.error,
                        "verdict": r.verdict,
                        "mos_proxy": r.mos_proxy,
                        "duration_s": round(r.duration_s, 2),
                    }
                    for r in self.results
                ],
            },
            indent=2,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        lines = [
            "# Watch scan",
            "",
            f"- New files: {self.new_files} "
            f"({self.succeeded} ok, {self.failed} failed, {self.skipped} skipped)",
            "",
            "| job | status | verdict | MOS proxy |",
            "| --- | --- | --- | --- |",
        ]
        for r in self.results:
            lines.append(
                f"| {r.name} | {r.status} | {r.verdict or '-'} | "
                f"{r.mos_proxy if r.mos_proxy is not None else '-'} |"
            )
        failures = [r for r in self.results if r.status == "failed"]
        if failures:
            lines += ["", "## Failures", ""]
            for r in failures:
                lines.append(f"- **{r.name}**: {r.error}")
        return "\n".join(lines) + "\n"


def _job_for_file(config: WatchConfig, path: str, name: str) -> BatchJob:
    return BatchJob(
        name=name,
        target=config.target,
        profile=config.profile,
        video=path,
        source=config.source,
        backend=config.backend,
        model=config.model,
        script=config.script,
        glossary=config.glossary,
        max_chars=config.max_chars,
        mux=config.mux,
        profiles=config.profiles,
        bed_level=config.bed_level,
        enhance=config.enhance,
    )


def watch_once(
    config: WatchConfig,
    runner: Callable[[BatchJob, str, bool], BatchJobResult] = run_job,
    state: WatchState | None = None,
) -> WatchScan:
    """Scan once and dub every new or changed file.

    Only jobs that finish ``ok`` are marked processed — a failed job is
    retried on the next scan, and a changed file is re-dubbed with
    ``force=True`` so the old output does not make run_job skip it. The
    state file is saved after the scan even when nothing was found, so
    the first run creates it. ``runner`` is injectable so tests can stub
    the pipeline; the default is :func:`video_dubber.batch.run_job`.
    """
    state = state if state is not None else WatchState.load(config.resolved_state_path)
    found = scan_new_files(config.watch_dir, state, config.extensions)
    taken: set[str] = set()
    results: list[BatchJobResult] = []
    for path, fingerprint, changed in found:
        name = sanitize_job_name(os.path.splitext(os.path.basename(path))[0], taken)
        taken.add(name)
        job = _job_for_file(config, path, name)
        result = runner(job, config.out_dir, changed)
        results.append(result)
        if result.status == "ok":
            rel_path = os.path.relpath(path, config.watch_dir)
            state.mark_processed(rel_path, fingerprint)
    state.save(config.resolved_state_path)
    return WatchScan(results=results, new_files=len(found))


def watch_loop(
    config: WatchConfig,
    interval_s: float = 60.0,
    max_scans: int | None = None,
    runner: Callable[[BatchJob, str, bool], BatchJobResult] = run_job,
    sleep: Callable[[float], None] = time.sleep,
    on_scan: Callable[[int, WatchScan], None] | None = None,
) -> list[WatchScan]:
    """Poll the watch folder forever (or for ``max_scans`` scans).

    One scan runs immediately, then the loop sleeps ``interval_s``
    between scans. Ctrl-C stops the loop cleanly and the scans so far
    are returned. ``max_scans`` exists for tests and for one-shot
    scripting; the CLI's ``--once`` is ``max_scans=1``. The state is
    reloaded from disk on every scan, so a second watcher (or a manual
    edit of the state file) is picked up without a restart.
    """
    if interval_s <= 0:
        raise WatchError("watch interval must be positive")
    if max_scans is not None and max_scans <= 0:
        raise WatchError("max_scans must be positive")
    scans: list[WatchScan] = []
    scan_no = 0
    try:
        while max_scans is None or scan_no < max_scans:
            scan_no += 1
            state = WatchState.load(config.resolved_state_path)
            scan = watch_once(config, runner=runner, state=state)
            scans.append(scan)
            if on_scan is not None:
                on_scan(scan_no, scan)
            if max_scans is None or scan_no < max_scans:
                sleep(interval_s)
    except KeyboardInterrupt:
        pass
    return scans

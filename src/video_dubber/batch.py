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
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from video_dubber.assemble import build_report, dub_transcript, mux_audio
from video_dubber.audio import DEFAULT_SAMPLE_RATE, load_wav, resample, save_wav
from video_dubber.enhance import enhance_audio
from video_dubber.extract import extract_audio
from video_dubber.quality import wer_from_transcripts
from video_dubber.transcribe import (
    ScriptedTranscriber,
    WhisperCliTranscriber,
    transcribe_file,
)
from video_dubber.transcript import Transcript
from video_dubber.translate import (
    DictionaryTranslator,
    Glossary,
    translate_transcript,
)
from video_dubber.voice import ScriptedVoiceCloner, SpeakerProfile, VoiceCloner

__all__ = [
    "BatchError",
    "BatchJob",
    "BatchJobResult",
    "BatchManifest",
    "BatchSummary",
    "run_batch",
    "run_job",
    "summarize",
]

_PATH_FIELDS = ("video", "transcript", "script", "profile", "glossary", "reference")
_JOB_KEYS = frozenset(_PATH_FIELDS) | {
    "name",
    "target",
    "source",
    "backend",
    "max_chars",
    "mux",
    "model",
    "profiles",
    "bed_level",
    "enhance",
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
    profiles: dict[str, str] | None = None  # speaker name -> profile JSON path
    bed_level: float | None = None  # keep the video's music bed under the dub (0-1)
    enhance: bool = False  # enhance the finished dub track (no trimming: timing preserved)

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
        if self.profiles is not None and not isinstance(self.profiles, dict):
            raise BatchError(f"job {self.name!r}: 'profiles' must be a speaker->path object")
        if self.bed_level is not None and not 0.0 <= self.bed_level <= 1.0:
            raise BatchError(f"job {self.name!r}: 'bed_level' must be between 0 and 1")
        if self.bed_level is not None and not self.video:
            raise BatchError(
                f"job {self.name!r}: 'bed_level' needs a video job (no original audio)"
            )


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
            if isinstance(merged.get("profiles"), dict):
                merged["profiles"] = {
                    speaker: _resolve(path, base_dir)
                    for speaker, path in merged["profiles"].items()
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
    if job.profiles:
        for speaker, path in job.profiles.items():
            if not os.path.exists(path):
                raise BatchError(
                    f"job {job.name!r}: profile for speaker {speaker!r} not found: {path}"
                )


# --- Running ------------------------------------------------------------------


@dataclass
class BatchJobResult:
    """Outcome of one job: what it produced, how good it is, what failed."""

    name: str
    status: str  # "ok" | "failed" | "skipped"
    error: str | None = None
    outputs: dict[str, str] = field(default_factory=dict)
    transcription_wer: float | None = None  # vs job.reference, when given
    mos_proxy: float | None = None
    verdict: str | None = None
    duration_s: float = 0.0


def _load_transcript(path: str) -> Transcript:
    with open(path, encoding="utf-8") as fh:
        return Transcript.from_json(fh.read())


def _load_profiles(
    paths: dict[str, str] | None,
) -> dict[str, SpeakerProfile] | None:
    """Load a speaker-name -> profile-file map into real profiles."""
    if not paths:
        return None
    profiles: dict[str, SpeakerProfile] = {}
    for speaker, path in paths.items():
        try:
            with open(path, encoding="utf-8") as fh:
                profiles[speaker] = SpeakerProfile.from_json(fh.read())
        except OSError as exc:
            raise BatchError(f"cannot read profile for speaker {speaker!r}: {exc}") from exc
    return profiles


def run_job(
    job: BatchJob,
    out_dir: str,
    cloner: VoiceCloner | None = None,
    force: bool = False,
) -> BatchJobResult:
    """Run one job's full pipeline into ``<out_dir>/<job.name>/``.

    Video jobs: extract → transcribe → translate → dub (+ mux).
    Transcript jobs: translate → dub. Skips the job when its dubbed WAV
    already exists and ``force`` is false. Never raises: failures are
    reported as a ``failed`` result so one bad job does not kill a batch.
    """
    started = time.monotonic()
    job_dir = os.path.join(out_dir, job.name)
    dubbed_wav = os.path.join(job_dir, "dubbed.wav")
    if not force and os.path.exists(dubbed_wav):
        return BatchJobResult(
            name=job.name, status="skipped", duration_s=time.monotonic() - started
        )
    try:
        os.makedirs(job_dir, exist_ok=True)
        with open(job.profile, encoding="utf-8") as fh:
            profile = SpeakerProfile.from_json(fh.read())
        original = None
        if job.video is not None:
            audio_path = extract_audio(job.video, os.path.join(job_dir, "audio.wav"))
            original = resample(load_wav(audio_path), DEFAULT_SAMPLE_RATE)
        if job.transcript is not None:
            transcript = _load_transcript(job.transcript)
        elif job.backend == "scripted":
            transcriber = ScriptedTranscriber.from_json_file(job.script or "")
            transcript = transcribe_file(
                os.path.join(job_dir, "audio.wav"), transcriber, language=job.source
            )
        else:
            transcript = transcribe_file(
                os.path.join(job_dir, "audio.wav"),
                WhisperCliTranscriber(model=job.model),
                language=job.source,
            )
        wer: float | None = None
        if job.reference is not None:
            reference = _load_transcript(job.reference)
            wer = wer_from_transcripts(reference, transcript).wer
        source = job.source or transcript.language or "en"
        translator = DictionaryTranslator(source=source, target=job.target)
        glossary = Glossary.from_json_file(job.glossary) if job.glossary else None
        translated = translate_transcript(
            transcript,
            translator,
            target=job.target,
            source=source,
            glossary=glossary,
            max_chars=job.max_chars,
        )
        dub_audio, fit_factors, placed = dub_transcript(
            translated,
            cloner or ScriptedVoiceCloner(),
            profile,
            original,
            profiles=_load_profiles(job.profiles),
            bed_level=job.bed_level,
        )
        enhancement = None
        if job.enhance:
            # trim=False: the dub sits on a timed bed, so its duration
            # and segment start times must not move.
            dub_audio, enhancement = enhance_audio(dub_audio, trim=False)
        save_wav(dubbed_wav, dub_audio)
        report = build_report(
            translated,
            placed,
            fit_factors,
            dub_audio,
            original,
            translation_coverage=translator.last_coverage,
            bed_level=job.bed_level,
            enhancement=enhancement,
        )
        outputs = {"dubbed_wav": dubbed_wav}
        if job.video is not None and job.mux:
            muxed = os.path.join(job_dir, "dubbed.mp4")
            mux_audio(job.video, dubbed_wav, muxed)
            outputs["dubbed_video"] = muxed
        transcript_path = os.path.join(job_dir, "transcript.json")
        with open(transcript_path, "w", encoding="utf-8") as fh:
            fh.write(transcript.to_json())
        outputs["transcript"] = transcript_path
        translated_path = os.path.join(job_dir, "translated.json")
        with open(translated_path, "w", encoding="utf-8") as fh:
            fh.write(translated.to_json())
        outputs["translated"] = translated_path
        srt_path = os.path.join(job_dir, "translated.srt")
        with open(srt_path, "w", encoding="utf-8") as fh:
            fh.write(translated.to_srt())
        outputs["translated_srt"] = srt_path
        vtt_path = os.path.join(job_dir, "translated.vtt")
        with open(vtt_path, "w", encoding="utf-8") as fh:
            fh.write(translated.to_vtt())
        outputs["translated_vtt"] = vtt_path
        report_path = os.path.join(job_dir, "report.json")
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(report.to_json())
        outputs["report"] = report_path
        return BatchJobResult(
            name=job.name,
            status="ok",
            outputs=outputs,
            transcription_wer=wer,
            mos_proxy=report.mos_proxy,
            verdict=report.verdict,
            duration_s=time.monotonic() - started,
        )
    except Exception as exc:  # noqa: BLE001 — one bad job must not kill the batch
        return BatchJobResult(
            name=job.name,
            status="failed",
            error=str(exc),
            duration_s=time.monotonic() - started,
        )


def run_batch(
    manifest: BatchManifest,
    out_dir: str,
    force: bool = False,
    stop_on_error: bool = False,
    runner: Callable[[BatchJob, str, bool], BatchJobResult] = run_job,
) -> list[BatchJobResult]:
    """Run every job in the manifest, in order.

    ``runner`` is injectable so tests can stub the pipeline; the default
    is :func:`run_job`. With ``stop_on_error`` the batch stops after the
    first failed job instead of continuing with the rest.
    """
    results: list[BatchJobResult] = []
    for job in manifest.jobs:
        result = runner(job, out_dir, force)
        results.append(result)
        if stop_on_error and result.status == "failed":
            break
    return results


# --- Summary report -----------------------------------------------------------


@dataclass
class BatchSummary:
    """Aggregate outcome of a batch run (JSON + Markdown)."""

    results: list[BatchJobResult]

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

    @property
    def mean_wer(self) -> float | None:
        wers = [r.transcription_wer for r in self.results if r.transcription_wer is not None]
        return sum(wers) / len(wers) if wers else None

    @property
    def mean_mos(self) -> float | None:
        scores = [r.mos_proxy for r in self.results if r.mos_proxy is not None]
        return sum(scores) / len(scores) if scores else None

    def to_json(self) -> str:
        return json.dumps(
            {
                "total": self.total,
                "succeeded": self.succeeded,
                "failed": self.failed,
                "skipped": self.skipped,
                "mean_transcription_wer": (
                    None if self.mean_wer is None else round(self.mean_wer, 3)
                ),
                "mean_mos_proxy": (None if self.mean_mos is None else round(self.mean_mos, 2)),
                "jobs": [
                    {
                        "name": r.name,
                        "status": r.status,
                        "error": r.error,
                        "verdict": r.verdict,
                        "mos_proxy": r.mos_proxy,
                        "transcription_wer": r.transcription_wer,
                        "duration_s": round(r.duration_s, 2),
                        "outputs": r.outputs,
                    }
                    for r in self.results
                ],
            },
            indent=2,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        lines = [
            "# Batch dub report",
            "",
            (
                f"- Jobs: {self.total} "
                f"({self.succeeded} ok, {self.failed} failed, {self.skipped} skipped)"
            ),
        ]
        if self.mean_wer is not None:
            lines.append(f"- Mean transcription WER: {self.mean_wer:.3f}")
        if self.mean_mos is not None:
            lines.append(f"- Mean MOS proxy: {self.mean_mos:.2f}")
        lines += [
            "",
            "| job | status | verdict | MOS proxy | WER |",
            "| --- | --- | --- | --- | --- |",
        ]
        for r in self.results:
            lines.append(
                f"| {r.name} | {r.status} | {r.verdict or '-'} | "
                f"{r.mos_proxy if r.mos_proxy is not None else '-'} | "
                f"{r.transcription_wer if r.transcription_wer is not None else '-'} |"
            )
        failures = [r for r in self.results if r.status == "failed"]
        if failures:
            lines += ["", "## Failures", ""]
            for r in failures:
                lines.append(f"- **{r.name}**: {r.error}")
        return "\n".join(lines) + "\n"


def summarize(results: list[BatchJobResult], out_dir: str) -> BatchSummary:
    """Build the batch summary and write batch-report.json/.md into ``out_dir``."""
    summary = BatchSummary(results=results)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "batch-report.json"), "w", encoding="utf-8") as fh:
        fh.write(summary.to_json())
    with open(os.path.join(out_dir, "batch-report.md"), "w", encoding="utf-8") as fh:
        fh.write(summary.to_markdown())
    return summary

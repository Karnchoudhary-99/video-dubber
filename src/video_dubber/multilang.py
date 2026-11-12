"""Multi-target dubbing: one source transcript dubbed into several languages.

A studio rarely ships one language. ``dub_all_targets`` runs the whole
pipeline — translate, synthesise, fit to the original timing windows,
loudness-match, report — once per target language and writes a
per-language bundle (dubbed WAV, quality report, dub-synced subtitles)
plus one summary (JSON + Markdown) comparing the targets.

A target that fails (no dictionary for that pair, synthesis error, …)
is recorded in the summary and the run carries on with the next
language, unless ``stop_on_error`` is set — a half-finished language
should not silently kill the languages that would have worked.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field

from video_dubber.assemble import DubReport, build_report, dub_transcript
from video_dubber.audio import AudioData, save_wav
from video_dubber.enhance import enhance_audio
from video_dubber.style import DubStyle
from video_dubber.subtitles import retime_transcript
from video_dubber.transcript import Transcript
from video_dubber.translate import Glossary, Translator, translate_transcript
from video_dubber.voice import SpeakerProfile, VoiceCloner

__all__ = [
    "MultiTargetError",
    "MultiTargetSummary",
    "TargetResult",
    "dub_all_targets",
    "parse_targets",
]


class MultiTargetError(Exception):
    """Raised when a multi-target run cannot start, or aborts on ``stop_on_error``."""


def parse_targets(raw: str) -> list[str]:
    """Parse a comma-separated ``--targets`` value into deduped language codes.

    Codes are lower-cased and stripped; duplicates (``hi,es,hi``) collapse
    to their first occurrence, order preserved. An empty value raises
    :class:`MultiTargetError`.
    """
    targets: list[str] = []
    for part in raw.split(","):
        code = part.strip().lower()
        if code and code not in targets:
            targets.append(code)
    if not targets:
        raise MultiTargetError("no target languages given: pass --targets hi,es,…")
    return targets


@dataclass(frozen=True)
class TargetResult:
    """One finished language: its quality numbers and the files written."""

    target: str
    dub_path: str
    report_path: str
    subtitle_paths: list[str]
    report: DubReport
    translation_coverage: float | None = None

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "verdict": self.report.verdict,
            "mos_proxy": round(self.report.mos_proxy, 2),
            "segments_dubbed": self.report.segments_dubbed,
            "mean_sync_drift_s": round(self.report.mean_sync_drift, 3),
            "max_sync_drift_s": round(self.report.max_sync_drift, 3),
            "mean_fit_factor": round(self.report.mean_fit_factor, 3),
            "translation_coverage": self.translation_coverage,
            "dub_path": self.dub_path,
            "report_path": self.report_path,
            "subtitle_paths": list(self.subtitle_paths),
        }


@dataclass
class MultiTargetSummary:
    """Aggregate outcome of a multi-target run (JSON + Markdown)."""

    source_language: str
    targets: list[str]  # requested targets, deduped, in order
    results: list[TargetResult] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def succeeded(self) -> int:
        return len(self.results)

    @property
    def failed(self) -> int:
        return len(self.failures)

    @property
    def mean_mos(self) -> float | None:
        scores = [r.report.mos_proxy for r in self.results]
        return sum(scores) / len(scores) if scores else None

    @property
    def verdicts(self) -> dict[str, str]:
        return {r.target: r.report.verdict for r in self.results}

    def to_dict(self) -> dict:
        return {
            "source_language": self.source_language,
            "targets": list(self.targets),
            "succeeded": self.succeeded,
            "failed": self.failed,
            "mean_mos_proxy": (None if self.mean_mos is None else round(self.mean_mos, 2)),
            "results": [r.to_dict() for r in self.results],
            "failures": dict(self.failures),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"

    def to_markdown(self) -> str:
        lines = [
            "# Multi-target dubbing summary",
            "",
            f"Source language: `{self.source_language}` — "
            f"{self.succeeded} of {len(self.targets)} targets dubbed"
            + (f", {self.failed} failed" if self.failed else ""),
            "",
            "| target | verdict | MOS proxy | mean drift | mean fit | coverage |",
            "| ------ | ------- | --------: | ---------: | -------: | -------: |",
        ]
        for result in self.results:
            coverage = (
                f"{result.translation_coverage:.0%}"
                if result.translation_coverage is not None
                else "—"
            )
            lines.append(
                f"| {result.target} | {result.report.verdict} | "
                f"{result.report.mos_proxy:.2f} | "
                f"{result.report.mean_sync_drift:.3f}s | "
                f"{result.report.mean_fit_factor:.2f} | {coverage} |"
            )
        if self.failures:
            lines += ["", "## Failures", ""]
            for target, error in self.failures.items():
                lines.append(f"- **{target}**: {error}")
        lines.append("")
        return "\n".join(lines)


def dub_all_targets(
    transcript: Transcript,
    targets: list[str],
    make_translator: Callable[[str], Translator],
    cloner: VoiceCloner,
    profile: SpeakerProfile,
    out_dir: str,
    stem: str = "dub",
    *,
    source: str | None = None,
    glossary: Glossary | None = None,
    max_chars: int | None = None,
    original: AudioData | None = None,
    profiles: dict[str, SpeakerProfile] | None = None,
    bed_level: float | None = None,
    enhance: bool = False,
    subtitles: bool = True,
    stop_on_error: bool = False,
    style: DubStyle | str | None = None,
) -> MultiTargetSummary:
    """Dub ``transcript`` into every language in ``targets``.

    ``make_translator`` is called once per target (a dictionary
    translator is bound to one language pair; an Argos translator is
    pair-agnostic, so a factory returning the same instance works too).
    Per target the pipeline is: translate (timing windows preserved),
    dub (synthesise + fit + loudness-match, optional music bed),
    optionally enhance, then write ``{stem}.{target}.wav``,
    ``{stem}.{target}.report.json`` and — with ``subtitles`` —
    ``{stem}.{target}.srt`` / ``.vtt`` retimed to the placed dub audio.

    Raises :class:`MultiTargetError` when the target list is empty or
    contains the source language (dubbing into the source language is a
    no-op, not a dub). A per-target failure is recorded in
    ``summary.failures`` and the run continues, unless ``stop_on_error``
    is set, in which case the first failure aborts the run.
    """
    if not targets:
        raise MultiTargetError("no target languages given")
    source_language = source or transcript.language
    if not source_language:
        raise MultiTargetError(
            "source language unknown: pass source= or use a transcript with a language set"
        )
    cleaned: list[str] = []
    for target in targets:
        code = target.strip().lower()
        if not code:
            raise MultiTargetError("empty target language in target list")
        if code == source_language.lower():
            raise MultiTargetError(f"target {code!r} is the source language — nothing to dub")
        if code not in cleaned:
            cleaned.append(code)

    os.makedirs(out_dir, exist_ok=True)
    summary = MultiTargetSummary(source_language=source_language, targets=cleaned)
    for target in cleaned:
        try:
            translator = make_translator(target)
            translated = translate_transcript(
                transcript,
                translator,
                target=target,
                source=source_language,
                glossary=glossary,
                max_chars=max_chars,
            )
            coverage = getattr(translator, "last_coverage", None)
            dub_audio, fit_factors, placed = dub_transcript(
                translated,
                cloner,
                profile,
                original,
                profiles=profiles,
                bed_level=bed_level,
                style=style,
            )
            enhancement = None
            if enhance:
                # trim=False: the dub sits on a timed bed, so its duration
                # and segment start times must not move.
                dub_audio, enhancement = enhance_audio(dub_audio, trim=False)
            dub_path = os.path.join(out_dir, f"{stem}.{target}.wav")
            save_wav(dub_path, dub_audio)
            report = build_report(
                translated,
                placed,
                fit_factors,
                dub_audio,
                original,
                translation_coverage=coverage,
                bed_level=bed_level,
                enhancement=enhancement,
                style=style,
            )
            report_path = os.path.join(out_dir, f"{stem}.{target}.report.json")
            with open(report_path, "w", encoding="utf-8") as fh:
                fh.write(report.to_json())
            subtitle_paths: list[str] = []
            if subtitles:
                synced = retime_transcript(translated, placed)
                for ext, text in (
                    ("srt", synced.to_srt()),
                    ("vtt", synced.to_vtt()),
                ):
                    path = os.path.join(out_dir, f"{stem}.{target}.{ext}")
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(text)
                    subtitle_paths.append(path)
            summary.results.append(
                TargetResult(
                    target=target,
                    dub_path=dub_path,
                    report_path=report_path,
                    subtitle_paths=subtitle_paths,
                    report=report,
                    translation_coverage=coverage,
                )
            )
        except Exception as exc:  # recorded in the summary, run continues
            summary.failures[target] = str(exc)
            if stop_on_error:
                raise MultiTargetError(f"target {target!r} failed: {exc}") from exc
    return summary

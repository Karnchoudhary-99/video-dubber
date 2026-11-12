"""Voice-consistency verification: does the dubbed audio sound like the speaker it claims?

Dub assembly, patching and multi-speaker routing all move audio between
voices — a segment can end up spoken by the wrong voice, or drift so far
from its profile that the listener hears a stranger. This module checks
the finished dub *before* the video is muxed, when a re-dub is still
cheap:

* Each labelled segment is sliced out of the dubbed track and its
  spectral embedding (:func:`video_dubber.voice.extract_embedding`) is
  compared by cosine similarity with the claimed speaker's stored
  profile.
* **match** — similarity at or above ``match_threshold`` (0.90 by
  default, the same bar :meth:`VoiceLibrary.match` uses) *and* no other
  library voice beats the claimed one by more than 0.03: the segment
  sounds like its speaker, and not more like someone else. A segment
  that clears the bar but sits closer to another voice drops to
  ``uncertain`` — near-twin voices are exactly where a routing
  mix-up hides. (Embeddings here are deterministic, so a 0.03 gap is
  a real timbre difference, not measurement noise.)
* **uncertain** — between ``uncertain_threshold`` (0.80) and the match
  bar: plausibly the right voice, but close enough to another voice
  that a human should listen.
* **mismatch** — below the uncertain bar: the segment does not sound
  like its speaker. When another library voice matches the segment
  better, the check names it ("sounds like Ravi (0.97), not Asha"),
  which is usually a routing mix-up rather than a bad synthesis.
* **skipped** — wordless segments, unlabelled segments (no claim to
  check), silent slices and segments shorter than ``min_segment_s``
  (0.5 s: too little audio for a stable embedding) are reported, not
  scored.

A claimed speaker with no profile in the library is a hard error —
verifying a voice nobody recorded is meaningless, and failing loudly
beats silently passing every segment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from video_dubber.audio import AudioData
from video_dubber.transcript import Transcript
from video_dubber.voice import VoiceCloneError, VoiceLibrary, cosine_similarity, extract_embedding

__all__ = [
    "DEFAULT_MATCH_THRESHOLD",
    "DEFAULT_MIN_SEGMENT_S",
    "DEFAULT_UNCERTAIN_THRESHOLD",
    "SegmentVoiceCheck",
    "VerifyConfig",
    "VerifyError",
    "VerifyReport",
    "build_verify_report",
    "verify_dub",
]

DEFAULT_MATCH_THRESHOLD = 0.90  # same bar as VoiceLibrary.match
DEFAULT_UNCERTAIN_THRESHOLD = 0.80  # below this, the voice is not the claimed speaker
DEFAULT_MIN_SEGMENT_S = 0.5  # shorter slices give unstable embeddings


class VerifyError(Exception):
    """Raised when a verification run cannot start (bad config, unknown speaker)."""


@dataclass(frozen=True)
class VerifyConfig:
    """Tuning knobs for :func:`verify_dub`."""

    match_threshold: float = DEFAULT_MATCH_THRESHOLD
    uncertain_threshold: float = DEFAULT_UNCERTAIN_THRESHOLD
    min_segment_s: float = DEFAULT_MIN_SEGMENT_S


@dataclass(frozen=True)
class SegmentVoiceCheck:
    """The verification result for one transcript segment.

    ``similarity`` is the cosine similarity between the segment's
    embedding and the claimed speaker's profile (``None`` when the
    segment was skipped). ``best_match``/``best_similarity`` name the
    library voice the segment sounds *most* like — useful when the
    verdict is ``mismatch`` and the segment was routed to the wrong
    voice. ``verdict`` is ``match``, ``uncertain``, ``mismatch`` or
    ``skipped``.
    """

    index: int  # 1-based segment number in the transcript
    speaker: str | None
    start: float
    end: float
    similarity: float | None
    best_match: str | None
    best_similarity: float | None
    verdict: str
    detail: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "speaker": self.speaker,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "similarity": round(self.similarity, 4) if self.similarity is not None else None,
            "best_match": self.best_match,
            "best_similarity": (
                round(self.best_similarity, 4) if self.best_similarity is not None else None
            ),
            "verdict": self.verdict,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SegmentVoiceCheck:
        """Rebuild from :meth:`to_dict` output."""
        try:
            return cls(
                index=int(data["index"]),
                speaker=data["speaker"],
                start=float(data["start"]),
                end=float(data["end"]),
                similarity=(float(data["similarity"]) if data["similarity"] is not None else None),
                best_match=data["best_match"],
                best_similarity=(
                    float(data["best_similarity"]) if data["best_similarity"] is not None else None
                ),
                verdict=str(data["verdict"]),
                detail=str(data["detail"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"not a valid voice check: {exc}") from exc


def _best_match(embedding: list[float], library: VoiceLibrary) -> tuple[str | None, float | None]:
    """Best-matching library voice for ``embedding`` (name, similarity)."""
    best_name: str | None = None
    best_score: float | None = None
    for name in library.names():
        score = cosine_similarity(embedding, library.get(name).embedding)
        if best_score is None or score > best_score:
            best_name, best_score = name, score
    return best_name, best_score


def verify_dub(
    audio: AudioData,
    transcript: Transcript,
    library: VoiceLibrary,
    config: VerifyConfig | None = None,
) -> list[SegmentVoiceCheck]:
    """Check every labelled segment of a dubbed track against its claimed voice.

    Returns one :class:`SegmentVoiceCheck` per transcript segment, in
    order. Raises :class:`VerifyError` on a bad config, an empty
    transcript, or a claimed speaker with no profile in ``library``.
    """
    config = config or VerifyConfig()
    if not 0.0 < config.uncertain_threshold <= 1.0:
        raise VerifyError(
            f"uncertain_threshold must be in (0, 1], got {config.uncertain_threshold}"
        )
    if not 0.0 < config.match_threshold <= 1.0:
        raise VerifyError(f"match_threshold must be in (0, 1], got {config.match_threshold}")
    if config.match_threshold <= config.uncertain_threshold:
        raise VerifyError(
            "match_threshold must be above uncertain_threshold, got "
            f"{config.match_threshold} <= {config.uncertain_threshold}"
        )
    if config.min_segment_s < 0:
        raise VerifyError(f"min_segment_s must be non-negative, got {config.min_segment_s}")
    if not transcript.segments:
        raise VerifyError("cannot verify a transcript with no segments")
    if len(library) == 0:
        raise VerifyError("cannot verify against an empty voice library")

    checks: list[SegmentVoiceCheck] = []
    for i, segment in enumerate(transcript.segments, start=1):
        start, end = segment.start, segment.end
        base = {"index": i, "speaker": segment.speaker, "start": start, "end": end}

        if not segment.words:
            checks.append(
                SegmentVoiceCheck(
                    **base,
                    similarity=None,
                    best_match=None,
                    best_similarity=None,
                    verdict="skipped",
                    detail="wordless segment has no speech to check",
                )
            )
            continue
        if segment.speaker is None:
            checks.append(
                SegmentVoiceCheck(
                    **base,
                    similarity=None,
                    best_match=None,
                    best_similarity=None,
                    verdict="skipped",
                    detail="no speaker label, so no voice claim to check",
                )
            )
            continue
        if segment.speaker not in library.names():
            raise VerifyError(
                f"segment {i} claims speaker {segment.speaker!r} "
                f"but the library has no such profile (have: {', '.join(library.names())})"
            )
        if end - start < config.min_segment_s:
            checks.append(
                SegmentVoiceCheck(
                    **base,
                    similarity=None,
                    best_match=None,
                    best_similarity=None,
                    verdict="skipped",
                    detail=(
                        f"too short to verify: segment lasts {end - start:.2f}s "
                        f"(min {config.min_segment_s:.2f}s for a stable embedding)"
                    ),
                )
            )
            continue

        try:
            embedding = extract_embedding(audio.slice(start, end))
        except VoiceCloneError:
            checks.append(
                SegmentVoiceCheck(
                    **base,
                    similarity=None,
                    best_match=None,
                    best_similarity=None,
                    verdict="skipped",
                    detail="segment audio is silent (no frame above the noise floor)",
                )
            )
            continue

        profile = library.get(segment.speaker)
        similarity = cosine_similarity(embedding, profile.embedding)
        best_name, best_score = _best_match(embedding, library)
        closer_to_other = (
            best_name is not None
            and best_name != segment.speaker
            and best_score is not None
            and best_score - similarity > 0.03
        )
        if similarity >= config.match_threshold and not closer_to_other:
            verdict, detail = "match", f"sounds like {segment.speaker} ({similarity:.2f})"
        elif closer_to_other and similarity >= config.uncertain_threshold:
            # Downgrades a match, never rescues a mismatch: below the
            # uncertain bar the segment is simply the wrong voice.
            verdict = "uncertain"
            detail = (
                f"closer to {best_name} ({best_score:.2f}) than to "
                f"{segment.speaker} ({similarity:.2f}) — check the routing"
            )
        elif similarity >= config.uncertain_threshold:
            verdict = "uncertain"
            detail = (
                f"only {similarity:.2f} from {segment.speaker} "
                f"(match bar {config.match_threshold:.2f}) — listen before shipping"
            )
        else:
            verdict = "mismatch"
            if best_name is not None and best_name != segment.speaker and best_score is not None:
                detail = (
                    f"sounds like {best_name} ({best_score:.2f}), "
                    f"not {segment.speaker} ({similarity:.2f})"
                )
            else:
                detail = (
                    f"only {similarity:.2f} from {segment.speaker} "
                    f"(uncertain bar {config.uncertain_threshold:.2f})"
                )
        checks.append(
            SegmentVoiceCheck(
                **base,
                similarity=similarity,
                best_match=best_name,
                best_similarity=best_score,
                verdict=verdict,
                detail=detail,
            )
        )
    return checks


@dataclass
class VerifyReport:
    """What voice verification found over a whole dubbed track (JSON + Markdown).

    The verdict is ``pass`` when every scored segment matched its
    claimed speaker — uncertain and mismatch checks both flip it to
    ``review``. Skipped segments never fail the run: silence and
    wordless lines are a property of the source, not a voice error.
    ``mean_similarity`` and ``worst_similarity`` cover scored segments
    only; ``per_speaker`` breaks the same counts down by claimed
    speaker so a single bad voice does not hide in a good average.
    """

    match_threshold: float
    uncertain_threshold: float
    min_segment_s: float
    segments_checked: int
    segments_skipped: int
    matches: int
    uncertain: int
    mismatches: int
    mean_similarity: float | None
    worst_index: int | None  # 1-based index of the lowest-similarity scored check
    worst_similarity: float | None
    verdict: str  # "pass" or "review"
    per_speaker: dict[str, dict] = field(default_factory=dict)
    checks: list[SegmentVoiceCheck] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "match_threshold": self.match_threshold,
                "uncertain_threshold": self.uncertain_threshold,
                "min_segment_s": self.min_segment_s,
                "segments_checked": self.segments_checked,
                "segments_skipped": self.segments_skipped,
                "matches": self.matches,
                "uncertain": self.uncertain,
                "mismatches": self.mismatches,
                "mean_similarity": (
                    round(self.mean_similarity, 4) if self.mean_similarity is not None else None
                ),
                "worst_index": self.worst_index,
                "worst_similarity": (
                    round(self.worst_similarity, 4) if self.worst_similarity is not None else None
                ),
                "verdict": self.verdict,
                "per_speaker": self.per_speaker,
                "checks": [check.to_dict() for check in self.checks],
            },
            indent=2,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        lines = [
            "# Voice verification report",
            "",
            f"- Verdict: **{self.verdict}**",
            f"- Segments checked: {self.segments_checked}",
            f"- Segments skipped: {self.segments_skipped}",
            f"- Matches: {self.matches}, uncertain: {self.uncertain}, mismatches: {self.mismatches}",
            (
                f"- Mean similarity: {self.mean_similarity:.2f}"
                if self.mean_similarity is not None
                else "- Mean similarity: n/a (no scored segments)"
            ),
        ]
        if self.worst_index is not None and self.worst_similarity is not None:
            lines.append(
                f"- Worst segment: {self.worst_index} (similarity {self.worst_similarity:.2f})"
            )
        lines.append(
            f"- Thresholds: match >= {self.match_threshold:.2f}, "
            f"uncertain >= {self.uncertain_threshold:.2f}"
        )
        if self.per_speaker:
            lines.append("")
            lines.append("## Per speaker")
            for name in sorted(self.per_speaker):
                entry = self.per_speaker[name]
                lines.append(
                    f"- {name}: {entry['checked']} checked, "
                    f"{entry['matches']} match / {entry['uncertain']} uncertain / "
                    f"{entry['mismatches']} mismatch, "
                    f"mean {entry['mean_similarity']:.2f}, "
                    f"worst {entry['worst_similarity']:.2f}"
                )
        if self.checks:
            lines.append("")
            lines.append("## Segments")
            for check in self.checks:
                sim = f"{check.similarity:.2f}" if check.similarity is not None else "n/a"
                lines.append(
                    f"- Segment {check.index} ({check.start:.2f}-"
                    f"{check.end:.2f}s) {check.speaker or 'unlabelled'} "
                    f"[{check.verdict}] sim {sim}: {check.detail}"
                )
        return "\n".join(lines) + "\n"

    @classmethod
    def from_dict(cls, data: dict) -> VerifyReport:
        """Rebuild from :meth:`to_json` output."""
        if not isinstance(data, dict):
            raise TypeError("verify report must be a JSON object")
        try:
            return cls(
                match_threshold=float(data["match_threshold"]),
                uncertain_threshold=float(data["uncertain_threshold"]),
                min_segment_s=float(data["min_segment_s"]),
                segments_checked=int(data["segments_checked"]),
                segments_skipped=int(data["segments_skipped"]),
                matches=int(data["matches"]),
                uncertain=int(data["uncertain"]),
                mismatches=int(data["mismatches"]),
                mean_similarity=(
                    float(data["mean_similarity"]) if data["mean_similarity"] is not None else None
                ),
                worst_index=(int(data["worst_index"]) if data["worst_index"] is not None else None),
                worst_similarity=(
                    float(data["worst_similarity"])
                    if data["worst_similarity"] is not None
                    else None
                ),
                verdict=str(data["verdict"]),
                per_speaker={str(k): dict(v) for k, v in data.get("per_speaker", {}).items()},
                checks=[SegmentVoiceCheck.from_dict(row) for row in data.get("checks", [])],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"not a valid verify report: {exc}") from exc

    @classmethod
    def from_json(cls, text: str) -> VerifyReport:
        """Rebuild from :meth:`to_json` output."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"not valid JSON: {exc}") from exc
        try:
            return cls.from_dict(data)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc


def build_verify_report(
    checks: list[SegmentVoiceCheck],
    config: VerifyConfig | None = None,
) -> VerifyReport:
    """Aggregate per-segment :class:`SegmentVoiceCheck` results into a report."""
    config = config or VerifyConfig()
    scored = [c for c in checks if c.similarity is not None]
    skipped = len(checks) - len(scored)
    matches = sum(1 for c in checks if c.verdict == "match")
    uncertain = sum(1 for c in checks if c.verdict == "uncertain")
    mismatches = sum(1 for c in checks if c.verdict == "mismatch")
    mean_similarity = (
        sum(c.similarity for c in scored if c.similarity is not None) / len(scored)
        if scored
        else None
    )
    worst = min(scored, key=lambda c: c.similarity or 0.0) if scored else None

    per_speaker: dict[str, dict] = {}
    for check in scored:
        name = check.speaker or "unknown"
        entry = per_speaker.setdefault(
            name,
            {"checked": 0, "matches": 0, "uncertain": 0, "mismatches": 0, "sims": []},
        )
        entry["checked"] += 1
        entry[check.verdict] += 1
        entry["sims"].append(check.similarity)
    for entry in per_speaker.values():
        sims = entry.pop("sims")
        entry["mean_similarity"] = round(sum(sims) / len(sims), 4)
        entry["worst_similarity"] = round(min(sims), 4)

    return VerifyReport(
        match_threshold=config.match_threshold,
        uncertain_threshold=config.uncertain_threshold,
        min_segment_s=config.min_segment_s,
        segments_checked=len(scored),
        segments_skipped=skipped,
        matches=matches,
        uncertain=uncertain,
        mismatches=mismatches,
        mean_similarity=round(mean_similarity, 4) if mean_similarity is not None else None,
        worst_index=worst.index if worst is not None else None,
        worst_similarity=round(worst.similarity, 4)
        if worst is not None and worst.similarity is not None
        else None,
        verdict="pass" if uncertain == 0 and mismatches == 0 else "review",
        per_speaker=per_speaker,
        checks=list(checks),
    )

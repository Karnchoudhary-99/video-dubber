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

from dataclasses import dataclass

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

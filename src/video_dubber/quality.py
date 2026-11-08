"""Quality metrics for dubs and transcripts.

Two numbers the roadmap's quality report is built on:

* **WER** (:func:`word_error_rate`) — the standard speech-recognition
  metric: how far a transcript is from a ground-truth reference, as
  (substitutions + deletions + insertions) / reference words. Use it to
  score a transcription against a human-checked reference, or a dub
  against the checked text it was supposed to say. It is a same-language
  comparison — comparing an English transcript with a Hindi dub is
  meaningless and will just report ~1.0.
* **MOS proxy** (:func:`mos_proxy`) — a 1–5 stand-in for a listening
  test, derived from the dub's own numbers (sync drift, fit factors,
  loudness error, WER when known). It is a *proxy*: a heuristic that
  ranks dubs and flags bad ones, not a substitute for human ears, and
  the report says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from video_dubber.transcript import Transcript

__all__ = ["WerResult", "mos_proxy", "normalize_words", "wer_from_transcripts", "word_error_rate"]

_WORD_RE = re.compile(r"[^\w']+", re.UNICODE)


def normalize_words(text: str) -> list[str]:
    """Lowercase words with punctuation stripped (apostrophes kept)."""
    return [w for w in (_WORD_RE.sub(" ", text.lower()).split()) if w]


@dataclass(frozen=True)
class WerResult:
    """Word error rate between a reference and a hypothesis."""

    wer: float  # (S + D + I) / reference words; can exceed 1.0
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    hypothesis_words: int


def word_error_rate(reference: list[str], hypothesis: list[str]) -> WerResult:
    """Levenshtein WER over word lists (case-insensitive, punctuation-free).

    Words are compared after :func:`normalize_words`-style cleaning when
    passed as raw text — callers passing lists should normalise first
    (see :func:`wer_from_transcripts`). An empty reference scores 0.0
    against an empty hypothesis and 1.0 against anything else (there is
    nothing to be right about, so any output is fully wrong).
    """
    ref = [w.lower() for w in reference]
    hyp = [w.lower() for w in hypothesis]
    n, m = len(ref), len(hyp)
    if n == 0:
        return WerResult(
            wer=0.0 if m == 0 else 1.0,
            substitutions=0,
            deletions=0,
            insertions=m,
            reference_words=0,
            hypothesis_words=m,
        )
    # Classic DP table: table[i][j] = edits to turn ref[:i] into hyp[:j].
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        table[0][j] = j
    for i in range(n + 1):
        table[i][0] = i
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1, table[i - 1][j - 1] + cost)
    edits = table[n][m]
    # Backtrack to attribute substitutions / deletions / insertions.
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and table[i][j] == table[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1)
        ):
            if ref[i - 1] != hyp[j - 1]:
                subs += 1
            i -= 1
            j -= 1
        elif i > 0 and table[i][j] == table[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    assert subs + dels + ins == edits  # backtrack must agree with the DP cost
    return WerResult(
        wer=edits / n,
        substitutions=subs,
        deletions=dels,
        insertions=ins,
        reference_words=n,
        hypothesis_words=m,
    )


def wer_from_transcripts(reference: Transcript, hypothesis: Transcript) -> WerResult:
    """WER between two transcripts' word streams (order matters)."""
    return word_error_rate(normalize_words(reference.text), normalize_words(hypothesis.text))


def mos_proxy(
    mean_sync_drift: float,
    max_sync_drift: float,
    mean_fit_factor: float,
    extreme_fits: int,
    segments: int,
    loudness_error_db: float | None = None,
    wer: float | None = None,
) -> float:
    """Heuristic 1–5 quality score for a dub (a proxy, not a listening test).

    Starts at 5.0 and loses points for the things listeners actually
    notice: drift out of sync, speech stretched unnaturally, volume
    jumps, and wrong words. Penalties:

    * mean drift above 0.05s: −0.5 per 0.10s over
    * max drift above 0.25s: −1.0 (a single bad segment is audible)
    * each extreme fit (outside 0.8–1.25): −0.25, capped at −1.5
    * loudness error beyond ±3 dB: −0.5 per dB over
    * WER: −2.0 × WER, capped at −2.0
    * mean fit far from 1.0 (rushed/dragged overall): −0.5 per 0.1 off

    Clamped to [1.0, 5.0], rounded to 2 decimals.
    """
    score = 5.0
    if mean_sync_drift > 0.05:
        score -= 0.5 * ((mean_sync_drift - 0.05) / 0.10)
    if max_sync_drift > 0.25:
        score -= 1.0
    if segments > 0 and extreme_fits:
        score -= min(1.5, 0.25 * extreme_fits)
    if abs(mean_fit_factor - 1.0) > 0.1:
        score -= 0.5 * ((abs(mean_fit_factor - 1.0) - 0.1) / 0.1)
    if loudness_error_db is not None and abs(loudness_error_db) > 3.0:
        score -= 0.5 * (abs(loudness_error_db) - 3.0)
    if wer is not None:
        score -= min(2.0, 2.0 * wer)
    return round(max(1.0, min(5.0, score)), 2)

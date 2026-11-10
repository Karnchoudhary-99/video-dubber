"""Demo: the synthesis cache renders each unique line once.

Synthesises the same line twice in the same voice through a
:class:`CachedVoiceCloner` and prints how many times the underlying
cloner actually ran (once — the second render is a cache hit), then a
different line (a genuine miss). A re-dub of an unchanged transcript
therefore costs no synthesis at all, and a patch re-renders only the
segment it touches.

No model, no network, no ffmpeg needed — run it with
``python examples/demo_cache.py``.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.cache import CachedVoiceCloner, SynthesisCache
from video_dubber.voice import ScriptedVoiceCloner, SpeakerProfile


class CountingCloner:
    """Counts real synthesis calls so the demo can show the saving."""

    def __init__(self) -> None:
        self.inner = ScriptedVoiceCloner()
        self.calls = 0

    @property
    def name(self) -> str:
        return self.inner.name

    def clone(self, name, audios, reference_files=None):
        return self.inner.clone(name, audios, reference_files)

    def synthesize(self, text, profile, words_per_minute=None):
        self.calls += 1
        return self.inner.synthesize(text, profile, words_per_minute)


def main() -> None:
    profile = SpeakerProfile(
        name="asha",
        embedding=[0.25] * 16,
        base_pitch_hz=180.0,
        speaking_rate_wpm=150.0,
    )
    with tempfile.TemporaryDirectory() as tmp:
        inner = CountingCloner()
        cloner = CachedVoiceCloner(inner, SynthesisCache(Path(tmp) / "cache"))
        first = cloner.synthesize("hello world, welcome to mars", profile)
        second = cloner.synthesize("hello world, welcome to mars", profile)
        print(f"same line twice: {inner.calls} synthesis call(s) (cache hits: {cloner.hits})")
        # The cache stores 16-bit WAV, so the round-trip differs from the
        # raw floats by at most one quantisation step — the same tolerance
        # the test suite uses.
        same = len(first.samples) == len(second.samples) and all(
            abs(a - b) <= 1 / 32767 for a, b in zip(first.samples, second.samples)
        )
        print(f"identical audio (16-bit round-trip): {same}")
        cloner.synthesize("a different line entirely", profile)
        print(f"new line: {inner.calls} synthesis call(s) (cache misses: {cloner.misses})")
        stats = cloner.cache.stats()
        print(f"cache holds {stats.entries} segment(s), hit rate {stats.hit_rate:.2f}")


if __name__ == "__main__":
    main()

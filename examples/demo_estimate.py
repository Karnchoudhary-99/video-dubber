"""Demo: estimate a dub before running it.

Builds a two-line Hindi transcript — one segment that fits its window
comfortably, one that would need a hard stretch — pre-synthesises the
first line into a cache, then prints the estimate: predicted speech
per segment, fit factors, the cache saving, and the price at a
per-character rate. The estimate's scripted-backend durations are
exact, so the demo also verifies the prediction against a real render
of the second line.

No model, no network, no ffmpeg needed — run it with
``python examples/demo_estimate.py``.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.cache import CachedVoiceCloner, SynthesisCache
from video_dubber.estimate import estimate_dub
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.voice import ScriptedVoiceCloner, SpeakerProfile


def main() -> None:
    profile = SpeakerProfile(
        name="asha",
        embedding=[0.25] * 16,
        base_pitch_hz=180.0,
        speaking_rate_wpm=150.0,
    )
    transcript = Transcript(
        segments=[
            Segment(
                words=[
                    Word("namaste", 0.0, 0.5),
                    Word("dosto", 0.6, 1.0),
                ],
                speaker="A",
            ),
            Segment(
                words=[
                    Word("swagat", 2.0, 2.2),
                    Word("hai", 2.3, 2.4),
                    Word("aapka", 2.5, 2.7),
                    Word("mars", 2.8, 3.0),
                    Word("par", 3.1, 3.2),
                ],
                speaker="A",
            ),
        ],
        language="hi",
        source="clip.wav",
    )
    with tempfile.TemporaryDirectory() as tmp:
        cache = SynthesisCache(Path(tmp) / "cache")
        cloner = CachedVoiceCloner(ScriptedVoiceCloner(), cache)
        # The first line is already dubbed once, so it sits in the cache.
        cloner.synthesize(transcript.segments[0].text, profile)

        estimate = estimate_dub(
            transcript,
            backend="scripted",
            target="hi",
            words_per_minute=profile.speaking_rate_wpm,
            price_per_char=0.002,
            cache=cache,
            profile=profile,
        )
        print(estimate.to_markdown(), end="")
        print(
            f"cache saved {estimate.cached_segments} segment(s): "
            f"{estimate.total_chars - estimate.chars_to_synthesize} chars already rendered"
        )
        # The scripted prediction is exact: verify against a real render.
        rendered = ScriptedVoiceCloner().synthesize(
            transcript.segments[1].text, profile
        )
        print(
            f"predicted {estimate.rows[1].predicted_s:.2f}s for segment 2, "
            f"rendered {rendered.duration:.2f}s"
        )
        assert abs(rendered.duration - estimate.rows[1].predicted_s) < 0.01
        print("estimate verified against a real render")


if __name__ == "__main__":
    main()

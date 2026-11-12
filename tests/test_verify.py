"""Tests for voice-consistency verification of dubbed audio."""

import pytest

from video_dubber.audio import concat, make_sine
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.verify import verify_dub
from video_dubber.voice import VoiceLibrary, clone_from_references


def _library() -> VoiceLibrary:
    """Two voices a sine apart: Asha at 220 Hz, Ravi at 110 Hz."""
    library = VoiceLibrary()
    library.add(clone_from_references("Asha", [make_sine(220.0, 1.0)]))
    library.add(clone_from_references("Ravi", [make_sine(110.0, 1.0)]))
    return library


def _transcript(speakers: list[str | None]) -> Transcript:
    segments = []
    for i, speaker in enumerate(speakers):
        segments.append(
            Segment(
                words=[
                    Word("hello", float(i), float(i) + 0.5),
                    Word("mars", float(i) + 0.5, float(i) + 1.0),
                ],
                speaker=speaker,
            )
        )
    return Transcript(segments=segments, language="en")


class TestScoringBands:
    def test_same_voice_segment_matches(self):
        audio = concat(make_sine(220.0, 1.0), make_sine(110.0, 1.0))
        checks = verify_dub(audio, _transcript(["Asha", "Ravi"]), _library())
        assert [c.verdict for c in checks] == ["match", "match"]
        assert checks[0].similarity == pytest.approx(1.0, abs=0.01)
        assert checks[0].best_match == "Asha"
        assert "Asha" in checks[0].detail

    def test_distant_voice_is_a_mismatch(self):
        # 440 Hz is far from both profiles: claimed Asha, actually neither.
        audio = make_sine(440.0, 1.0)
        checks = verify_dub(audio, _transcript(["Asha"]), _library())
        assert checks[0].verdict == "mismatch"
        assert checks[0].similarity is not None and checks[0].similarity < 0.80

    def test_in_between_voice_is_uncertain(self):
        # 330 Hz sits 0.86 from Asha: close, but below the 0.90 match bar.
        audio = make_sine(330.0, 1.0)
        checks = verify_dub(audio, _transcript(["Asha"]), _library())
        assert checks[0].verdict == "uncertain"
        assert checks[0].similarity == pytest.approx(0.86, abs=0.02)

    def test_segment_closer_to_another_voice_drops_to_uncertain(self):
        # 165 Hz clears the 0.90 bar for Asha (0.92) but sits closer to
        # Ravi (0.97) — a near-twin routing mix-up, not a clean match.
        audio = make_sine(165.0, 1.0)
        checks = verify_dub(audio, _transcript(["Asha"]), _library())
        assert checks[0].verdict == "uncertain"
        assert checks[0].best_match == "Ravi"
        assert "closer to Ravi" in checks[0].detail

    def test_wrong_voice_is_a_mismatch_naming_the_real_voice(self):
        # 440 Hz claimed as Asha: 0.77 from her, 0.99 from Ravi — the
        # segment was routed to the wrong voice, and the check says so.
        audio = make_sine(440.0, 1.0)
        checks = verify_dub(audio, _transcript(["Asha"]), _library())
        assert checks[0].verdict == "mismatch"
        assert checks[0].best_match == "Ravi"
        assert "sounds like Ravi" in checks[0].detail
        assert "not Asha" in checks[0].detail

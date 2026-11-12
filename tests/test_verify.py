"""Tests for voice-consistency verification of dubbed audio."""

import pytest

from video_dubber.audio import concat, make_silence, make_sine
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.verify import (
    SegmentVoiceCheck,
    VerifyConfig,
    VerifyError,
    VerifyReport,
    build_verify_report,
    verify_dub,
)
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


class TestSkippedSegments:
    def test_silent_segment_is_skipped_not_an_error(self):
        audio = concat(make_sine(220.0, 1.0), make_silence(1.0))
        checks = verify_dub(audio, _transcript(["Asha", "Ravi"]), _library())
        assert checks[0].verdict == "match"
        assert checks[1].verdict == "skipped"
        assert checks[1].similarity is None
        assert "silent" in checks[1].detail

    def test_too_short_segment_is_skipped(self):
        transcript = Transcript(
            segments=[
                Segment(
                    words=[Word("hi", 0.0, 0.2)],
                    speaker="Asha",
                )
            ],
            language="en",
        )
        checks = verify_dub(make_sine(220.0, 1.0), transcript, _library())
        assert checks[0].verdict == "skipped"
        assert "too short" in checks[0].detail

    def test_wordless_segment_is_skipped(self):
        transcript = Transcript(
            segments=[
                Segment(words=[], speaker="Asha"),
                Segment(
                    words=[Word("hello", 1.0, 1.5), Word("mars", 1.5, 2.0)],
                    speaker="Asha",
                ),
            ],
            language="en",
        )
        audio = concat(make_silence(1.0), make_sine(220.0, 1.0))
        checks = verify_dub(audio, transcript, _library())
        assert checks[0].verdict == "skipped"
        assert "wordless" in checks[0].detail
        assert checks[1].verdict == "match"

    def test_unlabelled_segment_is_skipped(self):
        checks = verify_dub(make_sine(220.0, 1.0), _transcript([None]), _library())
        assert checks[0].verdict == "skipped"
        assert "no speaker label" in checks[0].detail

    def test_short_threshold_is_configurable(self):
        transcript = Transcript(
            segments=[
                Segment(
                    words=[Word("hi", 0.0, 0.2)],
                    speaker="Asha",
                )
            ],
            language="en",
        )
        checks = verify_dub(
            make_sine(220.0, 1.0),
            transcript,
            _library(),
            config=VerifyConfig(min_segment_s=0.1),
        )
        assert checks[0].verdict == "match"


class TestVerifyReport:
    def _mixed_checks(self):
        audio = concat(
            make_sine(220.0, 1.0),
            make_sine(330.0, 1.0),
            make_sine(440.0, 1.0),
            make_silence(1.0),
        )
        transcript = Transcript(
            segments=[
                Segment(
                    words=[Word("hello", 0.0, 0.5), Word("mars", 0.5, 1.0)],
                    speaker="Asha",
                ),
                Segment(
                    words=[Word("good", 1.0, 1.5), Word("morning", 1.5, 2.0)],
                    speaker="Asha",
                ),
                Segment(
                    words=[Word("and", 2.0, 2.5), Word("you", 2.5, 3.0)],
                    speaker="Asha",
                ),
                Segment(
                    words=[Word("bye", 3.0, 3.5), Word("now", 3.5, 4.0)],
                    speaker="Ravi",
                ),
            ],
            language="en",
        )
        return verify_dub(audio, transcript, _library())

    def test_mixed_report_counts_and_verdict(self):
        report = build_verify_report(self._mixed_checks())
        assert report.segments_checked == 3
        assert report.segments_skipped == 1
        assert report.matches == 1
        assert report.uncertain == 1
        assert report.mismatches == 1
        assert report.verdict == "review"
        assert report.worst_index == 3
        assert report.worst_similarity == pytest.approx(0.77, abs=0.02)
        assert report.mean_similarity == pytest.approx(0.88, abs=0.03)

    def test_per_speaker_breakdown(self):
        report = build_verify_report(self._mixed_checks())
        asha = report.per_speaker["Asha"]
        assert asha["checked"] == 3
        assert asha["matches"] == 1
        assert asha["uncertain"] == 1
        assert asha["mismatches"] == 1
        assert asha["worst_similarity"] == pytest.approx(0.77, abs=0.02)
        # Ravi's only segment is silent, so he has no scored entry.
        assert "Ravi" not in report.per_speaker

    def test_all_skipped_report_still_passes(self):
        # Every segment silent: nothing scored, so nothing can fail —
        # but the report must say so honestly instead of claiming a
        # similarity it never measured.
        audio = make_silence(2.0)
        report = build_verify_report(verify_dub(audio, _transcript(["Asha", "Ravi"]), _library()))
        assert report.verdict == "pass"
        assert report.segments_checked == 0
        assert report.segments_skipped == 2
        assert report.mean_similarity is None
        assert report.worst_index is None
        assert report.per_speaker == {}
        assert "n/a (no scored segments)" in report.to_markdown()

    def test_clean_report_passes(self):
        audio = concat(make_sine(220.0, 1.0), make_sine(110.0, 1.0))
        report = build_verify_report(verify_dub(audio, _transcript(["Asha", "Ravi"]), _library()))
        assert report.verdict == "pass"
        assert report.matches == 2
        assert report.mismatches == 0
        assert report.uncertain == 0

    def test_report_json_round_trip(self):
        report = build_verify_report(self._mixed_checks())
        restored = VerifyReport.from_json(report.to_json())
        assert restored.verdict == report.verdict
        assert restored.matches == report.matches
        assert restored.worst_index == report.worst_index
        assert restored.per_speaker == report.per_speaker
        assert len(restored.checks) == len(report.checks)
        assert restored.checks[2].verdict == "mismatch"
        assert restored.checks[2].best_match == "Ravi"

    def test_report_markdown(self):
        report = build_verify_report(self._mixed_checks())
        text = report.to_markdown()
        assert "Verdict: **review**" in text
        assert "Matches: 1, uncertain: 1, mismatches: 1" in text
        assert "## Per speaker" in text
        assert "Asha: 3 checked" in text
        assert "Segment 3" in text and "[mismatch]" in text
        assert "sounds like Ravi" in text

    def test_check_dict_round_trip(self):
        check = self._mixed_checks()[0]
        restored = SegmentVoiceCheck.from_dict(check.to_dict())
        assert restored == check


class TestValidation:
    def test_match_bar_must_clear_the_uncertain_bar(self):
        with pytest.raises(VerifyError):
            verify_dub(
                make_sine(220.0, 1.0),
                _transcript(["Asha"]),
                _library(),
                config=VerifyConfig(match_threshold=0.80, uncertain_threshold=0.80),
            )

    def test_thresholds_must_be_in_range(self):
        with pytest.raises(VerifyError):
            verify_dub(
                make_sine(220.0, 1.0),
                _transcript(["Asha"]),
                _library(),
                config=VerifyConfig(match_threshold=1.5),
            )
        with pytest.raises(VerifyError):
            verify_dub(
                make_sine(220.0, 1.0),
                _transcript(["Asha"]),
                _library(),
                config=VerifyConfig(uncertain_threshold=0.0),
            )

    def test_empty_transcript_and_empty_library_are_errors(self):
        with pytest.raises(VerifyError):
            verify_dub(make_sine(220.0, 1.0), Transcript(segments=[], language="en"), _library())
        with pytest.raises(VerifyError):
            verify_dub(make_sine(220.0, 1.0), _transcript(["Asha"]), VoiceLibrary())

    def test_claimed_speaker_with_no_profile_is_a_hard_error(self):
        with pytest.raises(VerifyError, match="no such profile"):
            verify_dub(make_sine(220.0, 1.0), _transcript(["Nobody"]), _library())

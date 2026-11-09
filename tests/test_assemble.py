"""Tests for dub assembly: fitting, loudness, placement, quality report."""

import json
import math

import pytest

from video_dubber.assemble import (
    AssemblyError,
    assemble_dub,
    build_report,
    dub_transcript,
    fit_to_window,
    loudness_match,
)
from video_dubber.audio import AudioData, make_silence, make_sine
from video_dubber.bed import BedError
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.voice import EMBEDDING_DIM, ScriptedVoiceCloner, SpeakerProfile


def _words(texts, start=0.0, step=0.5):
    return [
        Word(text=t, start=start + i * step, end=start + (i + 1) * step)
        for i, t in enumerate(texts)
    ]


def _profile() -> SpeakerProfile:
    return SpeakerProfile(name="Test", embedding=[0.25] * EMBEDDING_DIM, base_pitch_hz=180.0)


class TestFitToWindow:
    def test_stretches_to_target_duration(self):
        audio = make_sine(220.0, 1.0)
        fitted, factor = fit_to_window(audio, 2.0)
        # A 2x stretch exceeds MAX_STRETCH, so it clamps at 1.6x.
        assert fitted.duration == pytest.approx(1.6, rel=0.02)
        assert factor == pytest.approx(1.6)
        fitted, factor = fit_to_window(audio, 1.2)
        assert fitted.duration == pytest.approx(1.2, rel=0.02)
        assert factor == pytest.approx(1.2)

    def test_factor_reported_honestly_when_clamped(self):
        audio = make_sine(220.0, 1.0)
        _, factor = fit_to_window(audio, 0.2)
        assert factor == pytest.approx(0.6)  # clamped at MIN_STRETCH

    def test_fitted_clip_is_exactly_the_window_long(self):
        # Regression: resampling rounded the sample count, so a clip
        # stretched to exactly its window overran the dub bed by one
        # sample and assemble_dub rejected the whole dub.
        # (Targets stay inside the 0.6-1.6 clamp range, so no clamping
        # shortfall is in play — the clip must land exactly on target.)
        for target in (1.6, 1.0, 2.5, 1.33):
            fitted, _ = fit_to_window(make_sine(220.0, 1.59), target)
            assert len(fitted.samples) == round(target * 16000)
            assert fitted.duration == pytest.approx(target, abs=1 / 16000)

    def test_empty_audio_is_a_noop(self):
        fitted, factor = fit_to_window(AudioData([], 16000), 1.0)
        assert fitted.samples == []
        assert factor == 1.0

    def test_invalid_target_raises(self):
        with pytest.raises(AssemblyError, match="target duration must be positive"):
            fit_to_window(make_sine(220.0, 0.5), 0)


class TestLoudnessMatch:
    def test_matches_reference_rms(self):
        quiet = make_sine(220.0, 0.5, amplitude=0.1)
        loud = make_sine(440.0, 0.5, amplitude=0.8)
        matched = loudness_match(quiet, loud)
        rms = math.sqrt(sum(s * s for s in matched.samples) / len(matched.samples))
        ref = math.sqrt(sum(s * s for s in loud.samples) / len(loud.samples))
        assert rms == pytest.approx(ref, rel=0.05)

    def test_silence_stays_silent(self):
        matched = loudness_match(make_silence(0.5), make_sine(440.0, 0.5))
        assert matched.samples == [0.0] * len(matched.samples)

    def test_silent_reference_leaves_audio_untouched(self):
        audio = make_sine(220.0, 0.5)
        assert loudness_match(audio, make_silence(0.5)).samples == audio.samples


class TestAssembleDub:
    def test_clips_land_at_their_start_times(self):
        bed = assemble_dub([(0.0, make_sine(440.0, 0.5)), (1.0, make_sine(440.0, 0.5))], 2.0)
        assert bed.duration == pytest.approx(2.0)
        first = bed.samples[0:8000]
        gap = bed.samples[8000:16000]
        second = bed.samples[16000:24000]
        assert max(abs(s) for s in first) > 0.0  # first clip audible from 0s
        assert max(abs(s) for s in gap) == 0.0  # gap between clips stays silent
        assert max(abs(s) for s in second) > 0.0  # second clip audible from 1.0s

    def test_overlapping_clips_mix_without_clipping(self):
        bed = assemble_dub(
            [
                (0.0, make_sine(440.0, 0.5, amplitude=0.9)),
                (0.0, make_sine(440.0, 0.5, amplitude=0.9)),
            ],
            1.0,
        )
        assert bed.peak <= 1.0

    def test_negative_start_raises(self):
        with pytest.raises(AssemblyError, match="starts before 0"):
            assemble_dub([(-0.5, make_sine(440.0, 0.5))], 2.0)

    def test_overrun_raises(self):
        with pytest.raises(AssemblyError, match="overruns"):
            assemble_dub([(1.5, make_sine(440.0, 1.0))], 2.0)

    def test_invalid_total_duration_raises(self):
        with pytest.raises(AssemblyError, match="total duration must be positive"):
            assemble_dub([], 0)


class TestDubTranscript:
    def test_full_dub_spans_transcript_and_reports_fits(self):
        transcript = Transcript(
            segments=[
                Segment(words=_words(["hello", "world"], 0.0)),
                Segment(words=_words(["good", "morning"], 2.0)),
            ],
            language="hi",
        )
        dub, factors, placed = dub_transcript(transcript, ScriptedVoiceCloner(), _profile())
        assert dub.duration == pytest.approx(3.0)
        assert len(factors) == 2
        assert len(placed) == 2
        assert placed[0][0] == 0.0
        assert placed[1][0] == 2.0
        assert dub.peak > 0.0

    def test_empty_transcript_raises(self):
        with pytest.raises(AssemblyError, match="empty transcript"):
            dub_transcript(Transcript(segments=[]), ScriptedVoiceCloner(), _profile())

    def test_empty_segments_are_skipped(self):
        transcript = Transcript(
            segments=[Segment(words=[]), Segment(words=_words(["hello"], 1.0))],
            language="hi",
        )
        _dub, factors, placed = dub_transcript(transcript, ScriptedVoiceCloner(), _profile())
        assert len(factors) == 1
        assert len(placed) == 1


class TestDubBed:
    def _two_segment_transcript(self) -> Transcript:
        return Transcript(
            segments=[
                Segment(words=_words(["hello", "world"], 0.0)),
                Segment(words=_words(["good", "morning"], 2.0)),
            ],
            language="hi",
        )

    def test_bed_audible_in_gaps_silent_without_it(self):
        transcript = self._two_segment_transcript()
        # 97.3 Hz "music" tone across the whole original, so the bed has
        # energy in the gaps between the speech segments.
        original = make_sine(97.3, 3.0, amplitude=0.5)
        gap = int(1.5 * 16_000)
        plain, _, _ = dub_transcript(transcript, ScriptedVoiceCloner(), _profile(), original)
        assert plain.samples[gap] == pytest.approx(0.0)
        dubbed, _, _ = dub_transcript(
            transcript, ScriptedVoiceCloner(), _profile(), original, bed_level=0.3
        )
        assert dubbed.samples[gap] == pytest.approx(original.samples[gap] * 0.3, abs=0.005)

    def test_bed_is_silent_under_dubbed_speech(self):
        # The bed is extracted by muting the original's speech windows,
        # so the original voice can never bleed through under the dub:
        # the with-bed dub is identical to the plain dub under speech.
        # (duck_bed exists for beds supplied from elsewhere, e.g. a
        # music file — see tests/test_bed.py.)
        transcript = self._two_segment_transcript()
        original = make_sine(97.3, 3.0, amplitude=0.5)
        mid_speech = int(0.5 * 16_000)
        plain, _, _ = dub_transcript(transcript, ScriptedVoiceCloner(), _profile(), original)
        dubbed, _, _ = dub_transcript(
            transcript, ScriptedVoiceCloner(), _profile(), original, bed_level=0.3
        )
        assert dubbed.samples[mid_speech] == pytest.approx(plain.samples[mid_speech])

    def test_bed_needs_original_audio(self):
        with pytest.raises(AssemblyError, match="bed_level needs the original"):
            dub_transcript(
                self._two_segment_transcript(),
                ScriptedVoiceCloner(),
                _profile(),
                bed_level=0.3,
            )

    def test_bed_rejects_out_of_range_level(self):
        original = make_sine(97.3, 3.0, amplitude=0.5)
        with pytest.raises(BedError):
            dub_transcript(
                self._two_segment_transcript(),
                ScriptedVoiceCloner(),
                _profile(),
                original,
                bed_level=1.5,
            )


class TestBuildReport:
    def test_good_verdict_for_clean_dub(self):
        transcript = Transcript(segments=[Segment(words=_words(["hello"], 0.0))], language="hi")
        dub, factors, placed = dub_transcript(transcript, ScriptedVoiceCloner(), _profile())
        report = build_report(transcript, placed, factors, dub)
        assert report.verdict == "good"
        assert report.segments_dubbed == 1
        assert report.max_sync_drift < 0.25
        assert "good" in report.to_markdown()
        assert '"verdict": "good"' in report.to_json()

    def test_review_verdict_for_extreme_fit(self):
        transcript = Transcript(segments=[Segment(words=_words(["hello"], 0.0))], language="hi")
        # A fit factor of 1.6 is outside the 0.8-1.25 band.
        dub = make_sine(220.0, 0.5)
        report = build_report(transcript, [(0.0, 0.5)], [1.6], dub)
        assert report.verdict == "review"
        assert report.extreme_fits == 1

    def test_mos_proxy_is_high_for_clean_dub(self):
        transcript = Transcript(segments=[Segment(words=_words(["hello"], 0.0))], language="hi")
        dub, factors, placed = dub_transcript(transcript, ScriptedVoiceCloner(), _profile())
        report = build_report(transcript, placed, factors, dub)
        # Not exactly 5.0: the demo synthesis quantises word timings, so a
        # real dub always carries a few dozen ms of drift.
        assert report.mos_proxy >= 4.5
        assert report.word_error_rate is None
        assert json.loads(report.to_json())["mos_proxy"] == report.mos_proxy
        assert "MOS proxy" in report.to_markdown()

    def test_wer_zero_when_dub_matches_reference(self):
        transcript = Transcript(
            segments=[Segment(words=_words(["hello", "mars"], 0.0))], language="hi"
        )
        dub, factors, placed = dub_transcript(transcript, ScriptedVoiceCloner(), _profile())
        report = build_report(transcript, placed, factors, dub, reference_transcript=transcript)
        assert report.word_error_rate == 0.0
        assert report.verdict == "good"
        assert "Word error rate vs reference: 0.000" in report.to_markdown()

    def test_wer_flags_wrong_words_and_drops_mos(self):
        transcript = Transcript(
            segments=[Segment(words=_words(["hello", "mars"], 0.0))], language="hi"
        )
        reference = Transcript(
            segments=[Segment(words=_words(["hello", "venus"], 0.0))], language="hi"
        )
        dub, factors, placed = dub_transcript(transcript, ScriptedVoiceCloner(), _profile())
        report = build_report(transcript, placed, factors, dub, reference_transcript=reference)
        assert report.word_error_rate == pytest.approx(0.5)
        assert report.verdict == "review"  # WER 0.5 > 0.25 threshold
        assert report.mos_proxy < 5.0

    def test_loudness_error_measured_against_original(self):
        transcript = Transcript(segments=[Segment(words=_words(["hello"], 0.0))], language="hi")
        dub, factors, placed = dub_transcript(
            transcript,
            ScriptedVoiceCloner(),
            _profile(),
            original=make_sine(220.0, 1.0, amplitude=0.1),
        )
        report = build_report(
            transcript, placed, factors, dub, make_sine(220.0, 1.0, amplitude=0.1)
        )
        assert report.loudness_error_db is not None
        assert abs(report.loudness_error_db) < 3.0


class _RecordingCloner:
    """VoiceCloner double that records which profile each segment used."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return "recording"

    def clone(self, name, audios, reference_files=None):
        raise NotImplementedError

    def synthesize(self, text, profile, words_per_minute=None):
        self.calls.append((text, profile.name))
        return make_sine(profile.base_pitch_hz, 0.5)


def _two_speaker_transcript() -> Transcript:
    return Transcript(
        segments=[
            Segment(words=_words(["hello", "there"], start=0.0), speaker="Asha"),
            Segment(words=_words(["good", "morning"], start=2.0), speaker="Ravi"),
            Segment(words=_words(["and", "welcome"], start=4.0)),
        ],
        language="hi",
    )


class TestMultiSpeakerDub:
    def test_each_speaker_gets_their_own_voice(self):
        asha = SpeakerProfile(name="Asha", embedding=[0.25] * EMBEDDING_DIM, base_pitch_hz=220.0)
        ravi = SpeakerProfile(name="Ravi", embedding=[0.1] * EMBEDDING_DIM, base_pitch_hz=110.0)
        default = SpeakerProfile(
            name="Narrator", embedding=[0.2] * EMBEDDING_DIM, base_pitch_hz=150.0
        )
        cloner = _RecordingCloner()
        audio, _, _ = dub_transcript(
            _two_speaker_transcript(),
            cloner,
            default,
            profiles={"Asha": asha, "Ravi": ravi},
        )
        assert [name for _, name in cloner.calls] == ["Asha", "Ravi", "Narrator"]
        assert audio.duration == pytest.approx(5.0)

    def test_unknown_speaker_falls_back_to_default_voice(self):
        default = _profile()
        cloner = _RecordingCloner()
        transcript = Transcript(
            segments=[Segment(words=_words(["mystery", "voice"], start=0.0), speaker="Ghost")],
            language="hi",
        )
        dub_transcript(transcript, cloner, default, profiles={})
        assert cloner.calls == [("mystery voice", "Test")]

    def test_report_counts_speakers_dubbed(self):
        transcript = _two_speaker_transcript()
        report = build_report(
            transcript, [(0.0, 1.0), (2.0, 1.0), (4.0, 1.0)], [1.0] * 3, make_sine(220.0, 5.0)
        )
        assert report.speakers_dubbed == 2
        plain = Transcript(segments=[Segment(words=_words(["solo"], start=0.0))])
        report = build_report(plain, [(0.0, 0.5)], [1.0], make_sine(220.0, 0.5))
        assert report.speakers_dubbed == 0

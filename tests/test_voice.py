"""Tests for voice cloning: profiles, embeddings, synthesis, prosody."""

import math

import pytest

from video_dubber.audio import AudioData, concat, make_silence, make_sine
from video_dubber.voice import (
    EMBEDDING_DIM,
    ScriptedVoiceCloner,
    SpeakerProfile,
    VoiceCloneError,
    adjust_prosody,
    clone_from_references,
    cosine_similarity,
    extract_embedding,
    extract_features,
    synthesize_text,
    time_stretch,
)


def _voice(freq: float, duration: float = 1.0) -> AudioData:
    """A steady tone standing in for a voiced reference recording."""
    return make_sine(freq, duration)


class TestSpeakerProfile:
    def test_json_round_trip(self):
        profile = SpeakerProfile(
            name="Asha",
            embedding=[0.25] * EMBEDDING_DIM,
            base_pitch_hz=220.0,
            speaking_rate_wpm=160.0,
            reference_files=["ref1.wav"],
        )
        again = SpeakerProfile.from_json(profile.to_json())
        assert again.name == "Asha"
        assert again.embedding == [0.25] * EMBEDDING_DIM
        assert again.base_pitch_hz == 220.0
        assert again.speaking_rate_wpm == 160.0
        assert again.reference_files == ["ref1.wav"]

    def test_wrong_embedding_dim_rejected(self):
        with pytest.raises(VoiceCloneError, match="embedding must be 16 floats"):
            SpeakerProfile(name="x", embedding=[0.1] * 8)

    def test_invalid_pitch_and_rate_rejected(self):
        with pytest.raises(VoiceCloneError, match="base_pitch_hz must be positive"):
            SpeakerProfile(name="x", embedding=[0.25] * EMBEDDING_DIM, base_pitch_hz=0)
        with pytest.raises(VoiceCloneError, match="speaking_rate_wpm must be positive"):
            SpeakerProfile(
                name="x", embedding=[0.25] * EMBEDDING_DIM, speaking_rate_wpm=-5
            )

    def test_bad_profile_json_raises(self):
        with pytest.raises(VoiceCloneError, match="invalid speaker profile JSON"):
            SpeakerProfile.from_json("{nope")
        with pytest.raises(VoiceCloneError, match="invalid speaker profile JSON"):
            SpeakerProfile.from_json('{"name": "x"}')


class TestFeaturesAndEmbedding:
    def test_embedding_is_unit_length_16_dim(self):
        emb = extract_embedding(_voice(220.0, 1.0))
        assert len(emb) == EMBEDDING_DIM
        assert math.sqrt(sum(v * v for v in emb)) == pytest.approx(1.0)

    def test_embedding_is_deterministic(self):
        a = extract_embedding(_voice(220.0, 1.0))
        b = extract_embedding(_voice(220.0, 1.0))
        assert a == b

    def test_different_pitches_give_different_embeddings(self):
        low = extract_embedding(_voice(110.0, 1.0))
        high = extract_embedding(_voice(330.0, 1.0))
        assert cosine_similarity(low, high) < cosine_similarity(low, low)
        assert cosine_similarity(low, high) < 0.99

    def test_silence_raises(self):
        with pytest.raises(VoiceCloneError, match="silent"):
            extract_embedding(make_silence(1.0))

    def test_empty_audio_raises(self):
        with pytest.raises(VoiceCloneError):
            extract_embedding(AudioData([], 16000))

    def test_features_report_voiced_ratio(self):
        feats = extract_features(concat(_voice(220.0, 0.5), make_silence(0.5)))
        assert 0.3 < feats["voiced_ratio"] < 0.9
        assert feats["pitch"] == pytest.approx(220.0, rel=0.1)


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        v = [0.5] * EMBEDDING_DIM
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_opposite_vectors_score_minus_one(self):
        a = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
        b = [-1.0] + [0.0] * (EMBEDDING_DIM - 1)
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_mismatched_dims_raise(self):
        with pytest.raises(VoiceCloneError, match="cannot compare embeddings"):
            cosine_similarity([0.1] * 4, [0.1] * 8)


class TestCloneFromReferences:
    def test_profile_averages_references(self):
        profile = clone_from_references(
            "Duo", [_voice(220.0, 0.5), _voice(220.0, 0.5)], ["a.wav", "b.wav"]
        )
        assert profile.name == "Duo"
        assert profile.reference_files == ["a.wav", "b.wav"]
        assert profile.base_pitch_hz == pytest.approx(220.0, rel=0.1)
        assert math.sqrt(sum(v * v for v in profile.embedding)) == pytest.approx(1.0)

    def test_same_voice_matches_itself_better_than_another(self):
        a = clone_from_references("Low", [_voice(110.0, 1.0)])
        b = clone_from_references("High", [_voice(330.0, 1.0)])
        probe = extract_embedding(_voice(110.0, 1.0))
        assert cosine_similarity(probe, a.embedding) > cosine_similarity(probe, b.embedding)

    def test_no_references_raises(self):
        with pytest.raises(VoiceCloneError, match="at least one reference"):
            clone_from_references("Nobody", [])


def _profile(pitch: float = 180.0, wpm: float = 150.0) -> SpeakerProfile:
    return SpeakerProfile(
        name="Test", embedding=[0.25] * EMBEDDING_DIM, base_pitch_hz=pitch,
        speaking_rate_wpm=wpm,
    )


class TestSynthesis:
    def test_duration_follows_word_count_and_rate(self):
        fast = synthesize_text("one two three four", _profile(), words_per_minute=300)
        slow = synthesize_text("one two three four", _profile(), words_per_minute=150)
        assert slow.duration > fast.duration
        assert slow.duration == pytest.approx(2 * fast.duration, rel=0.15)

    def test_longer_words_take_longer(self):
        short = synthesize_text("a", _profile())
        long = synthesize_text("extraordinary", _profile())
        assert long.duration > short.duration

    def test_synthesis_is_audible_and_bounded(self):
        audio = synthesize_text("hello world", _profile())
        assert audio.peak > 0.05
        assert audio.peak <= 1.0
        assert audio.sample_rate == 16000

    def test_empty_text_returns_silence(self):
        assert synthesize_text("   ", _profile()).samples == []

    def test_different_pitch_changes_waveform(self):
        low = synthesize_text("hello", _profile(pitch=110.0))
        high = synthesize_text("hello", _profile(pitch=330.0))
        assert low.samples != high.samples

    def test_same_word_same_voice_is_deterministic(self):
        a = synthesize_text("hello world", _profile())
        b = synthesize_text("hello world", _profile())
        assert a.samples == b.samples

    def test_invalid_wpm_raises(self):
        with pytest.raises(VoiceCloneError, match="words_per_minute must be positive"):
            synthesize_text("hello", _profile(), words_per_minute=0)

    def test_scripted_cloner_end_to_end(self):
        cloner = ScriptedVoiceCloner()
        profile = cloner.clone("Demo", [_voice(220.0, 0.5)])
        audio = cloner.synthesize("hello mars", profile)
        assert audio.duration > 0.2
        assert audio.peak > 0.05


class TestProsody:
    def test_pitch_shift_one_octave(self):
        shifted = adjust_prosody(_profile(pitch=120.0), semitones=12)
        assert shifted.base_pitch_hz == pytest.approx(240.0)
        assert shifted.embedding == _profile(pitch=120.0).embedding

    def test_rate_override(self):
        faster = adjust_prosody(_profile(), words_per_minute=300)
        assert faster.speaking_rate_wpm == 300
        assert faster.base_pitch_hz == _profile().base_pitch_hz

    def test_invalid_rate_raises(self):
        with pytest.raises(VoiceCloneError, match="words_per_minute must be positive"):
            adjust_prosody(_profile(), words_per_minute=0)

    def test_time_stretch_changes_duration(self):
        audio = synthesize_text("hello world", _profile())
        stretched = time_stretch(audio, 2.0)
        assert stretched.duration == pytest.approx(2 * audio.duration, rel=0.02)
        assert stretched.sample_rate == audio.sample_rate
        compressed = time_stretch(audio, 0.5)
        assert compressed.duration == pytest.approx(0.5 * audio.duration, rel=0.02)

    def test_time_stretch_invalid_factor_raises(self):
        with pytest.raises(VoiceCloneError, match="factor must be positive"):
            time_stretch(_voice(220.0, 0.5), 0)

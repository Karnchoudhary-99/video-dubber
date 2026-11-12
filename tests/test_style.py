"""Tests for dubbing style presets: lookup, validation, application."""

import math

import pytest

from video_dubber.style import (
    BUILTIN_STYLES,
    DubStyle,
    StyleError,
    apply_style,
    get_style,
    list_styles,
)
from video_dubber.voice import EMBEDDING_DIM, SpeakerProfile, synthesize_text


def _profile() -> SpeakerProfile:
    return SpeakerProfile(
        name="Asha",
        embedding=[0.25] * EMBEDDING_DIM,
        base_pitch_hz=220.0,
        speaking_rate_wpm=160.0,
    )


class TestBuiltinStyles:
    def test_five_styles_listed(self):
        names = [style.name for style in list_styles()]
        assert names == ["documentary", "conversational", "energetic", "narration", "dramatic"]

    def test_conversational_is_the_neutral_default(self):
        style = get_style("conversational")
        assert style.rate_factor == 1.0
        assert style.pitch_semitones == 0.0

    def test_energetic_is_faster_and_higher(self):
        style = get_style("energetic")
        assert style.rate_factor > 1.0
        assert style.pitch_semitones > 0

    def test_dramatic_is_slower_and_lower(self):
        style = get_style("dramatic")
        assert style.rate_factor < 1.0
        assert style.pitch_semitones < 0

    def test_lookup_is_case_insensitive_and_trims_space(self):
        assert get_style("  Energetic ") is get_style("ENERGETIC")
        assert get_style("narration").name == "narration"

    def test_unknown_style_names_the_known_ones(self):
        with pytest.raises(StyleError, match="unknown dub style 'shouty'"):
            get_style("shouty")
        with pytest.raises(StyleError, match="known styles:"):
            get_style("nope")

    def test_style_dict_round_trip(self):
        style = get_style("documentary")
        data = style.to_dict()
        assert data["name"] == "documentary"
        assert data["rate_factor"] == 0.95
        assert data["pitch_semitones"] == 0.0
        assert "Measured" in data["description"]


class TestStyleValidation:
    def test_rate_out_of_range_rejected(self):
        with pytest.raises(StyleError, match="rate_factor must be between 0.5 and 2.0"):
            DubStyle(name="x", description="d", rate_factor=0.4)
        with pytest.raises(StyleError, match="rate_factor must be between 0.5 and 2.0"):
            DubStyle(name="x", description="d", rate_factor=2.5)

    def test_pitch_out_of_range_rejected(self):
        with pytest.raises(StyleError, match="pitch_semitones must be between -12 and 12"):
            DubStyle(name="x", description="d", pitch_semitones=13.0)

    def test_empty_name_rejected(self):
        with pytest.raises(StyleError, match="style name must not be empty"):
            DubStyle(name="  ", description="d")

    def test_builtin_styles_all_valid(self):
        for style in BUILTIN_STYLES:  # __post_init__ already ran at import
            assert 0.5 <= style.rate_factor <= 2.0
            assert -12.0 <= style.pitch_semitones <= 12.0


class TestApplyStyle:
    def test_energetic_scales_rate_and_raises_pitch(self):
        styled = apply_style(_profile(), "energetic")
        assert styled.speaking_rate_wpm == pytest.approx(160.0 * 1.15)
        assert styled.base_pitch_hz == pytest.approx(220.0 * 2 ** (2.0 / 12.0))

    def test_narration_slows_and_lowers(self):
        styled = apply_style(_profile(), get_style("narration"))
        assert styled.speaking_rate_wpm == pytest.approx(160.0 * 0.90)
        assert styled.base_pitch_hz == pytest.approx(220.0 * 2 ** (-1.0 / 12.0))

    def test_conversational_leaves_the_voice_unchanged(self):
        styled = apply_style(_profile(), "conversational")
        assert styled.speaking_rate_wpm == 160.0
        assert styled.base_pitch_hz == 220.0

    def test_identity_is_untouched(self):
        styled = apply_style(_profile(), "energetic")
        assert styled.name == "Asha"
        assert styled.embedding == [0.25] * EMBEDDING_DIM

    def test_original_profile_is_not_mutated(self):
        profile = _profile()
        apply_style(profile, "dramatic")
        assert profile.speaking_rate_wpm == 160.0
        assert profile.base_pitch_hz == 220.0

    def test_styled_voice_renders_shorter_audio(self):
        # Same text, faster style: the rendered clip is measurably shorter.
        text = "one two three four five six seven eight"
        plain = synthesize_text(text, _profile())
        fast = synthesize_text(text, apply_style(_profile(), "energetic"))
        slow = synthesize_text(text, apply_style(_profile(), "dramatic"))
        assert fast.duration < plain.duration < slow.duration
        # 1.15x rate -> ~1/1.15 the duration (word floor of 0.12s blurs it slightly).
        assert fast.duration == pytest.approx(plain.duration / 1.15, rel=0.05)

    def test_unknown_style_name_raises(self):
        with pytest.raises(StyleError, match="unknown dub style"):
            apply_style(_profile(), "shouty")

    def test_pitch_shift_matches_the_semitone_formula(self):
        styled = apply_style(_profile(), DubStyle("x", "d", 1.0, 12.0))
        assert styled.base_pitch_hz == pytest.approx(220.0 * 2.0)
        assert math.isclose(styled.base_pitch_hz / 220.0, 2.0)

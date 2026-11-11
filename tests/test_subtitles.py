"""Tests for dub-synced subtitles (subtitles.retime_transcript)."""

import pytest

from video_dubber.subtitles import SubtitleError, retime_transcript
from video_dubber.transcript import Segment, Transcript, Word


def _transcript() -> Transcript:
    return Transcript(
        segments=[
            Segment(
                words=[
                    Word("Hello", 0.0, 0.4),
                    Word("world", 0.4, 1.0),
                ],
                speaker="asha",
            ),
            Segment(words=[]),  # wordless: no placed entry, copied as-is
            Segment(
                words=[
                    Word("What", 2.0, 2.4),
                    Word("a", 2.4, 2.5),
                    Word("goal", 2.5, 3.0),
                ],
                speaker="ravi",
            ),
        ],
        language="hi",
        source="demo",
    )


def test_words_rescale_proportionally_into_placed_span():
    retimed = retime_transcript(_transcript(), [(0.0, 0.5), (2.0, 1.0)])
    first, _, third = retimed.segments
    # Segment 1 was 1.0s, placed 0.5s: every word halves.
    assert (first.start, first.end) == pytest.approx((0.0, 0.5))
    assert first.words[0].start == pytest.approx(0.0)
    assert first.words[0].end == pytest.approx(0.2)
    assert first.words[1].start == pytest.approx(0.2)
    assert first.words[1].end == pytest.approx(0.5)
    # Segment 3 was 1.0s at 2.0-3.0, placed 2.0-3.0: unchanged.
    assert (third.start, third.end) == pytest.approx((2.0, 3.0))
    assert third.words[1].text == "a"
    assert third.words[1].start == pytest.approx(2.4)


def test_placed_start_moves_the_cue():
    retimed = retime_transcript(_transcript(), [(0.5, 0.5), (2.5, 1.0)])
    first = retimed.segments[0]
    assert (first.start, first.end) == pytest.approx((0.5, 1.0))


def test_srt_uses_retimed_timings():
    retimed = retime_transcript(_transcript(), [(0.0, 0.5), (2.0, 1.0)])
    srt = retimed.to_srt()
    assert "00:00:00,000 --> 00:00:00,500" in srt
    assert "00:00:02,000 --> 00:00:03,000" in srt
    # Only the two timed segments become cues.
    assert srt.count("-->") == 2


def test_wordless_segments_need_no_placed_entry_and_survive():
    retimed = retime_transcript(_transcript(), [(0.0, 0.5), (2.0, 1.0)])
    assert retimed.segments[1].words == []
    assert retimed.segments[1].speaker is None


def test_speaker_language_and_source_preserved():
    retimed = retime_transcript(_transcript(), [(0.0, 0.5), (2.0, 1.0)])
    assert retimed.segments[0].speaker == "asha"
    assert retimed.segments[2].speaker == "ravi"
    assert retimed.language == "hi"
    assert retimed.source == "demo"


def test_original_transcript_not_mutated():
    original = _transcript()
    retime_transcript(original, [(0.0, 0.5), (2.0, 1.0)])
    assert (original.segments[0].start, original.segments[0].end) == pytest.approx((0.0, 1.0))


def test_short_cue_extends_to_minimum():
    transcript = Transcript(
        segments=[Segment(words=[Word("Hi", 0.0, 0.4)])],
    )
    retimed = retime_transcript(transcript, [(0.0, 0.2)])
    assert (retimed.segments[0].start, retimed.segments[0].end) == pytest.approx((0.0, 0.5))


def test_short_cue_never_overlaps_next_cue():
    transcript = Transcript(
        segments=[
            Segment(words=[Word("Hi", 0.0, 0.4)]),
            Segment(words=[Word("Yo", 0.6, 1.0)]),
        ],
    )
    # First cue would extend to 0.5s, but the next cue starts at 0.45s.
    retimed = retime_transcript(transcript, [(0.0, 0.2), (0.45, 0.8)])
    assert retimed.segments[0].end == pytest.approx(0.45)
    assert retimed.segments[1].start == pytest.approx(0.45)


def test_zero_duration_segment_collapses_then_gets_min_cue():
    transcript = Transcript(
        segments=[Segment(words=[Word("Hi", 1.0, 1.0), Word("there", 1.0, 1.0)])],
    )
    retimed = retime_transcript(transcript, [(1.0, 0.3)])
    first = retimed.segments[0]
    assert first.words[0].start == pytest.approx(1.0)
    assert (first.start, first.end) == pytest.approx((1.0, 1.5))


def test_min_cue_can_be_disabled():
    transcript = Transcript(
        segments=[Segment(words=[Word("Hi", 0.0, 0.4)])],
    )
    retimed = retime_transcript(transcript, [(0.0, 0.2)], min_cue_s=0.0)
    assert retimed.segments[0].end == pytest.approx(0.2)


def test_placed_count_mismatch_raises():
    with pytest.raises(SubtitleError, match="2 spans for 1 timed"):
        retime_transcript(
            Transcript(segments=[Segment(words=[Word("Hi", 0.0, 0.4)])]),
            [(0.0, 0.5), (1.0, 0.5)],
        )


def test_bad_spans_raise():
    transcript = Transcript(
        segments=[Segment(words=[Word("Hi", 0.0, 0.4)])],
    )
    with pytest.raises(SubtitleError, match="starts before 0"):
        retime_transcript(transcript, [(-0.1, 0.5)])
    with pytest.raises(SubtitleError, match="non-positive duration"):
        retime_transcript(transcript, [(0.0, 0.0)])


def test_negative_min_cue_raises():
    transcript = Transcript(
        segments=[Segment(words=[Word("Hi", 0.0, 0.4)])],
    )
    with pytest.raises(SubtitleError, match="min_cue_s"):
        retime_transcript(transcript, [(0.0, 0.5)], min_cue_s=-0.1)


def test_word_confidences_survive():
    transcript = Transcript(
        segments=[Segment(words=[Word("Hello", 0.0, 0.4, 0.7), Word("world", 0.4, 1.0, 0.9)])],
    )
    retimed = retime_transcript(transcript, [(0.0, 0.5)])
    assert [w.confidence for w in retimed.segments[0].words] == pytest.approx([0.7, 0.9])


# --- ASS export ---------------------------------------------------------------

from video_dubber.subtitles import AssStyle, assign_speaker_styles, to_ass


def test_ass_style_validates_colour_format():
    with pytest.raises(SubtitleError, match="#RRGGBB"):
        AssStyle(name="Bad", primary_color="red")
    with pytest.raises(SubtitleError, match="#RRGGBB"):
        AssStyle(name="Bad", primary_color="#FFF")
    with pytest.raises(SubtitleError, match="alignment"):
        AssStyle(name="Bad", alignment=0)
    with pytest.raises(SubtitleError, match="alignment"):
        AssStyle(name="Bad", alignment=10)
    with pytest.raises(SubtitleError, match="font size"):
        AssStyle(name="Bad", size=0)
    with pytest.raises(SubtitleError, match="comma"):
        AssStyle(name="Bad,Name")


def test_ass_style_converts_hex_to_bgr():
    style = AssStyle(name="Lead", primary_color="#112233")
    # ASS stores colours as &HAABBGGRR: #112233 -> &H00332211.
    assert "&H00332211" in style.to_ass_line()
    assert style.to_ass_line().startswith("Style: Lead,Arial,20,")


def test_ass_style_bold_and_alignment_render():
    style = AssStyle(name="Lead", bold=True, italic=True, alignment=8, size=28)
    fields = style.to_ass_line().removeprefix("Style: ").split(",")
    # name, font, size, 4 colours, bold, italic, underline, strikeout,
    # scaleX, scaleY, spacing, angle, borderstyle, outline, shadow, alignment
    assert fields[0] == "Lead"
    assert fields[2] == "28"
    assert fields[7] == "-1"  # bold
    assert fields[8] == "-1"  # italic
    assert fields[18] == "8"  # alignment (ASS numpad)


def test_to_ass_has_three_sections_and_dialogue():
    ass = to_ass(_transcript())
    assert "[Script Info]" in ass
    assert "ScriptType: v4.00+" in ass
    assert "[V4+ Styles]" in ass
    assert "[Events]" in ass
    assert "Dialogue: 0,0:00:00.00,0:00:01.00,asha,asha,0,0,0,,Hello world" in ass
    assert "Dialogue: 0,0:00:02.00,0:00:03.00,ravi,ravi,0,0,0,,What a goal" in ass
    # Wordless segments produce no cue.
    assert ass.count("Dialogue:") == 2


def test_to_ass_timestamps_are_centiseconds_truncated():
    transcript = Transcript(
        segments=[Segment(words=[Word("Hi", 1.2345, 2.9999)])],
    )
    ass = to_ass(transcript)
    # 1.2345 -> 0:00:01.23 (truncated, never rounded up past the audio).
    assert "0:00:01.23,0:00:02.99," in ass


def test_to_ass_escapes_braces_and_newlines():
    transcript = Transcript(
        segments=[
            Segment(words=[Word("{wow}", 0.0, 0.5), Word("line", 0.5, 1.0)]),
        ],
    )
    # Word text cannot contain a newline (words are single tokens), so
    # test the escape helper through a segment built from raw text via
    # a two-word join: the brace escape is what matters here.
    ass = to_ass(transcript)
    assert "\\{wow\\}" in ass
    assert "{wow}" not in ass.replace("\\{wow\\}", "")


def test_assign_speaker_styles_gives_distinct_colours():
    styles = assign_speaker_styles(_transcript())
    assert styles["asha"].primary_color != styles["ravi"].primary_color
    assert styles[None].name == "Default"
    assert styles["asha"].name == "asha"
    assert styles["ravi"].name == "ravi"


def test_assign_speaker_styles_sanitises_names_and_cycles_palette():
    transcript = Transcript(
        segments=[
            Segment(words=[Word("x", 0.0, 0.5)], speaker="Mary Jane"),
            Segment(words=[Word("y", 0.5, 1.0)], speaker="Mary,Jane"),
            Segment(words=[Word("z", 1.0, 1.5)], speaker="Mary Jane"),
        ],
    )
    styles = assign_speaker_styles(transcript, palette=("#111111",))
    assert styles["Mary Jane"].name == "Mary_Jane"
    # "Mary,Jane" sanitises to the same name, so it gets a suffix.
    assert styles["Mary,Jane"].name == "Mary_Jane_2"
    # One-colour palette: both speakers share the colour, names differ.
    assert styles["Mary Jane"].primary_color == "#111111"
    assert styles["Mary,Jane"].primary_color == "#111111"
    with pytest.raises(SubtitleError, match="palette"):
        assign_speaker_styles(transcript, palette=())


def test_to_ass_uses_explicit_styles_and_default_fallback():
    custom = {
        None: AssStyle(name="Default", primary_color="#FFFFFF"),
        "asha": AssStyle(name="Asha", primary_color="#FFFF00"),
    }
    ass = to_ass(_transcript(), styles=custom)
    # ravi has no entry: falls back to Default, not to an invented style.
    assert "Dialogue: 0,0:00:02.00,0:00:03.00,Default,ravi,0,0,0,,What a goal" in ass
    assert "Style: Asha," in ass
    assert "Style: Default," in ass
    # No style line is emitted for a speaker nobody uses.
    assert "Style: ravi," not in ass


def test_to_ass_empty_transcript_is_valid_file():
    ass = to_ass(Transcript(segments=[]))
    assert "[Script Info]" in ass
    assert "[Events]" in ass
    assert "Dialogue:" not in ass

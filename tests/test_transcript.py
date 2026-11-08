import pytest

from video_dubber.transcript import Segment, Transcript, Word, from_words


def _sample() -> Transcript:
    return Transcript(
        segments=[
            Segment(
                words=[
                    Word("hello", 0.0, 0.4, 0.95),
                    Word("world", 0.5, 0.9, 0.9),
                ],
                speaker="A",
            ),
            Segment(words=[Word("goodbye", 2.0, 2.6, 0.8)]),
        ],
        language="en",
        source="clip.wav",
    )


def test_word_validation():
    with pytest.raises(ValueError, match="precedes start"):
        Word("x", 1.0, 0.5)
    with pytest.raises(ValueError, match="confidence"):
        Word("x", 0.0, 0.5, confidence=1.5)


def test_segment_properties():
    segment = _sample().segments[0]
    assert segment.text == "hello world"
    assert segment.start == 0.0
    assert segment.end == 0.9
    assert segment.duration == pytest.approx(0.9)
    assert segment.mean_confidence == pytest.approx(0.925)
    assert Segment().text == ""
    assert Segment().mean_confidence == 0.0


def test_transcript_properties():
    t = _sample()
    assert t.word_count == 3
    assert t.duration == pytest.approx(2.6)
    assert t.text == "hello world goodbye"
    assert t.words_per_minute == pytest.approx(3 / 2.6 * 60)
    assert Transcript().words_per_minute == 0.0


def test_json_round_trip():
    t = _sample()
    restored = Transcript.from_json(t.to_json())
    assert restored == t
    assert restored.segments[0].speaker == "A"
    assert restored.language == "en"


def test_from_json_rejects_bad_input():
    with pytest.raises(ValueError, match="invalid transcript JSON"):
        Transcript.from_json("{not json")
    with pytest.raises(ValueError, match="invalid transcript JSON"):
        Transcript.from_json('{"segments": [{"words": [{"text": "x"}]}]}')


def test_to_srt_exact():
    expected = (
        "1\n00:00:00,000 --> 00:00:00,900\nhello world\n"
        "\n2\n00:00:02,000 --> 00:00:02,600\ngoodbye\n"
    )
    assert _sample().to_srt() == expected


def test_to_vtt_exact():
    text = _sample().to_vtt()
    assert text.startswith("WEBVTT\n")
    assert "00:00:00,000 --> 00:00:00,900\nhello world" in text
    assert "00:00:02,000 --> 00:00:02,600\ngoodbye" in text


def test_empty_segments_skipped_in_subtitles():
    t = Transcript(segments=[Segment(), _sample().segments[0]])
    assert t.to_srt().startswith("1\n00:00:00,000 --> 00:00:00,900\nhello world")


def test_from_words_groups_at_pauses():
    words = [
        Word("one", 0.0, 0.3),
        Word("two", 0.4, 0.7),
        Word("three", 2.0, 2.3),  # 1.3 s pause -> new segment
        Word("four", 2.4, 2.7),
    ]
    t = from_words(words, max_gap=0.8, language="en")
    assert len(t.segments) == 2
    assert t.segments[0].text == "one two"
    assert t.segments[1].text == "three four"
    assert t.language == "en"
    assert from_words([]).segments == []

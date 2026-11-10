import pytest

from video_dubber.patch import PatchError, patch_transcript, split_words_across_span
from video_dubber.transcript import Segment, Transcript, Word


def _transcript() -> Transcript:
    return Transcript(
        segments=[
            Segment(words=[Word("hello", 0.0, 0.5), Word("world", 0.5, 1.0)]),
            Segment(words=[]),  # wordless segments are not patchable, keep numbering
            Segment(
                words=[Word("second", 2.0, 2.5), Word("line", 2.5, 3.0)],
                speaker="Asha",
            ),
        ],
        language="hi",
    )


def test_split_words_proportional_to_length():
    words = split_words_across_span("a longerword", 0.0, 2.0)
    assert [w.text for w in words] == ["a", "longerword"]
    # "a" is 1 of 11 chars -> 2/11s; the long word takes the rest to 2.0s.
    assert words[0].start == pytest.approx(0.0)
    assert words[0].end == pytest.approx(2 / 11)
    assert words[1].start == pytest.approx(2 / 11)
    assert words[1].end == pytest.approx(2.0)


def test_split_words_rejects_blank_text():
    with pytest.raises(PatchError):
        split_words_across_span("   ", 0.0, 1.0)


def test_split_words_rejects_non_positive_span():
    with pytest.raises(PatchError):
        split_words_across_span("hi", 1.0, 1.0)


def test_patch_transcript_rewords_one_segment():
    patched = patch_transcript(_transcript(), 2, "nayi line")
    assert patched.segments[2].text == "nayi line"
    assert patched.segments[2].speaker == "Asha"
    assert patched.segments[2].start == pytest.approx(2.0)
    assert patched.segments[2].end == pytest.approx(3.0)
    # The other segment is untouched, and the wordless one stays wordless.
    assert patched.segments[0].text == "hello world"
    assert patched.segments[1].words == []
    assert patched.language == "hi"


def test_patch_transcript_does_not_mutate_input():
    original = _transcript()
    patch_transcript(original, 1, "badla hua")
    assert original.segments[0].text == "hello world"


def test_patch_transcript_index_is_among_timed_segments():
    # Index 2 is the second *timed* segment (the wordless one is skipped),
    # so patching it must touch the "second line" segment, not the gap.
    patched = patch_transcript(_transcript(), 2, "nayi line")
    assert patched.segments[2].text == "nayi line"


def test_patch_transcript_rejects_bad_index():
    with pytest.raises(PatchError):
        patch_transcript(_transcript(), 0, "x")
    with pytest.raises(PatchError):
        patch_transcript(_transcript(), 3, "x")


def test_patch_transcript_rejects_blank_text():
    with pytest.raises(PatchError):
        patch_transcript(_transcript(), 1, "  ")

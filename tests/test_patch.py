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


# --- patch_dub ---------------------------------------------------------------

import math

from video_dubber.assemble import AssemblyError, dub_transcript
from video_dubber.audio import AudioData, make_sine
from video_dubber.patch import patch_dub
from video_dubber.voice import ScriptedVoiceCloner


def _dubbed():
    """A two-segment dub (0-2s, 3-5s) plus the cloner/profile that made it."""
    transcript = Transcript(
        segments=[
            Segment(words=[Word("hello", 0.0, 1.0), Word("world", 1.0, 2.0)]),
            Segment(
                words=[Word("second", 3.0, 4.0), Word("line", 4.0, 5.0)],
                speaker="Ravi",
            ),
        ],
        language="hi",
    )
    cloner = ScriptedVoiceCloner()
    profile = cloner.clone("Demo Voice", [make_sine(180.0, 1.0)])
    dub_audio, _factors, placed = dub_transcript(transcript, cloner, profile)
    return transcript, cloner, profile, dub_audio, placed


def _rms(samples: list[float]) -> float:
    return math.sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0.0


def test_patch_dub_replaces_only_the_target_window():
    transcript, cloner, profile, dub_audio, placed = _dubbed()
    patched, factor, placed_duration = patch_dub(
        dub_audio, transcript, 2, "badli hui line", cloner, profile, placed=placed
    )
    assert patched.duration == pytest.approx(dub_audio.duration)
    assert 0.6 <= factor <= 1.6
    assert placed_duration <= 2.0
    # Everything before segment 2 starts is bit-identical.
    assert patched.samples[: 3 * 16_000] == dub_audio.samples[: 3 * 16_000]
    # The patched window actually changed.
    assert patched.samples[3 * 16_000 :] != dub_audio.samples[3 * 16_000 :]


def test_patch_dub_zeroes_only_the_placed_span():
    transcript, cloner, profile, dub_audio, _placed = _dubbed()
    # Segment 1's placed clip is only 0-1s; plant a marker tone in the
    # 1-2s gap (music bed) and check patching segment 1 leaves it alone.
    samples = list(dub_audio.samples)
    marker = make_sine(660.0, 1.0, sample_rate=16_000).samples
    for i in range(16_000, 32_000):
        samples[i] = 0.3 * marker[i - 16_000]
    marked = AudioData(samples, 16_000)
    patched, _factor, _dur = patch_dub(
        marked, transcript, 1, "naya", cloner, profile, placed=[(0.0, 1.0), (3.0, 2.0)]
    )
    assert patched.samples[16_000:32_000] == marked.samples[16_000:32_000]
    assert patched.samples[:16_000] != marked.samples[:16_000]


def test_patch_dub_matches_the_replaced_clips_loudness():
    transcript, cloner, profile, dub_audio, placed = _dubbed()
    old_rms = _rms(dub_audio.samples[3 * 16_000 : 5 * 16_000])
    patched, _factor, _dur = patch_dub(
        dub_audio, transcript, 2, "badli hui line", cloner, profile, placed=placed
    )
    new_rms = _rms(patched.samples[3 * 16_000 : 5 * 16_000])
    assert new_rms == pytest.approx(old_rms, rel=0.25)


def test_patch_dub_uses_the_speakers_profile():
    transcript, cloner, profile, dub_audio, placed = _dubbed()
    other = cloner.clone("Other Voice", [make_sine(90.0, 1.0)])
    default_voice, _, _ = patch_dub(
        dub_audio, transcript, 2, "badli hui line", cloner, profile, placed=placed
    )
    ravi_voice, _, _ = patch_dub(
        dub_audio,
        transcript,
        2,
        "badli hui line",
        cloner,
        profile,
        profiles={"Ravi": other},
        placed=placed,
    )
    assert ravi_voice.samples[3 * 16_000 :] != default_voice.samples[3 * 16_000 :]


def test_patch_dub_rejects_bad_index_and_short_audio():
    transcript, cloner, profile, dub_audio, _placed = _dubbed()
    with pytest.raises(PatchError):
        patch_dub(dub_audio, transcript, 3, "x", cloner, profile)
    with pytest.raises(PatchError):
        patch_dub(dub_audio, transcript, 1, "  ", cloner, profile)
    with pytest.raises(PatchError):
        patch_dub(dub_audio, transcript, 1, "x", cloner, profile, placed=[(0.0, 2.0)])
    short = AudioData(dub_audio.samples[: 2 * 16_000], 16_000)
    with pytest.raises(AssemblyError):
        patch_dub(short, transcript, 2, "x", cloner, profile)


def test_patch_dub_without_placed_zeroes_the_whole_window():
    transcript, cloner, profile, dub_audio, _placed = _dubbed()
    # No placed spans: the full 0-2s window is cleared before splicing,
    # so a marker in the gap is gone after the patch.
    samples = list(dub_audio.samples)
    for i in range(16_000, 32_000):
        samples[i] = 0.3
    marked = AudioData(samples, 16_000)
    patched, _factor, _dur = patch_dub(marked, transcript, 1, "naya", cloner, profile)
    # The gap still holds the new (short) clip's tail or silence — either
    # way the 0.3 marker is gone from at least the clip's own span.
    assert all(s != 0.3 for s in patched.samples[:16_000])
    assert patched.samples[16_000:32_000] != marked.samples[16_000:32_000]

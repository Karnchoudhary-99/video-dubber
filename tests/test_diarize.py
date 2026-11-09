"""Tests for speaker diarization: segment labelling against a voice library."""

import json

import pytest

from video_dubber.audio import concat, make_silence, make_sine
from video_dubber.diarize import (
    DiarizationReport,
    DiarizeError,
    SegmentLabel,
    diarize_transcript,
)
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.voice import VoiceLibrary, clone_from_references


def _library() -> VoiceLibrary:
    """Two voices a sine apart: Asha at 220 Hz, Ravi at 110 Hz."""
    library = VoiceLibrary()
    library.add(clone_from_references("Asha", [make_sine(220.0, 1.0)]))
    library.add(clone_from_references("Ravi", [make_sine(110.0, 1.0)]))
    return library


def _two_speaker_audio():
    """1 s of Asha (220 Hz) then 1 s of Ravi (110 Hz)."""
    return concat(make_sine(220.0, 1.0), make_sine(110.0, 1.0))


def _two_speaker_transcript() -> Transcript:
    return Transcript(
        segments=[
            Segment(words=[Word("hello", 0.0, 0.5), Word("mars", 0.5, 1.0)]),
            Segment(words=[Word("good", 1.0, 1.5), Word("morning", 1.5, 2.0)]),
        ],
        language="en",
    )


class TestDiarizeTranscript:
    def test_labels_each_segment_with_the_right_speaker(self):
        labelled, report = diarize_transcript(
            _two_speaker_transcript(), _two_speaker_audio(), _library()
        )
        assert [s.speaker for s in labelled.segments] == ["Asha", "Ravi"]
        assert report.matched == 2
        assert report.unknown == 0
        assert report.speakers == {"Asha": 1, "Ravi": 1}
        assert report.labels[0].status == "matched"
        assert report.labels[0].similarity == pytest.approx(1.0, abs=0.01)

    def test_unknown_voice_is_not_mislabelled(self):
        # 330 Hz is in neither profile (0.79-0.86 from both) -> unknown.
        audio = concat(make_sine(330.0, 1.0), make_sine(330.0, 1.0))
        labelled, report = diarize_transcript(_two_speaker_transcript(), audio, _library())
        assert [s.speaker for s in labelled.segments] == [None, None]
        assert report.matched == 0
        assert report.unknown == 2
        assert report.labels[0].status == "unknown"

    def test_silent_segment_is_unknown_not_an_error(self):
        audio = concat(make_sine(220.0, 1.0), make_silence(1.0))
        labelled, report = diarize_transcript(_two_speaker_transcript(), audio, _library())
        assert labelled.segments[0].speaker == "Asha"
        assert labelled.segments[1].speaker is None
        assert report.labels[1].status == "silent"
        assert report.unknown == 1

    def test_empty_segment_is_skipped_as_silent(self):
        transcript = Transcript(segments=[Segment(words=[]), _two_speaker_transcript().segments[0]])
        labelled, report = diarize_transcript(transcript, _two_speaker_audio(), _library())
        assert report.labels[0].status == "silent"
        assert labelled.segments[1].speaker == "Asha"

    def test_existing_labels_are_kept_unless_overwrite(self):
        transcript = _two_speaker_transcript()
        transcript.segments[0].speaker = "Ravi"  # wrong label, must survive
        labelled, report = diarize_transcript(transcript, _two_speaker_audio(), _library())
        assert labelled.segments[0].speaker == "Ravi"
        assert report.labels[0].status == "kept"
        assert report.kept == 1
        relabelled, _ = diarize_transcript(
            transcript, _two_speaker_audio(), _library(), overwrite=True
        )
        assert relabelled.segments[0].speaker == "Asha"

    def test_input_transcript_is_not_mutated(self):
        transcript = _two_speaker_transcript()
        diarize_transcript(transcript, _two_speaker_audio(), _library())
        assert all(s.speaker is None for s in transcript.segments)

    def test_threshold_controls_matching(self):
        # 330 Hz sits ~0.86 from the 220 Hz profile: matches at 0.80, not 0.90.
        transcript = Transcript(
            segments=[Segment(words=[Word("hello", 0.0, 0.5), Word("mars", 0.5, 1.0)])]
        )
        audio = make_sine(330.0, 1.0)
        labelled, _ = diarize_transcript(transcript, audio, _library(), threshold=0.80)
        assert labelled.segments[0].speaker == "Asha"
        labelled, _ = diarize_transcript(transcript, audio, _library(), threshold=0.90)
        assert labelled.segments[0].speaker is None

    def test_empty_library_and_bad_threshold_raise(self):
        with pytest.raises(DiarizeError, match="empty voice library"):
            diarize_transcript(_two_speaker_transcript(), _two_speaker_audio(), VoiceLibrary())
        with pytest.raises(DiarizeError, match="threshold must be in"):
            diarize_transcript(
                _two_speaker_transcript(), _two_speaker_audio(), _library(), threshold=1.5
            )

    def test_language_and_source_survive(self):
        transcript = _two_speaker_transcript()
        transcript.source = "clip.mp4"
        labelled, _ = diarize_transcript(transcript, _two_speaker_audio(), _library())
        assert labelled.language == "en"
        assert labelled.source == "clip.mp4"

    def test_synthesized_voices_need_a_higher_threshold(self):
        """Formant-synthesized demo voices sit close together.

        Measured: 110 vs 220 Hz synthesized speech cross at 0.906-0.921
        cosine similarity, above the 0.90 default — so similar demo voices
        need ``threshold=0.95`` (same-voice is 0.995+) to label correctly.
        This test pins that guidance instead of silently mislabelling.
        """
        from video_dubber.voice import EMBEDDING_DIM, SpeakerProfile, synthesize_text

        asha = SpeakerProfile(name="Asha", embedding=[0.25] * EMBEDDING_DIM, base_pitch_hz=220.0)
        ravi = SpeakerProfile(name="Ravi", embedding=[0.25] * EMBEDDING_DIM, base_pitch_hz=110.0)
        first = synthesize_text("hello mars welcome back", asha)
        second = synthesize_text("good morning from phobos", ravi)
        audio = concat(first, second)
        split = first.duration
        transcript = Transcript(
            segments=[
                Segment(
                    words=[
                        Word("hello", 0.0, split / 4),
                        Word("mars", split / 4, split / 2),
                        Word("welcome", split / 2, split * 3 / 4),
                        Word("back", split * 3 / 4, split),
                    ]
                ),
                Segment(
                    words=[
                        Word("good", split, split + second.duration / 4),
                        Word("morning", split + second.duration / 4, split + second.duration / 2),
                        Word("from", split + second.duration / 2, split + second.duration * 3 / 4),
                        Word("phobos", split + second.duration * 3 / 4, split + second.duration),
                    ]
                ),
            ],
            language="en",
        )
        labelled, report = diarize_transcript(transcript, audio, _library(), threshold=0.95)
        assert [s.speaker for s in labelled.segments] == ["Asha", "Ravi"]
        assert report.labels[0].speaker == "Asha"
        assert report.labels[1].speaker == "Ravi"


class TestDiarizationReport:
    def test_json_round_trip_and_markdown(self):
        _, report = diarize_transcript(_two_speaker_transcript(), _two_speaker_audio(), _library())
        raw = json.loads(report.to_json())
        assert raw["labels"][0]["speaker"] == "Asha"
        assert raw["labels"][1]["speaker"] == "Ravi"
        markdown = report.to_markdown()
        assert "2 segments: 2 matched, 0 kept, 0 unknown." in markdown
        assert "Asha" in markdown and "Ravi" in markdown

    def test_report_dataclass_shape(self):
        label = SegmentLabel(
            index=0,
            speaker="Asha",
            similarity=0.99,
            status="matched",
            start=0.0,
            end=1.0,
            text="hello mars",
        )
        report = DiarizationReport(labels=[label])
        assert report.matched == 1
        assert report.speakers == {"Asha": 1}

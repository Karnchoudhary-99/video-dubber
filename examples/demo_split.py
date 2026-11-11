"""Demo: split a long transcript segment at silence gaps.

Builds one 9.1-second segment — twelve words with a real 0.9-second pause
after word five — then splits it twice: once from the word timings alone,
once with the source audio (two tone bursts around a second of silence)
so the VAD confirms the cut point. Both splits land on the pause, the
longest piece drops from 9.1s to ~4.1s, and a continuous-speech segment
with no pause at all gets a counted forced cut instead of failing.

No model, no network, no ffmpeg needed — run it with
``python examples/demo_split.py``.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.audio import concat, make_silence, make_sine
from video_dubber.split import SplitConfig, split_transcript
from video_dubber.transcript import Segment, Transcript, Word


def _long_segment() -> Segment:
    words = []
    cursor = 0.0
    for i in range(12):
        words.append(Word(f"w{i}", cursor, cursor + 0.6))
        cursor += 0.7
        if i == 5:
            cursor += 0.8  # the real pause
    return Segment(words=words, speaker="A")


def main() -> None:
    transcript = Transcript(segments=[_long_segment()], language="en", source="clip.wav")
    print(
        f"before: {len(transcript.segments)} segment, {transcript.segments[0].duration:.2f}s long"
    )

    split, report = split_transcript(transcript, config=SplitConfig(max_duration=5.0))
    print(report.to_markdown(), end="")
    assert len(split.segments) == 2
    assert report.splits[0].reason == "silence"
    assert report.splits[0].at == transcript.segments[0].words[5].end
    assert report.max_duration_after <= 5.0

    # Same split, with the source audio: two tone bursts around 1s of
    # silence, so the VAD confirms the pause even if the timings hid it.
    audio = concat(make_sine(440.0, 3.6), make_silence(1.0), make_sine(440.0, 3.6))
    coarse = Transcript(
        segments=[
            Segment(
                words=[Word(f"w{i}", i * 0.6, i * 0.6 + 0.6) for i in range(12)],
                speaker="A",
            )
        ],
        language="en",
    )
    _, audio_report = split_transcript(coarse, audio=audio, config=SplitConfig(max_duration=4.0))
    print(
        f"audio-confirmed split: {audio_report.total_splits} cut, "
        f"reason {audio_report.splits[0].reason}"
    )
    assert audio_report.splits[0].reason == "silence"

    # Continuous speech has no pause anywhere: the split still happens,
    # but the report says so plainly instead of pretending it was clean.
    continuous = Transcript(
        segments=[Segment(words=[Word(f"w{i}", i * 0.6, i * 0.6 + 0.6) for i in range(12)])],
        language="en",
    )
    _, forced = split_transcript(continuous, config=SplitConfig(max_duration=4.0))
    print(f"continuous speech: {forced.total_splits} cut, {forced.forced_splits} forced")
    assert forced.forced_splits == 1

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "split.json"
        path.write_text(split.to_json(), encoding="utf-8")
        reloaded = Transcript.from_json(path.read_text(encoding="utf-8"))
        assert len(reloaded.segments) == 2
        print(f"split transcript round-trips through {path.name}")


if __name__ == "__main__":
    main()

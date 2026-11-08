"""End-to-end demo of the audio extraction + transcription pipeline.

Builds a synthetic 4-second clip (tone burst, silence, tone burst,
silence), runs VAD on it, transcribes it with the scripted demo backend
(including one word placed in the silence gap, which VAD alignment drops
as a hallucination), and prints the resulting subtitles.

No model and no ffmpeg needed — run it with ``python examples/demo.py``.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.audio import concat, make_silence, make_sine, save_wav
from video_dubber.transcribe import ScriptedTranscriber, align_to_speech
from video_dubber.transcript import Word
from video_dubber.vad import detect_speech, speech_ratio


def main() -> int:
    audio = concat(
        make_sine(440.0, 1.0),
        make_silence(1.0),
        make_sine(440.0, 1.0),
        make_silence(1.0),
    )
    wav_path = Path(tempfile.mkdtemp(prefix="video-dubber-demo-")) / "demo-clip.wav"
    save_wav(str(wav_path), audio)

    segments = detect_speech(audio)
    print(f"clip: {audio.duration:.1f}s, {len(segments)} speech segments")
    print(f"speech ratio: {speech_ratio(audio):.2f}")
    for segment in segments:
        print(f"  [{segment.start:.2f} - {segment.end:.2f}]")

    words = [
        Word("hello", 0.1, 0.4),
        Word("world", 0.5, 0.9),
        Word("ghost", 1.2, 1.6),  # falls in the silence gap
        Word("goodbye", 2.1, 2.5),
        Word("mars", 2.6, 2.9),
    ]
    raw = ScriptedTranscriber(words).transcribe(audio)
    aligned = align_to_speech(raw, audio)
    print(f"\nraw transcript: {raw.word_count} words ({raw.text!r})")
    print(f"VAD-aligned: {aligned.word_count} words ({aligned.text!r})")
    print("\nSRT output:\n")
    print(aligned.to_srt())
    print(f"demo clip saved to {wav_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""End-to-end demo of the voice cloning pipeline.

Clones a voice from two synthetic reference tones (a low voice and a
higher one), shows the embeddings tell them apart, synthesises the same
line in each voice, and saves both WAVs so you can hear the pitch and
timbre difference.

No model and no real recordings needed — run it with
``python examples/demo_voice.py``.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.audio import make_sine, save_wav
from video_dubber.voice import (
    ScriptedVoiceCloner,
    adjust_prosody,
    cosine_similarity,
    extract_embedding,
)


def main() -> int:
    cloner = ScriptedVoiceCloner()
    low_ref = make_sine(110.0, 1.0)
    high_ref = make_sine(260.0, 1.0)
    low = cloner.clone("Low Voice", [low_ref])
    high = cloner.clone("High Voice", [high_ref])
    print(f"cloned {low.name}: base pitch {low.base_pitch_hz:.0f} Hz")
    print(f"cloned {high.name}: base pitch {high.base_pitch_hz:.0f} Hz")
    print(
        f"same-voice similarity: {cosine_similarity(extract_embedding(low_ref), low.embedding):.3f}"
    )
    print(f"cross-voice similarity: {cosine_similarity(low.embedding, high.embedding):.3f}")

    out_dir = Path(tempfile.mkdtemp(prefix="video-dubber-voice-"))
    line = "hello mars welcome to the final"
    for profile in (low, high):
        audio = cloner.synthesize(line, profile)
        path = out_dir / f"{profile.name.replace(' ', '-').lower()}.wav"
        save_wav(str(path), audio)
        print(f"{profile.name}: {audio.duration:.2f}s, peak {audio.peak:.2f} -> {path}")

    shifted = adjust_prosody(low, semitones=12)
    audio = cloner.synthesize(line, shifted)
    path = out_dir / "low-voice-plus-12-semitones.wav"
    save_wav(str(path), audio)
    print(f"low voice +12 semitones: {audio.duration:.2f}s -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

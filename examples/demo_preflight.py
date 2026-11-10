"""Demo: pre-flight checks on a good, a clipped, and a silent recording.

Builds three synthetic source clips and runs :func:`run_preflight` on
each: a clean two-burst recording (passes), the same bursts driven to
full scale (fails on clipping), and pure silence (fails on level and
speech coverage). Prints the Markdown report for each, so the check
names and the plain-language fixes are visible.

No model, no network, no ffmpeg needed — run it with
``python examples/demo_preflight.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.audio import AudioData, concat, make_silence, make_sine
from video_dubber.preflight import run_preflight


def _speech_bursts(amplitude: float = 0.3) -> AudioData:
    """Two 1s tone bursts with 1s of silence between and after them."""
    return concat(
        make_sine(440.0, 1.0, amplitude=amplitude),
        make_silence(1.0),
        make_sine(440.0, 1.0, amplitude=amplitude),
        make_silence(1.0),
    )


def main() -> None:
    cases = [
        ("clean recording", _speech_bursts()),
        ("clipped recording", AudioData([1.0] * 64_000, 16_000)),
        ("silent recording", make_silence(4.0)),
    ]
    for label, audio in cases:
        print(f"=== {label} ===")
        print(run_preflight(audio).to_markdown(), end="")
        print()


if __name__ == "__main__":
    main()

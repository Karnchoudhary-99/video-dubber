"""Demo of environment diagnostics.

Runs the full doctor check on this machine and prints the report:
required tools (python, ffmpeg, ffprobe, the subtitles filter) and
optional ones (whisper, argos-translate) plus free disk space. On a
bare machine the verdict is honest about what is missing instead of
failing halfway through a dub.

Run it with ``python examples/demo_doctor.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.doctor import run_doctor


def main() -> int:
    report = run_doctor()
    print(report.to_markdown(), end="")
    print(f"verdict: {report.verdict} ({report.passed}/{len(report.checks)} checks pass)")

    # The report is always well-formed, whatever the machine has.
    assert report.verdict in ("ok", "ok_with_warnings", "fail")
    assert len(report.checks) == 7
    assert {c.name for c in report.checks} == {
        "python",
        "ffmpeg",
        "ffprobe",
        "ffmpeg subtitles filter",
        "whisper",
        "argos-translate",
        "disk space",
    }
    # Python itself must pass — the demo is running on it.
    python_check = next(c for c in report.checks if c.name == "python")
    assert python_check.status == "pass"
    print("checks passed: report is complete and well-formed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

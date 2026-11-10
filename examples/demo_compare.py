"""Demo: compare two dubs of the same source, metric by metric.

Builds two synthetic quality reports — one from a clean dub (tight
drift, near-1.0 fits, MOS proxy 4.5) and one from a struggling dub
(drift over 0.10s, a 0.7x mean fit, MOS proxy 3.2) — and prints the
Markdown comparison table with the verdict. The numbers are the ones
:func:`compare_reports` actually scores on, so the table shows why the
winner won instead of just naming it.

No model, no network, no ffmpeg needed — run it with
``python examples/demo_compare.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.assemble import DubReport
from video_dubber.compare import compare_reports


def main() -> None:
    asha = DubReport(
        segments_dubbed=2,
        total_duration=2.5,
        mean_sync_drift=0.02,
        max_sync_drift=0.05,
        segments_over_drift=0,
        mean_fit_factor=1.02,
        max_fit_factor=1.1,
        extreme_fits=0,
        loudness_error_db=0.5,
        translation_coverage=0.9,
        mos_proxy=4.5,
        verdict="good",
    )
    ravi = DubReport(
        segments_dubbed=2,
        total_duration=2.5,
        mean_sync_drift=0.15,
        max_sync_drift=0.4,
        segments_over_drift=1,
        mean_fit_factor=0.7,
        max_fit_factor=1.6,
        extreme_fits=1,
        loudness_error_db=2.0,
        translation_coverage=0.6,
        word_error_rate=0.2,
        mos_proxy=3.2,
        verdict="review",
    )
    print(compare_reports(asha, ravi, "asha", "ravi").to_markdown(), end="")


if __name__ == "__main__":
    main()

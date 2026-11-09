"""Demo: the HTML quality report for a dub.

Runs the short end-to-end dub from demo_dub.py, then renders the
quality report as a self-contained HTML page — verdict banner, MOS
proxy bar, headline metrics, and a per-segment table that flags the
segments that drifted or stretched. The second segment is deliberately
slow (the Hindi line is longer than its window at demo speed), so the
report's verdict is ``review`` and the table flags that row: the HTML
report is the thing that names the segment to fix.

No model, no network, no ffmpeg needed — run it with
``python examples/demo_report.py`` and open the printed path in a browser.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.assemble import build_report, dub_transcript
from video_dubber.audio import make_sine, save_wav
from video_dubber.report import render_html_report
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.translate import DictionaryTranslator, translate_transcript
from video_dubber.voice import ScriptedVoiceCloner


def main() -> int:
    source = Transcript(
        segments=[
            Segment(
                words=[
                    Word("Hello", 0.0, 0.4),
                    Word("and", 0.4, 0.6),
                    Word("welcome", 0.6, 1.1),
                    Word("to", 1.1, 1.3),
                    Word("Mars", 1.3, 1.7),
                    Word("Cricket", 1.7, 2.2),
                ],
                speaker="commentator",
            ),
            Segment(
                words=[
                    Word("What", 2.5, 2.8),
                    Word("a", 2.8, 2.9),
                    Word("goal", 2.9, 3.3),
                    Word("by", 3.3, 3.5),
                    Word("the", 3.5, 3.6),
                    Word("champion", 3.6, 4.1),
                ],
                speaker="commentator",
            ),
        ],
        language="en",
    )
    translator = DictionaryTranslator(source="en", target="hi")
    translated = translate_transcript(source, translator, "hi")
    cloner = ScriptedVoiceCloner()
    profile = cloner.clone("Demo Commentator", [make_sine(180.0, 1.0)])
    dub_audio, fit_factors, placed = dub_transcript(translated, cloner, profile)
    report = build_report(
        translated,
        placed,
        fit_factors,
        dub_audio,
        translation_coverage=translator.last_coverage,
    )
    html = render_html_report(report)
    out_dir = Path(tempfile.mkdtemp(prefix="video-dubber-report-"))
    wav_path = out_dir / "dubbed.wav"
    save_wav(str(wav_path), dub_audio)
    report_path = out_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"dubbed track: {dub_audio.duration:.2f}s -> {wav_path}")
    print(f"verdict: {report.verdict} (MOS proxy {report.mos_proxy:.2f})")
    print(f"HTML report -> {report_path}")
    assert "<!DOCTYPE html>" in html
    assert 'class="banner review"' in html  # the slow second segment flags review
    assert "<h2>Segments</h2>" in html
    assert html.count('class="flagged"') == 1
    assert report_path.read_text(encoding="utf-8") == html
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

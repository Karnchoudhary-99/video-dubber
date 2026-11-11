"""Demo: render the dubbed video with subtitles burned into the picture.

Builds a tiny video (ffmpeg colour bars + tone), dubs two lines into
Hindi with the scripted voice double, retimes the subtitles to the
placed dub audio, then renders the full video with the cues burned in
— plus a 2-second preview slice starting at 1s, so a long dub can be
spot-checked without re-encoding the whole file.

Needs ffmpeg (with the libass ``subtitles`` filter) — run it with
``python examples/demo_render.py``.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.assemble import dub_transcript
from video_dubber.audio import make_sine, save_wav
from video_dubber.extract import ffmpeg_available
from video_dubber.render import RenderError, render_preview, render_video
from video_dubber.subtitles import retime_transcript
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.translate import DictionaryTranslator, translate_transcript
from video_dubber.voice import ScriptedVoiceCloner


def _make_video(path: Path, duration: float = 4.0) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=red:s=128x128:d={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )


def main() -> int:
    if not ffmpeg_available():
        raise RenderError("demo needs ffmpeg/ffprobe on PATH")
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
    translated = translate_transcript(source, DictionaryTranslator(source="en", target="hi"), "hi")
    cloner = ScriptedVoiceCloner()
    profile = cloner.clone("Demo Commentator", [make_sine(180.0, 1.0)])
    dub_audio, _factors, placed = dub_transcript(translated, cloner, profile)

    out_dir = Path(tempfile.mkdtemp(prefix="video-dubber-render-"))
    video = out_dir / "clip.mp4"
    _make_video(video)
    wav_path = out_dir / "dubbed.wav"
    save_wav(str(wav_path), dub_audio)
    synced = retime_transcript(translated, placed)
    srt_path = out_dir / "dubbed.srt"
    srt_path.write_text(synced.to_srt(), encoding="utf-8")

    rendered = render_video(
        str(video), str(wav_path), str(out_dir / "rendered.mp4"), subtitles_path=str(srt_path)
    )
    print(
        f"full render: {rendered.duration:.2f}s, "
        f"{rendered.subtitle_cues} cue(s) burned in "
        f"({rendered.filter_used} filter) -> {rendered.out_path}"
    )

    preview = render_preview(
        str(video),
        str(wav_path),
        str(out_dir / "preview.mp4"),
        subtitles_path=str(srt_path),
        start=1.0,
        duration=2.0,
    )
    print(
        f"preview slice: {preview.duration:.2f}s from 1.0s "
        f"({preview.subtitle_cues} cue(s)) -> {preview.out_path}"
    )
    print(
        "both files play anywhere — the subtitles are in the picture, "
        "no player subtitle support needed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

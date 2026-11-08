"""Command-line entry point.

Subcommands (see ``video-dubber <cmd> --help`` for details):

* ``extract`` — pull a video's audio track out to a 16 kHz mono WAV.
* ``transcribe`` — speech-to-text on a WAV (whisper CLI, or a scripted
  demo transcript with ``--backend scripted --script words.json``).
* ``vad`` — list the speech segments an energy VAD finds in a WAV.
* ``convert`` — re-serialise a transcript JSON file as SRT / WebVTT / JSON.
"""

import argparse
import sys

from video_dubber import __version__
from video_dubber.audio import DEFAULT_SAMPLE_RATE, load_wav, resample
from video_dubber.extract import ExtractionError, extract_audio
from video_dubber.transcribe import (
    ScriptedTranscriber,
    TranscriptionError,
    WhisperCliTranscriber,
    align_to_speech,
    transcribe_file,
)
from video_dubber.transcript import Transcript
from video_dubber.vad import VadConfig, detect_speech, speech_ratio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-dubber",
        description="Multilingual video dubbing with voice cloning.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_extract = sub.add_parser("extract", help="Extract audio from a video to WAV.")
    p_extract.add_argument("video", help="Input video file.")
    p_extract.add_argument("--out", required=True, help="Output WAV path.")
    p_extract.add_argument(
        "--rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Sample rate (default 16000)."
    )

    p_transcribe = sub.add_parser("transcribe", help="Transcribe a WAV file.")
    p_transcribe.add_argument("audio", help="Input WAV file.")
    p_transcribe.add_argument(
        "--backend",
        choices=["whisper", "scripted"],
        default="whisper",
        help="STT backend (default whisper).",
    )
    p_transcribe.add_argument("--model", default="tiny", help="Whisper model (default tiny).")
    p_transcribe.add_argument("--language", help="Source language code, e.g. en, hi.")
    p_transcribe.add_argument(
        "--script",
        help="JSON word list for the scripted backend: [{text, start, end}].",
    )
    p_transcribe.add_argument(
        "--align",
        action="store_true",
        help="Re-segment the transcript at VAD speech boundaries.",
    )
    p_transcribe.add_argument("--out", help="Write transcript to this file.")
    p_transcribe.add_argument(
        "--format",
        choices=["json", "srt", "vtt"],
        default="json",
        help="Serialisation for --out (default json).",
    )

    p_vad = sub.add_parser("vad", help="List speech segments in a WAV file.")
    p_vad.add_argument("audio", help="Input WAV file.")

    p_convert = sub.add_parser("convert", help="Convert a transcript JSON file.")
    p_convert.add_argument("transcript", help="Input transcript JSON file.")
    p_convert.add_argument(
        "--format", choices=["srt", "vtt", "json"], default="srt", help="Output format."
    )
    p_convert.add_argument("--out", help="Write to this file instead of stdout.")
    return parser


def _cmd_extract(args: argparse.Namespace) -> int:
    out = extract_audio(args.video, args.out, sample_rate=args.rate)
    audio = load_wav(out)
    print(f"wrote {out} ({audio.duration:.2f}s, {audio.sample_rate} Hz mono)")
    return 0


def _cmd_transcribe(args: argparse.Namespace) -> int:
    if args.backend == "scripted":
        if not args.script:
            print("error: --backend scripted needs --script words.json", file=sys.stderr)
            return 2
        transcriber = ScriptedTranscriber.from_json_file(args.script)
    else:
        transcriber = WhisperCliTranscriber(model=args.model)
    transcript = transcribe_file(args.audio, transcriber, language=args.language)
    if args.align:
        audio = resample(load_wav(args.audio), DEFAULT_SAMPLE_RATE)
        transcript = align_to_speech(transcript, audio)
    print(
        f"{transcript.word_count} words, {transcript.duration:.2f}s, "
        f"{transcript.words_per_minute:.0f} wpm, {len(transcript.segments)} segments"
    )
    if args.out:
        text = {
            "json": transcript.to_json(),
            "srt": transcript.to_srt(),
            "vtt": transcript.to_vtt(),
        }[args.format]
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        for segment in transcript.segments:
            print(f"[{segment.start:7.2f} - {segment.end:7.2f}] {segment.text}")
    return 0


def _cmd_vad(args: argparse.Namespace) -> int:
    audio = resample(load_wav(args.audio), DEFAULT_SAMPLE_RATE)
    segments = detect_speech(audio, VadConfig())
    print(
        f"{len(segments)} speech segments, "
        f"speech ratio {speech_ratio(audio):.2f}, duration {audio.duration:.2f}s"
    )
    for segment in segments:
        print(f"[{segment.start:7.2f} - {segment.end:7.2f}] {segment.duration:.2f}s")
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    with open(args.transcript, encoding="utf-8") as fh:
        transcript = Transcript.from_json(fh.read())
    text = {
        "json": transcript.to_json(),
        "srt": transcript.to_srt(),
        "vtt": transcript.to_vtt(),
    }[args.format]
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    try:
        if args.command == "extract":
            return _cmd_extract(args)
        if args.command == "transcribe":
            return _cmd_transcribe(args)
        if args.command == "vad":
            return _cmd_vad(args)
        if args.command == "convert":
            return _cmd_convert(args)
    except (ExtractionError, TranscriptionError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

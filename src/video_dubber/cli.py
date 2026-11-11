"""Command-line entry point.

Subcommands (see ``video-dubber <cmd> --help`` for details):

* ``extract`` — pull a video's audio track out to a 16 kHz mono WAV.
* ``transcribe`` — speech-to-text on a WAV (whisper CLI, or a scripted
  demo transcript with ``--backend scripted --script words.json``).
* ``vad`` — list the speech segments an energy VAD finds in a WAV.
* ``convert`` — re-serialise a transcript JSON file as SRT / WebVTT / ASS / JSON.
* ``translate`` — translate a transcript JSON file to another language
  (dictionary backend, or ``argos-translate`` when installed).
* ``clone`` — build a speaker profile from reference WAV recordings.
* ``voices`` — list the profiles in a voice library, or add one to it.
* ``diarize`` — label who speaks in each transcript segment, by matching
  the segment audio against a voice library.
* ``split`` — split long transcript segments at silence gaps, so each
  piece fits its dub window without an extreme time-stretch.
* ``merge`` — join over-split transcript segments back into readable
  lines, across short pauses with the same speaker.
* ``synthesize`` — render text as audio in a cloned voice (demo synthesis).
* ``cache`` — inspect or clear the segment synthesis cache.
* ``dub`` — render a full dubbed track from a translated transcript and
  a speaker profile, with a quality report (and optional video mux).
* ``patch`` — re-dub one segment of an existing dub in place (new text,
  same window), and refresh its transcript and quality report.
* ``report`` — re-render a saved JSON quality report as HTML or Markdown.
* ``enhance`` — clean up a WAV: trim silence, cut hum, gate room noise,
  normalize loudness to a target RMS.
* ``estimate`` — predict a dub before running it: speech durations, fit
  factors, wall time, cache savings and cost, from a translated transcript.
* ``batch`` — dub many videos/transcripts from one manifest JSON file.
* ``watch`` — dub every new video that lands in a folder, as it appears.
* ``config`` — show the effective configuration (dubber.toml defaults).
"""

import argparse
import os
import sys
from pathlib import Path

from video_dubber import __version__
from video_dubber.assemble import (
    AssemblyError,
    DubReport,
    SegmentRow,
    build_report,
    dub_transcript,
    mux_audio,
)
from video_dubber.audio import DEFAULT_SAMPLE_RATE, load_wav, resample, save_wav
from video_dubber.batch import (
    BatchError,
    BatchManifest,
    run_batch,
    summarize,
)
from video_dubber.cache import (
    CachedVoiceCloner,
    CacheError,
    SynthesisCache,
    default_cache_dir,
)
from video_dubber.compare import compare_reports
from video_dubber.config import ConfigError, DubberConfig, load_config
from video_dubber.diarize import (
    DEFAULT_THRESHOLD,
    DiarizeError,
    diarize_transcript,
)
from video_dubber.enhance import (
    DEFAULT_CEILING_DBFS,
    DEFAULT_GATE_THRESHOLD_DBFS,
    DEFAULT_HIGH_PASS_HZ,
    DEFAULT_TARGET_DBFS,
    enhance_audio,
)
from video_dubber.estimate import BACKEND_RTF, EstimateError, estimate_dub
from video_dubber.export import (
    ExportError,
    build_manifest,
    export_bundle,
    export_zip,
    verify_bundle,
)
from video_dubber.extract import ExtractionError, extract_audio
from video_dubber.fidelity import FidelityError, check_fidelity
from video_dubber.merge import MergeConfig, MergeError, merge_transcript
from video_dubber.multilang import MultiTargetError, dub_all_targets, parse_targets
from video_dubber.patch import PatchError, patch_dub, patch_transcript, refresh_report
from video_dubber.preflight import run_preflight
from video_dubber.render import RenderError, SubtitleStyle, render_preview, render_video
from video_dubber.report import render_html_report
from video_dubber.split import SplitConfig, SplitError, split_transcript
from video_dubber.subtitles import retime_transcript, to_ass
from video_dubber.transcribe import (
    ScriptedTranscriber,
    TranscriptionError,
    WhisperCliTranscriber,
    align_to_speech,
    transcribe_file,
)
from video_dubber.transcript import Transcript
from video_dubber.translate import (
    ArgosTranslator,
    DictionaryTranslator,
    Glossary,
    TranslationError,
    translate_transcript,
)
from video_dubber.vad import VadConfig, detect_speech, speech_ratio
from video_dubber.voice import (
    ScriptedVoiceCloner,
    SpeakerProfile,
    VoiceCloneError,
    VoiceLibrary,
    adjust_prosody,
)
from video_dubber.watch import WatchConfig, WatchError, watch_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-dubber",
        description="Multilingual video dubbing with voice cloning.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--config",
        help="Config file (dubber.toml) to load; default search: ./dubber.toml, "
        "~/.config/video-dubber/dubber.toml.",
    )
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
        "--format", choices=["srt", "vtt", "ass", "json"], default="srt", help="Output format."
    )
    p_convert.add_argument("--out", help="Write to this file instead of stdout.")

    p_translate = sub.add_parser("translate", help="Translate a transcript JSON file.")
    p_translate.add_argument("transcript", help="Input transcript JSON file.")
    p_translate.add_argument(
        "--target",
        help="Target language code, e.g. hi, es (or target_language in dubber.toml).",
    )
    p_translate.add_argument("--source", help="Source language code (default: transcript's).")
    p_translate.add_argument(
        "--backend",
        choices=["dictionary", "argos"],
        default=None,
        help="Translation backend (default dictionary, or backend in dubber.toml).",
    )
    p_translate.add_argument("--glossary", help="JSON glossary file of term overrides.")
    p_translate.add_argument(
        "--max-chars",
        type=int,
        help="Split long translated segments into cues of at most this many chars.",
    )
    p_translate.add_argument(
        "--round-trip",
        action="store_true",
        help="After translating, back-translate each segment and report "
        "round-trip fidelity (flags segments that came back mangled).",
    )
    p_translate.add_argument(
        "--fidelity-out",
        help="Write the fidelity report here (needs --round-trip).",
    )
    p_translate.add_argument(
        "--fidelity-threshold",
        type=float,
        default=0.6,
        help="Flag segments scoring below this in the round-trip check "
        "(0-1, default 0.6; needs --round-trip).",
    )
    p_translate.add_argument(
        "--fidelity-format",
        choices=["json", "md"],
        default="json",
        help="Fidelity report format (default json).",
    )
    p_translate.add_argument("--out", help="Write transcript to this file.")
    p_translate.add_argument(
        "--format",
        choices=["json", "srt", "vtt"],
        default="json",
        help="Serialisation for --out (default json).",
    )

    p_clone = sub.add_parser("clone", help="Build a speaker profile from reference WAVs.")
    p_clone.add_argument("references", nargs="+", help="Reference WAV recordings.")
    p_clone.add_argument("--name", required=True, help="Speaker name for the profile.")
    p_clone.add_argument("--wpm", type=float, help="Speaking rate if known (default 150).")
    p_clone.add_argument("--out", required=True, help="Output profile JSON path.")
    p_clone.add_argument(
        "--library",
        help="Voice library JSON to add this profile to (created if missing).",
    )

    p_voices = sub.add_parser("voices", help="List profiles in a voice library.")
    p_voices.add_argument("library", help="Voice library JSON path.")
    p_voices.add_argument("--add", help="Profile JSON to add to the library (then list).")
    p_voices.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing profile of the same name when using --add.",
    )

    p_synth = sub.add_parser("synthesize", help="Render text in a cloned voice.")
    p_synth.add_argument("text", help="Text to speak.")
    p_synth.add_argument("--profile", required=True, help="Speaker profile JSON path.")
    p_synth.add_argument("--wpm", type=float, help="Speaking rate override.")
    p_synth.add_argument("--semitones", type=float, default=0.0, help="Pitch shift in semitones.")
    p_synth.add_argument("--out", required=True, help="Output WAV path.")
    p_synth.add_argument(
        "--cache-dir",
        help="Cache rendered segments here so re-rendering the same text in "
        "the same voice is instant (default: no cache).",
    )

    p_cache = sub.add_parser("cache", help="Inspect or clear the synthesis cache.")
    p_cache.add_argument(
        "--cache-dir",
        help="Cache directory to inspect (default: ~/.cache/video-dubber/synth).",
    )
    p_cache.add_argument(
        "--clear",
        action="store_true",
        help="Delete every cached segment and reset the hit/miss counts.",
    )

    p_diarize = sub.add_parser("diarize", help="Label who speaks in each transcript segment.")
    p_diarize.add_argument("transcript", help="Input transcript JSON file.")
    p_diarize.add_argument("--audio", required=True, help="Original audio WAV.")
    p_diarize.add_argument(
        "--library",
        help="Voice library JSON path (or voice_library in dubber.toml).",
    )
    p_diarize.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum cosine similarity to label a segment (default 0.90).",
    )
    p_diarize.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-label segments that already carry a speaker label.",
    )
    p_diarize.add_argument("--out", help="Write the labelled transcript to this file.")
    p_diarize.add_argument("--report-out", help="Write the diarization report to this file.")
    p_diarize.add_argument(
        "--report-format",
        choices=["json", "md"],
        default="json",
        help="Diarization report format (default json).",
    )

    p_dub = sub.add_parser("dub", help="Render a dubbed track from a translated transcript.")
    p_dub.add_argument("transcript", help="Translated transcript JSON file.")
    p_dub.add_argument("--profile", required=True, help="Speaker profile JSON path.")
    p_dub.add_argument(
        "--profiles",
        help="JSON file mapping speaker names to profile JSON paths, for "
        "multi-speaker transcripts (unlisted speakers use --profile).",
    )
    p_dub.add_argument(
        "--subtitles",
        action="store_true",
        help="Also write <out-stem>.srt, <out-stem>.vtt and <out-stem>.ass, "
        "retimed to the dubbed audio (cues match the placed speech, not "
        "the original windows).",
    )
    p_dub.add_argument("--original", help="Original audio WAV (for loudness matching).")
    p_dub.add_argument(
        "--cache-dir",
        help="Cache rendered segments here so re-dubbing the same transcript "
        "in the same voice skips synthesis (default: no cache).",
    )
    p_dub.add_argument(
        "--preflight",
        action="store_true",
        help="Check --original first and abort (exit 1) if it fails the "
        "pre-flight checks, instead of dubbing a bad recording.",
    )
    p_dub.add_argument(
        "--bed-level",
        type=float,
        default=None,
        help="Keep the original's music bed under the dub at this mix level "
        "(0-1; needs --original).",
    )
    p_dub.add_argument("--out", required=True, help="Output dubbed WAV path.")
    p_dub.add_argument("--video", help="Original video to mux the dub into.")
    p_dub.add_argument("--mux-out", help="Output video path (needs --video).")
    p_dub.add_argument(
        "--burn-subtitles",
        action="store_true",
        help="Burn the retimed subtitles into the muxed video (needs "
        "--video/--mux-out; writes the SRT next to the dub WAV).",
    )
    p_dub.add_argument(
        "--split-segments",
        action="store_true",
        help="Split segments longer than --max-segment-duration at silence "
        "gaps before dubbing, so long lines fit their windows (needs no "
        "extra input; --original sharpens the split points via VAD).",
    )
    p_dub.add_argument(
        "--max-segment-duration",
        type=float,
        default=None,
        help="Longest segment to dub as one piece, in seconds (default 8.0; "
        "only with --split-segments).",
    )
    p_dub.add_argument("--report-out", help="Write the quality report to this file.")
    p_dub.add_argument(
        "--reference",
        help="Same-language ground-truth transcript (JSON) of what the dub should "
        "say; enables WER in the quality report.",
    )
    p_dub.add_argument(
        "--enhance",
        action="store_true",
        help="Enhance the finished dub track (hum cut, noise gate, loudness "
        "normalize to -16 dBFS); timing is preserved, no trimming.",
    )
    p_dub.add_argument(
        "--report-format",
        choices=["json", "md", "html"],
        default="json",
        help="Quality report format (default json).",
    )

    p_dub_multi = sub.add_parser(
        "dub-multi",
        help="Dub one source transcript into several target languages at once.",
    )
    p_dub_multi.add_argument("transcript", help="Source transcript JSON file.")
    p_dub_multi.add_argument(
        "--targets",
        required=True,
        help="Comma-separated target language codes, e.g. hi,es,fr.",
    )
    p_dub_multi.add_argument("--profile", required=True, help="Speaker profile JSON path.")
    p_dub_multi.add_argument(
        "--profiles",
        help="JSON file mapping speaker names to profile JSON paths, for "
        "multi-speaker transcripts (unlisted speakers use --profile).",
    )
    p_dub_multi.add_argument("--source", help="Source language code (default: transcript's).")
    p_dub_multi.add_argument(
        "--backend",
        choices=["dictionary", "argos"],
        default=None,
        help="Translation backend (default dictionary, or backend in dubber.toml).",
    )
    p_dub_multi.add_argument("--glossary", help="JSON glossary file of term overrides.")
    p_dub_multi.add_argument(
        "--max-chars",
        type=int,
        help="Split long translated segments into cues of at most this many chars.",
    )
    p_dub_multi.add_argument("--original", help="Original audio WAV (for loudness matching).")
    p_dub_multi.add_argument(
        "--bed-level",
        type=float,
        default=None,
        help="Keep the original's music bed under each dub at this mix level "
        "(0-1; needs --original).",
    )
    p_dub_multi.add_argument(
        "--enhance",
        action="store_true",
        help="Enhance each finished dub track (hum cut, noise gate, loudness "
        "normalize to -16 dBFS); timing is preserved, no trimming.",
    )
    p_dub_multi.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Skip the per-target .srt/.vtt subtitle files.",
    )
    p_dub_multi.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the whole run when one target fails (default: record the "
        "failure in the summary and carry on with the next language).",
    )
    p_dub_multi.add_argument("--out-dir", required=True, help="Output directory for the bundles.")
    p_dub_multi.add_argument(
        "--stem",
        default="dub",
        help="File stem for per-target outputs: <stem>.<target>.wav etc. (default dub).",
    )

    p_patch = sub.add_parser(
        "patch", help="Re-dub one segment of an existing dub (new text, same window)."
    )
    p_patch.add_argument("--dub", required=True, help="Existing dubbed WAV file.")
    p_patch.add_argument("--transcript", required=True, help="Translated transcript JSON file.")
    p_patch.add_argument("--profile", required=True, help="Speaker profile JSON path.")
    p_patch.add_argument(
        "--profiles",
        help="JSON file mapping speaker names to profile JSON paths, for "
        "multi-speaker transcripts (unlisted speakers use --profile).",
    )
    p_patch.add_argument(
        "--segment",
        type=int,
        required=True,
        help="Segment to re-dub (1-based, among timed segments — the report's numbering).",
    )
    p_patch.add_argument("--text", required=True, help="Replacement text for the segment.")
    p_patch.add_argument("--original", help="Original audio WAV (for loudness matching).")
    p_patch.add_argument(
        "--cache-dir",
        help="Cache the re-rendered segment here so patching the same text "
        "again is instant (default: no cache).",
    )
    p_patch.add_argument("--out", required=True, help="Output patched WAV path.")
    p_patch.add_argument("--transcript-out", help="Write the patched transcript JSON here.")
    p_patch.add_argument(
        "--report", help="Saved dub report JSON (from dub --report-out) to refresh."
    )
    p_patch.add_argument(
        "--report-out",
        help="Write the refreshed report JSON here (default: overwrite --report).",
    )
    p_patch.add_argument(
        "--subtitles",
        action="store_true",
        help="Also write <out-stem>.srt, <out-stem>.vtt and <out-stem>.ass, "
        "retimed to the patched audio (cues match the placed speech, not "
        "the original windows).",
    )
    p_patch.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the fit factor, drift and verdict the patch would produce, "
        "without writing any files.",
    )

    p_report = sub.add_parser(
        "report",
        help="Re-render a saved JSON quality report as HTML or Markdown.",
    )
    p_report.add_argument("input", help="Saved dub report JSON file (from dub --report-out).")
    p_report.add_argument(
        "--format",
        choices=["json", "md", "html"],
        default="html",
        help="Output format (default html).",
    )
    p_report.add_argument("--out", help="Write the re-rendered report here (default: print).")

    p_enhance = sub.add_parser(
        "enhance", help="Clean up a WAV file (trim, de-hum, gate, normalize)."
    )
    p_enhance.add_argument("input", help="Input WAV file.")
    p_enhance.add_argument("--out", required=True, help="Output WAV path.")
    p_enhance.add_argument(
        "--target",
        type=float,
        default=None,
        help=f"Target RMS in dBFS (default {DEFAULT_TARGET_DBFS}, "
        "or enhance.target in dubber.toml).",
    )
    p_enhance.add_argument(
        "--ceiling",
        type=float,
        default=None,
        help=f"Peak ceiling in dBFS (default {DEFAULT_CEILING_DBFS}, "
        "or enhance.ceiling in dubber.toml).",
    )
    p_enhance.add_argument(
        "--high-pass",
        type=float,
        default=None,
        help=f"High-pass cutoff in Hz, 0 disables (default {DEFAULT_HIGH_PASS_HZ}, "
        "or enhance.high_pass in dubber.toml).",
    )
    p_enhance.add_argument(
        "--gate",
        type=float,
        default=None,
        help=f"Noise-gate threshold in dBFS (default {DEFAULT_GATE_THRESHOLD_DBFS}, "
        "or enhance.gate in dubber.toml).",
    )
    p_enhance.add_argument(
        "--no-trim",
        action="store_true",
        help="Keep leading/trailing silence (needed for timed dub tracks).",
    )
    p_enhance.add_argument("--report-out", help="Write the enhancement report (Markdown) here.")

    p_preflight = sub.add_parser(
        "preflight", help="Check a source WAV before dubbing (clipping, silence, SNR)."
    )
    p_preflight.add_argument("input", help="Input WAV file to check.")
    p_preflight.add_argument("--json", action="store_true", help="Print the report as JSON.")
    p_preflight.add_argument(
        "--report-out", help="Write the pre-flight report (Markdown, or JSON with --json) here."
    )
    p_preflight.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Exit 1 on warnings too, not just failures.",
    )

    p_split = sub.add_parser("split", help="Split long transcript segments at silence gaps.")
    p_split.add_argument("transcript", help="Transcript JSON file to split.")
    p_split.add_argument("--out", required=True, help="Output split transcript JSON path.")
    p_split.add_argument(
        "--audio",
        help="Source WAV: VAD silence gaps confirm split points the word "
        "timings hide (optional; without it, word-timing pauses are used).",
    )
    p_split.add_argument(
        "--max-duration",
        type=float,
        default=8.0,
        help="Split any segment longer than this, in seconds (default 8.0).",
    )
    p_split.add_argument(
        "--min-gap",
        type=float,
        default=0.4,
        help="A pause at least this long is a clean split point (default 0.4s).",
    )
    p_split.add_argument(
        "--min-segment",
        type=float,
        default=1.0,
        help="Never leave a piece shorter than this when a choice exists (default 1.0s).",
    )
    p_split.add_argument("--json", action="store_true", help="Print the report as JSON.")
    p_split.add_argument(
        "--report-out", help="Write the split report (Markdown, or JSON with --json) here."
    )

    p_merge = sub.add_parser(
        "merge", help="Join over-split transcript segments back into readable lines."
    )
    p_merge.add_argument("transcript", help="Transcript JSON file to merge.")
    p_merge.add_argument("--out", required=True, help="Output merged transcript JSON path.")
    p_merge.add_argument(
        "--max-gap",
        type=float,
        default=0.4,
        help="Join across a pause at most this long, in seconds (default 0.4).",
    )
    p_merge.add_argument(
        "--max-duration",
        type=float,
        default=8.0,
        help="Never produce a merged segment longer than this, in seconds (default 8.0).",
    )
    p_merge.add_argument("--json", action="store_true", help="Print the report as JSON.")
    p_merge.add_argument(
        "--report-out", help="Write the merge report (Markdown, or JSON with --json) here."
    )

    p_compare = sub.add_parser(
        "compare", help="Compare two dub quality reports (JSON) metric by metric."
    )
    p_compare.add_argument("report_a", help="First dub quality report (JSON).")
    p_compare.add_argument("report_b", help="Second dub quality report (JSON).")
    p_compare.add_argument("--label-a", default="A", help="Name for the first dub (default A).")
    p_compare.add_argument("--label-b", default="B", help="Name for the second dub (default B).")
    p_compare.add_argument("--json", action="store_true", help="Print the comparison as JSON.")
    p_compare.add_argument(
        "--report-out", help="Write the comparison (Markdown, or JSON with --json) here."
    )

    p_estimate = sub.add_parser(
        "estimate", help="Predict a dub's durations, wall time and cost before running it."
    )
    p_estimate.add_argument("transcript", help="Translated transcript JSON file to estimate.")
    p_estimate.add_argument("--target", help="Target language code (recorded in the estimate).")
    p_estimate.add_argument(
        "--backend",
        default="scripted",
        help=f"TTS backend (known: {', '.join(BACKEND_RTF)}; unknown backends need --rtf).",
    )
    p_estimate.add_argument(
        "--profile",
        help="Speaker profile JSON path (sets the speaking rate; needed with --cache-dir).",
    )
    p_estimate.add_argument(
        "--wpm",
        type=float,
        default=None,
        help="Speaking rate in words per minute (default: the profile's rate, else 150).",
    )
    p_estimate.add_argument(
        "--rtf",
        type=float,
        default=None,
        help="Real-time factor: seconds of audio rendered per second of wall time "
        "(overrides the backend default).",
    )
    p_estimate.add_argument(
        "--cache-dir",
        help="Synthesis cache directory: segments already cached there are "
        "counted as hits and excluded from wall time and cost (needs --profile).",
    )
    p_estimate.add_argument(
        "--price-per-char",
        type=float,
        default=None,
        help="Backend price per character (mutually exclusive with --price-per-minute).",
    )
    p_estimate.add_argument(
        "--price-per-minute",
        type=float,
        default=None,
        help="Backend price per minute of synthesised speech.",
    )
    p_estimate.add_argument(
        "--currency", default="USD", help="Currency for the cost (default USD)."
    )
    p_estimate.add_argument(
        "--format", choices=["json", "md"], default="md", help="Estimate format (default md)."
    )
    p_estimate.add_argument("--out", help="Write the estimate to this file instead of stdout.")

    p_render = sub.add_parser(
        "render", help="Render a dubbed video, burning subtitles into the picture."
    )
    p_render.add_argument("video", help="Original video to render from.")
    p_render.add_argument("audio", help="Dubbed audio track (WAV) to put on the video.")
    p_render.add_argument("--subtitles", help="Retimed SRT file to burn into the picture.")
    p_render.add_argument("--out", required=True, help="Output video path (MP4).")
    p_render.add_argument(
        "--font-size", type=int, default=24, help="Subtitle font size in px (default 24)."
    )
    p_render.add_argument("--font", help="Subtitle font family (default: libass default).")
    p_render.add_argument("--bold", action="store_true", help="Render subtitles in bold.")
    p_render.add_argument(
        "--alignment",
        type=int,
        default=2,
        help="Subtitle position, ASS numpad (default 2 = bottom-centre).",
    )
    p_render.add_argument(
        "--margin-v", type=int, default=24, help="Subtitle margin from the bottom edge in px."
    )
    p_render.add_argument("--crf", type=int, default=23, help="x264 quality, 0-51 (default 23).")
    p_render.add_argument(
        "--preset",
        default="veryfast",
        choices=[
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
            "placebo",
        ],
        help="x264 encoding preset (default veryfast).",
    )
    p_render.add_argument(
        "--preview-start",
        type=float,
        default=None,
        help="Render only a slice, starting at this many seconds in.",
    )
    p_render.add_argument(
        "--preview-duration",
        type=float,
        default=None,
        help="Preview slice length in seconds (default 10; needs --preview-start).",
    )
    p_render.add_argument("--report-out", help="Write the render report to this file.")
    p_render.add_argument(
        "--report-format",
        choices=["json", "md"],
        default="json",
        help="Render report format (default json).",
    )

    p_export = sub.add_parser(
        "export",
        help="Pack a finished dub into a portable bundle (directory or .zip).",
    )
    p_export.add_argument(
        "dub_audio",
        nargs="?",
        help="Dubbed audio track (WAV) to export (not needed with --verify).",
    )
    p_export.add_argument("--target", help="Target language code (e.g. hi).")
    p_export.add_argument("--source", default="en", help="Source language code (default en).")
    p_export.add_argument("--subtitles", help="Retimed SRT subtitles to include.")
    p_export.add_argument("--subtitles-vtt", help="Retimed WebVTT subtitles to include.")
    p_export.add_argument("--report", help="Dub quality report JSON to include.")
    p_export.add_argument("--original", help="Original video/audio to include.")
    p_export.add_argument("--video", help="Rendered dubbed video (MP4) to include.")
    p_export.add_argument("--speakers", help="Comma-separated speaker names for the manifest.")
    p_export.add_argument(
        "--out", help="Bundle directory, or .zip path (default format follows --out)."
    )
    p_export.add_argument(
        "--format",
        choices=["dir", "zip"],
        default=None,
        help="Bundle format (default: zip when --out ends in .zip, else a directory).",
    )
    p_export.add_argument(
        "--name", help="Bundle name (default: dubbed-audio stem + target language)."
    )
    p_export.add_argument(
        "--force", action="store_true", help="Overwrite a non-empty directory / file."
    )
    p_export.add_argument(
        "--verify",
        help="Verify a directory bundle against its manifest instead of exporting.",
    )

    p_batch = sub.add_parser("batch", help="Dub many jobs from a manifest JSON file.")
    p_batch.add_argument("manifest", help="Manifest JSON file (see README for the format).")
    p_batch.add_argument("--out-dir", required=True, help="Output directory; one subdir per job.")
    p_batch.add_argument(
        "--force",
        action="store_true",
        help="Re-run jobs whose dubbed WAV already exists.",
    )
    p_batch.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch after the first failed job.",
    )

    p_config = sub.add_parser("config", help="Show the effective configuration.")
    p_config.add_argument(
        "--format",
        choices=["json", "toml"],
        default="json",
        help="Output format (default json).",
    )

    p_watch = sub.add_parser(
        "watch", help="Dub every new video that lands in a folder, as it appears."
    )
    p_watch.add_argument("watch_dir", help="Folder to watch for new media files.")
    p_watch.add_argument("--out-dir", required=True, help="Output directory; one subdir per job.")
    p_watch.add_argument(
        "--target",
        help="Target language code (default: target_language from dubber.toml).",
    )
    p_watch.add_argument("--profile", required=True, help="Speaker profile JSON path.")
    p_watch.add_argument(
        "--source",
        help="Source language code (default: source_language from dubber.toml).",
    )
    p_watch.add_argument(
        "--backend",
        choices=["whisper", "scripted"],
        default="whisper",
        help="Transcription backend (default whisper).",
    )
    p_watch.add_argument("--model", default="tiny", help="Whisper model (default tiny).")
    p_watch.add_argument("--script", help="Scripted-backend word list (JSON).")
    p_watch.add_argument("--glossary", help="Glossary JSON file (pinned translations).")
    p_watch.add_argument("--max-chars", type=int, help="Max characters per subtitle cue.")
    p_watch.add_argument(
        "--no-mux",
        action="store_true",
        help="Skip muxing the dub back into the video.",
    )
    p_watch.add_argument(
        "--profiles",
        help="JSON file mapping speaker names to profile JSON paths, for "
        "multi-speaker videos (unlisted speakers use --profile).",
    )
    p_watch.add_argument(
        "--bed-level",
        type=float,
        default=None,
        help="Keep the video's music bed under the dub at this mix level "
        "(0-1; default: bed_level from dubber.toml).",
    )
    p_watch.add_argument(
        "--enhance",
        action="store_true",
        help="Enhance each finished dub track (hum cut, noise gate, loudness "
        "normalize to -16 dBFS); timing is preserved, no trimming.",
    )
    p_watch.add_argument(
        "--state",
        help="State file path (default <out-dir>/.dubber-watch.json).",
    )
    p_watch.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Seconds between scans (default 60).",
    )
    p_watch.add_argument(
        "--once",
        action="store_true",
        help="Scan once and exit instead of polling forever.",
    )
    p_watch.add_argument(
        "--scans",
        type=int,
        default=None,
        help="Stop after this many scans (for scripting; --once is 1 scan).",
    )
    p_watch.add_argument(
        "--extensions",
        help="Comma-separated media extensions to watch "
        "(default mp4,mkv,mov,webm,avi,wav,mp3,flac,m4a,ogg).",
    )
    return parser


def _load_config(args: argparse.Namespace) -> DubberConfig | None:
    """Load the config file for this invocation, or None on error.

    A bad config file is a hard error (exit 2), never a silent
    fall-back to defaults — a typo in dubber.toml should not quietly
    change the dub. No file found anywhere just means defaults.
    """
    try:
        return load_config(getattr(args, "config", None))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None


def _cmd_config(args: argparse.Namespace) -> int:
    config = _load_config(args)
    if config is None:
        return 2
    if args.format == "toml":
        print(_config_to_toml(config))
    else:
        import json

        print(json.dumps(config.to_dict(), indent=2))
    if config.path is not None:
        print(f"(loaded from {config.path})")
    else:
        print("(no config file found; showing defaults)")
    return 0


def _config_to_toml(config: DubberConfig) -> str:
    """Render the effective config as TOML (stdlib has no TOML writer)."""
    import json

    lines = ["[dubber]"]
    for key in ("target_language", "source_language", "backend", "voice_library"):
        value = getattr(config, key)
        if value is not None:
            lines.append(f"{key} = {json.dumps(value)}")
    if config.bed_level is not None:
        lines.append(f"bed_level = {config.bed_level}")
    lines.append("")
    lines.append("[dubber.enhance]")
    for key in ("target", "ceiling", "high_pass", "gate"):
        lines.append(f"{key} = {getattr(config.enhance, key)}")
    return "\n".join(lines)


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
        "ass": to_ass(transcript),
    }[args.format]
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def _cmd_translate(args: argparse.Namespace) -> int:
    config = _load_config(args)
    if config is None:
        return 2
    target = args.target or config.target_language
    if not target:
        print(
            "error: translate needs --target or target_language in dubber.toml",
            file=sys.stderr,
        )
        return 2
    with open(args.transcript, encoding="utf-8") as fh:
        transcript = Transcript.from_json(fh.read())
    source = args.source or transcript.language or config.source_language or "en"
    backend = args.backend or config.backend
    if backend == "argos":
        translator = ArgosTranslator()
    else:
        translator = DictionaryTranslator(source=source, target=target)
    glossary = Glossary.from_json_file(args.glossary) if args.glossary else None
    translated = translate_transcript(
        transcript,
        translator,
        target=target,
        source=source,
        glossary=glossary,
        max_chars=args.max_chars,
    )
    stats = (
        f"{translated.word_count} words, {len(translated.segments)} segments, "
        f"{source} -> {target} via {translator.name}"
    )
    if isinstance(translator, DictionaryTranslator):
        stats += f", dictionary coverage {translator.last_coverage:.0%}"
        if translator.last_unknown:
            stats += f" (untranslated: {', '.join(translator.last_unknown[:5])})"
    print(stats)
    if args.round_trip:
        if not 0.0 <= args.fidelity_threshold <= 1.0:
            print(
                "error: --fidelity-threshold must be between 0 and 1, "
                f"got {args.fidelity_threshold}",
                file=sys.stderr,
            )
            return 2
        try:
            report = check_fidelity(
                transcript,
                translated,
                translator,
                target=target,
                source=source,
                threshold=args.fidelity_threshold,
            )
        except FidelityError as exc:
            print(f"error: fidelity check failed: {exc}", file=sys.stderr)
            return 2
        print(
            f"fidelity: verdict {report.verdict} "
            f"(mean {report.mean_score:.2f}, min {report.min_score:.2f}, "
            f"{len(report.flagged)} of {len(report.segments)} segments flagged)"
        )
        for segment in report.flagged:
            print(
                f"  segment {segment.index} (score {segment.score:.2f}): "
                f"“{segment.source_text}” -> back “{segment.back_text}”",
                file=sys.stderr,
            )
        if args.fidelity_out:
            text = {"json": report.to_json(), "md": report.to_markdown()}[args.fidelity_format]
            with open(args.fidelity_out, "w", encoding="utf-8") as fh:
                fh.write(text)
            print(f"wrote {args.fidelity_out}")
    if args.out:
        text = {
            "json": translated.to_json(),
            "srt": translated.to_srt(),
            "vtt": translated.to_vtt(),
        }[args.format]
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        for segment in translated.segments:
            print(f"[{segment.start:7.2f} - {segment.end:7.2f}] {segment.text}")
    return 0


def _cmd_clone(args: argparse.Namespace) -> int:
    config = _load_config(args)
    if config is None:
        return 2
    library_path = args.library or config.voice_library
    cloner = ScriptedVoiceCloner()
    audios = [resample(load_wav(path), DEFAULT_SAMPLE_RATE) for path in args.references]
    profile = cloner.clone(args.name, audios, reference_files=list(args.references))
    if args.wpm:
        profile = adjust_prosody(profile, words_per_minute=args.wpm)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(profile.to_json())
    print(
        f"cloned {profile.name}: base pitch {profile.base_pitch_hz:.0f} Hz, "
        f"{profile.speaking_rate_wpm:.0f} wpm, {len(args.references)} reference(s)"
    )
    print(f"wrote {args.out}")
    if library_path:
        import os

        library = (
            VoiceLibrary.from_json_file(library_path)
            if os.path.exists(library_path)
            else VoiceLibrary()
        )
        library.add(profile, overwrite=True)
        library.save(library_path)
        print(f"added {profile.name} to library {library_path} ({len(library)} voices)")
    return 0


def _cmd_voices(args: argparse.Namespace) -> int:
    import os

    if not os.path.exists(args.library):
        if args.add is None:
            print(f"no voice library at {args.library} yet — clone a voice first")
            return 0
        library = VoiceLibrary()
    else:
        library = VoiceLibrary.from_json_file(args.library)
    if args.add:
        with open(args.add, encoding="utf-8") as fh:
            profile = SpeakerProfile.from_json(fh.read())
        library.add(profile, overwrite=args.overwrite)
        library.save(args.library)
        print(f"added {profile.name} to {args.library}")
    if len(library) == 0:
        print(f"{args.library}: no voices yet")
        return 0
    print(f"{args.library}: {len(library)} voice(s)")
    for name in library.names():
        profile = library.get(name)
        print(f"  {name}: {profile.base_pitch_hz:.0f} Hz, {profile.speaking_rate_wpm:.0f} wpm")
    return 0


def _cmd_diarize(args: argparse.Namespace) -> int:
    config = _load_config(args)
    if config is None:
        return 2
    library_path = args.library or config.voice_library
    if not library_path:
        print(
            "error: diarize needs --library or voice_library in dubber.toml",
            file=sys.stderr,
        )
        return 2
    with open(args.transcript, encoding="utf-8") as fh:
        transcript = Transcript.from_json(fh.read())
    audio = resample(load_wav(args.audio), DEFAULT_SAMPLE_RATE)
    library = VoiceLibrary.from_json_file(library_path)
    labelled, report = diarize_transcript(
        transcript,
        audio,
        library,
        threshold=args.threshold,
        overwrite=args.overwrite,
    )
    for label in report.labels:
        speaker = label.speaker or "unknown"
        similarity = f" (similarity {label.similarity:.3f})" if label.similarity is not None else ""
        print(f"[{label.start:7.2f} - {label.end:7.2f}] {speaker}{similarity} — {label.text}")
    print(
        f"{len(report.labels)} segments: {report.matched} matched, "
        f"{report.kept} kept, {report.unknown} unknown"
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(labelled.to_json())
        print(f"wrote {args.out}")
    else:
        print(labelled.to_json())
    if args.report_out:
        text = report.to_json() if args.report_format == "json" else report.to_markdown()
        with open(args.report_out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.report_out}")
    return 0


def _cached_cloner(args: argparse.Namespace) -> CachedVoiceCloner | ScriptedVoiceCloner:
    """The voice cloner for a command, wrapped in the synthesis cache when asked.

    ``--cache-dir`` opts a command into caching; without it the plain
    scripted cloner runs, so existing behaviour (and tests) are
    untouched. The returned object always satisfies the VoiceCloner
    protocol, so callers never branch on which one they got.
    """
    inner = ScriptedVoiceCloner()
    cache_dir = getattr(args, "cache_dir", None)
    if cache_dir is None:
        return inner
    return CachedVoiceCloner(inner, SynthesisCache(cache_dir))


def _cache_summary(cloner: CachedVoiceCloner | ScriptedVoiceCloner) -> str | None:
    """One-line cache report for a command that used a cached cloner."""
    if not isinstance(cloner, CachedVoiceCloner):
        return None
    return (
        f"cache: {cloner.hits} hit(s) / {cloner.misses} miss(es) "
        f"({cloner.cache.stats().hit_rate:.2f} hit rate, "
        f"{cloner.cache.stats().entries} segment(s) stored)"
    )


def _cmd_cache(args: argparse.Namespace) -> int:
    cache = SynthesisCache(args.cache_dir or default_cache_dir())
    if args.clear:
        removed = cache.clear()
        print(f"cleared {removed} cached segment(s) from {cache.directory}")
        return 0
    print(f"cache directory: {cache.directory}")
    print(cache.stats().to_markdown(), end="")
    return 0


def _cmd_synthesize(args: argparse.Namespace) -> int:
    with open(args.profile, encoding="utf-8") as fh:
        profile = SpeakerProfile.from_json(fh.read())
    if args.semitones or args.wpm:
        profile = adjust_prosody(profile, semitones=args.semitones, words_per_minute=args.wpm)
    cloner = _cached_cloner(args)
    audio = cloner.synthesize(args.text, profile)
    save_wav(args.out, audio)
    print(
        f"synthesized {len(args.text.split())} words in {profile.name}'s voice: "
        f"{audio.duration:.2f}s -> {args.out}"
    )
    if isinstance(cloner, CachedVoiceCloner):
        print(f"cache {'hit' if cloner.hits else 'miss'} ({_cache_summary(cloner)})")
    return 0


def _load_profiles_map(path: str) -> dict[str, SpeakerProfile]:
    """Load a speaker-name -> profile-file JSON map into real profiles."""
    import json
    import os

    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except OSError as exc:
        raise VoiceCloneError(f"cannot read profiles map {path!r}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise VoiceCloneError(f"profiles map {path!r} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise VoiceCloneError(f"profiles map {path!r} must be a JSON object")
    profiles: dict[str, SpeakerProfile] = {}
    for speaker, profile_path in raw.items():
        if not isinstance(profile_path, str):
            raise VoiceCloneError(
                f"profiles map {path!r}: path for speaker {speaker!r} is not a string"
            )
        resolved = profile_path
        if not os.path.isabs(resolved):
            resolved = os.path.join(os.path.dirname(os.path.abspath(path)), resolved)
        try:
            with open(resolved, encoding="utf-8") as fh:
                profiles[speaker] = SpeakerProfile.from_json(fh.read())
        except OSError as exc:
            raise VoiceCloneError(
                f"cannot read profile for speaker {speaker!r} ({resolved}): {exc}"
            ) from exc
    return profiles


def _cmd_dub(args: argparse.Namespace) -> int:
    config = _load_config(args)
    if config is None:
        return 2
    bed_level = args.bed_level if args.bed_level is not None else config.bed_level
    if args.video and not args.mux_out:
        print("error: --video needs --mux-out", file=sys.stderr)
        return 2
    if args.burn_subtitles and not args.video:
        print("error: --burn-subtitles needs --video/--mux-out", file=sys.stderr)
        return 2
    if bed_level is not None and not args.original:
        print("error: --bed-level needs --original", file=sys.stderr)
        return 2
    if bed_level is not None and not 0.0 <= bed_level <= 1.0:
        print("error: --bed-level must be between 0 and 1", file=sys.stderr)
        return 2
    with open(args.transcript, encoding="utf-8") as fh:
        transcript = Transcript.from_json(fh.read())
    with open(args.profile, encoding="utf-8") as fh:
        profile = SpeakerProfile.from_json(fh.read())
    profiles = _load_profiles_map(args.profiles) if args.profiles else None
    original = load_wav(args.original) if args.original else None
    if args.preflight:
        if original is None:
            print("error: --preflight needs --original", file=sys.stderr)
            return 2
        report = run_preflight(original)
        print(report.to_markdown(), end="")
        if report.verdict == "fail":
            print("pre-flight failed — dub aborted", file=sys.stderr)
            return 1
    if args.max_segment_duration is not None and not args.split_segments:
        print("error: --max-segment-duration needs --split-segments", file=sys.stderr)
        return 2
    original = resample(original, DEFAULT_SAMPLE_RATE) if original is not None else None
    reference = None
    if args.reference:
        with open(args.reference, encoding="utf-8") as fh:
            reference = Transcript.from_json(fh.read())
    if args.split_segments:
        split_config = SplitConfig(
            max_duration=args.max_segment_duration if args.max_segment_duration is not None else 8.0
        )
        transcript, split_report = split_transcript(transcript, original, split_config)
        print(split_report.to_markdown(), end="")
    cloner = _cached_cloner(args)
    dub_audio, fit_factors, placed = dub_transcript(
        transcript,
        cloner,
        profile,
        original,
        profiles=profiles,
        bed_level=bed_level,
    )
    enhancement = None
    if args.enhance:
        # trim=False: the dub sits on a timed bed, so its duration and
        # segment start times must not move.
        dub_audio, enhancement = enhance_audio(dub_audio, trim=False)
    save_wav(args.out, dub_audio)
    report = build_report(
        transcript,
        placed,
        fit_factors,
        dub_audio,
        original,
        reference_transcript=reference,
        bed_level=bed_level,
        enhancement=enhancement,
    )
    print(f"dubbed {report.segments_dubbed} segments, {dub_audio.duration:.2f}s -> {args.out}")
    summary = _cache_summary(cloner)
    if summary is not None:
        print(summary)
    print(
        f"sync drift mean {report.mean_sync_drift:.3f}s / max {report.max_sync_drift:.3f}s, "
        f"mean fit {report.mean_fit_factor:.2f}, MOS proxy {report.mos_proxy:.2f}, "
        f"verdict: {report.verdict}"
    )
    if report.word_error_rate is not None:
        print(f"word error rate vs reference: {report.word_error_rate:.3f}")
    if report.bed_level is not None:
        print(f"background bed level: {report.bed_level:.2f}")
    if enhancement is not None:
        print(
            f"enhanced: RMS {enhancement.before.rms_dbfs:.1f} -> "
            f"{enhancement.after.rms_dbfs:.1f} dBFS "
            f"(gain {enhancement.gain_db:+.1f} dB)"
        )
    if args.report_out:
        formats = {
            "json": report.to_json(),
            "md": report.to_markdown(),
            "html": render_html_report(report),
        }
        with open(args.report_out, "w", encoding="utf-8") as fh:
            fh.write(formats[args.report_format])
        print(f"wrote {args.report_out}")
    if report.speakers_dubbed:
        print(f"speakers dubbed: {report.speakers_dubbed}")
    srt_path = None
    if args.subtitles or args.burn_subtitles:
        import os

        stem, _ = os.path.splitext(args.out)
        # Cues follow the dubbed audio: a clamped stretch ends the speech
        # early, so original-window timings would linger over silence.
        synced = retime_transcript(transcript, placed)
        for ext, text in (
            ("srt", synced.to_srt()),
            ("vtt", synced.to_vtt()),
            ("ass", to_ass(synced)),
        ):
            path = f"{stem}.{ext}"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            print(f"wrote {path}")
        srt_path = f"{stem}.srt"
    if args.video:
        if args.burn_subtitles:
            report = render_video(args.video, args.out, args.mux_out, subtitles_path=srt_path)
            print(
                f"rendered {args.mux_out} with {report.subtitle_cues} subtitle cue(s) "
                f"burned in ({report.duration:.2f}s)"
            )
        else:
            mux_audio(args.video, args.out, args.mux_out)
            print(f"muxed dub into {args.mux_out}")
    return 0


def _cmd_dub_multi(args: argparse.Namespace) -> int:
    config = _load_config(args)
    if config is None:
        return 2
    bed_level = args.bed_level if args.bed_level is not None else config.bed_level
    if bed_level is not None and not args.original:
        print("error: --bed-level needs --original", file=sys.stderr)
        return 2
    if bed_level is not None and not 0.0 <= bed_level <= 1.0:
        print("error: --bed-level must be between 0 and 1", file=sys.stderr)
        return 2
    try:
        targets = parse_targets(args.targets)
    except MultiTargetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    with open(args.transcript, encoding="utf-8") as fh:
        transcript = Transcript.from_json(fh.read())
    with open(args.profile, encoding="utf-8") as fh:
        profile = SpeakerProfile.from_json(fh.read())
    profiles = _load_profiles_map(args.profiles) if args.profiles else None
    original = resample(load_wav(args.original), DEFAULT_SAMPLE_RATE) if args.original else None
    source = args.source or transcript.language or config.source_language or "en"
    backend = args.backend or config.backend
    glossary = Glossary.from_json_file(args.glossary) if args.glossary else None

    def make_translator(target: str):
        if backend == "argos":
            return ArgosTranslator()
        return DictionaryTranslator(source=source, target=target)

    summary = dub_all_targets(
        transcript,
        targets,
        make_translator,
        ScriptedVoiceCloner(),
        profile,
        args.out_dir,
        args.stem,
        source=source,
        glossary=glossary,
        max_chars=args.max_chars,
        original=original,
        profiles=profiles,
        bed_level=bed_level,
        enhance=args.enhance,
        subtitles=not args.no_subtitles,
        stop_on_error=args.stop_on_error,
    )
    summary_path = os.path.join(args.out_dir, f"{args.stem}.multilang.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(summary.to_json())
    md_path = os.path.join(args.out_dir, f"{args.stem}.multilang.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(summary.to_markdown())
    for result in summary.results:
        print(
            f"{result.target}: {result.report.verdict} "
            f"(MOS proxy {result.report.mos_proxy:.2f}, "
            f"mean drift {result.report.mean_sync_drift:.3f}s, "
            f"mean fit {result.report.mean_fit_factor:.2f}) -> {result.dub_path}"
        )
    for target, error in summary.failures.items():
        print(f"{target}: FAILED — {error}", file=sys.stderr)
    print(f"dubbed {summary.succeeded} of {len(summary.targets)} targets")
    print(f"wrote {summary_path}")
    print(f"wrote {md_path}")
    return 0 if not summary.failures else 1


def _cmd_patch(args: argparse.Namespace) -> int:
    with open(args.transcript, encoding="utf-8") as fh:
        transcript = Transcript.from_json(fh.read())
    with open(args.profile, encoding="utf-8") as fh:
        profile = SpeakerProfile.from_json(fh.read())
    profiles = _load_profiles_map(args.profiles) if args.profiles else None
    dub_audio = load_wav(args.dub)
    original = resample(load_wav(args.original), DEFAULT_SAMPLE_RATE) if args.original else None
    report = None
    if args.report:
        with open(args.report, encoding="utf-8") as fh:
            report = DubReport.from_json(fh.read())
    # Placed spans from the report when one is given (exact, including
    # clamped clips); otherwise the segment windows.
    placed: list[tuple[float, float]] | None = None
    if report is not None:
        placed = [
            (
                row.start,
                row.placed_duration if row.placed_duration is not None else row.end - row.start,
            )
            for row in report.segments
        ]
    patched_transcript = patch_transcript(transcript, args.segment, args.text)
    cloner = _cached_cloner(args)
    patched_audio, factor, placed_duration = patch_dub(
        dub_audio,
        transcript,
        args.segment,
        args.text,
        cloner,
        profile,
        original,
        profiles=profiles,
        placed=placed,
    )
    segment = [s for s in patched_transcript.segments if s.words][args.segment - 1]
    drift = abs((segment.start + placed_duration) - segment.end)
    refreshed = None
    if report is not None:
        row = SegmentRow(
            index=args.segment,
            start=segment.start,
            end=segment.end,
            drift=drift,
            fit_factor=factor,
            speaker=segment.speaker,
            text=args.text,
            placed_duration=placed_duration,
        )
        refreshed = refresh_report(report, row)
    if args.dry_run:
        # Nothing is written: the caller only learns what the patch
        # would do (a patch that needs an extreme stretch is worth
        # knowing about before it lands).
        print(
            f"dry run: segment {args.segment} would fit at {factor:.2f} "
            f"(placed {placed_duration:.2f}s, drift {drift:.3f}s)"
        )
        if refreshed is not None:
            print(
                f"dry run: report verdict would be {refreshed.verdict} "
                f"(MOS proxy {refreshed.mos_proxy:.2f})"
            )
        return 0
    save_wav(args.out, patched_audio)
    print(
        f"patched segment {args.segment} (fit {factor:.2f}, "
        f"placed {placed_duration:.2f}s) -> {args.out}"
    )
    summary = _cache_summary(cloner)
    if summary is not None:
        print(summary)
    if args.transcript_out:
        with open(args.transcript_out, "w", encoding="utf-8") as fh:
            fh.write(patched_transcript.to_json())
        print(f"wrote {args.transcript_out}")
    if refreshed is not None:
        report_out = args.report_out or args.report
        with open(report_out, "w", encoding="utf-8") as fh:
            fh.write(refreshed.to_json())
        print(f"wrote {report_out} (verdict: {refreshed.verdict})")
    if args.subtitles:
        import os

        timed = [s for s in patched_transcript.segments if s.words]
        spans = list(placed) if placed is not None else [(s.start, s.end - s.start) for s in timed]
        spans[args.segment - 1] = (timed[args.segment - 1].start, placed_duration)
        stem, _ = os.path.splitext(args.out)
        synced = retime_transcript(patched_transcript, spans)
        for ext, text in (
            ("srt", synced.to_srt()),
            ("vtt", synced.to_vtt()),
            ("ass", to_ass(synced)),
        ):
            path = f"{stem}.{ext}"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            print(f"wrote {path}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    with open(args.input, encoding="utf-8") as fh:
        report = DubReport.from_json(fh.read())
    formats = {
        "json": report.to_json(),
        "md": report.to_markdown(),
        "html": render_html_report(report),
    }
    text = formats[args.format]
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def _cmd_enhance(args: argparse.Namespace) -> int:
    config = _load_config(args)
    if config is None:
        return 2
    audio = load_wav(args.input)
    enhanced, report = enhance_audio(
        audio,
        target_dbfs=args.target if args.target is not None else config.enhance.target,
        ceiling_dbfs=args.ceiling if args.ceiling is not None else config.enhance.ceiling,
        high_pass_hz=args.high_pass if args.high_pass is not None else config.enhance.high_pass,
        gate_threshold_dbfs=args.gate if args.gate is not None else config.enhance.gate,
        trim=not args.no_trim,
    )
    save_wav(args.out, enhanced)
    print(report.to_markdown(), end="")
    print(f"wrote {args.out} ({enhanced.duration:.2f}s)")
    if args.report_out:
        with open(args.report_out, "w", encoding="utf-8") as fh:
            fh.write(report.to_markdown())
        print(f"wrote {args.report_out}")
    return 0


def _cmd_split(args: argparse.Namespace) -> int:
    with open(args.transcript, encoding="utf-8") as fh:
        transcript = Transcript.from_json(fh.read())
    audio = load_wav(args.audio) if args.audio else None
    config = SplitConfig(
        max_duration=args.max_duration,
        min_gap=args.min_gap,
        min_segment=args.min_segment,
    )
    split, report = split_transcript(transcript, audio, config)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(split.to_json())
    text = report.to_json() if args.json else report.to_markdown()
    print(text, end="" if text.endswith("\n") else "\n")
    print(f"wrote {args.out} ({report.segments_after} segments)")
    if args.report_out:
        with open(args.report_out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.report_out}")
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    with open(args.transcript, encoding="utf-8") as fh:
        transcript = Transcript.from_json(fh.read())
    try:
        config = MergeConfig(max_gap=args.max_gap, max_duration=args.max_duration)
    except MergeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    merged, report = merge_transcript(transcript, config)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(merged.to_json())
    text = report.to_json() if args.json else report.to_markdown()
    print(text, end="" if text.endswith("\n") else "\n")
    print(f"wrote {args.out} ({report.segments_after} segments)")
    if args.report_out:
        with open(args.report_out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.report_out}")
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    audio = load_wav(args.input)
    report = run_preflight(audio)
    text = report.to_json() if args.json else report.to_markdown()
    print(text, end="" if text.endswith("\n") else "\n")
    if args.report_out:
        with open(args.report_out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.report_out}")
    if report.verdict == "fail":
        return 1
    if report.verdict == "pass_with_warnings" and args.fail_on_warn:
        return 1
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    style = SubtitleStyle(
        font_size=args.font_size,
        font=args.font,
        bold=args.bold,
        alignment=args.alignment,
        margin_v=args.margin_v,
    )
    if args.preview_start is not None or args.preview_duration is not None:
        report = render_preview(
            args.video,
            args.audio,
            args.out,
            subtitles_path=args.subtitles,
            style=style,
            start=args.preview_start if args.preview_start is not None else 0.0,
            duration=args.preview_duration if args.preview_duration is not None else 10.0,
            crf=args.crf,
            preset=args.preset,
        )
    else:
        report = render_video(
            args.video,
            args.audio,
            args.out,
            subtitles_path=args.subtitles,
            style=style,
            crf=args.crf,
            preset=args.preset,
        )
    print(
        f"rendered {report.out_path} ({report.duration:.2f}s, "
        f"{report.filter_used} filter, CRF {report.crf})"
    )
    if report.subtitle_cues:
        print(f"burned in {report.subtitle_cues} subtitle cue(s)")
    if args.report_out:
        text = report.to_json() if args.report_format == "json" else report.to_markdown()
        with open(args.report_out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.report_out}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    if args.verify:
        problems = verify_bundle(args.verify)
        if problems:
            for problem in problems:
                print(f"FAIL: {problem}")
            return 1
        print(f"bundle intact: {args.verify}")
        return 0
    if not args.dub_audio:
        raise ValueError("export needs a dubbed audio file (or --verify <bundle-dir>)")
    if not args.target:
        raise ValueError("export needs --target (the dub's language code)")
    if not args.out:
        raise ValueError("export needs --out (bundle directory or .zip path)")
    name = args.name or f"{Path(args.dub_audio).stem}.{args.target}"
    speakers = [s.strip() for s in args.speakers.split(",") if s.strip()] if args.speakers else None
    manifest, sources = build_manifest(
        name,
        dub_audio=args.dub_audio,
        target=args.target,
        source=args.source,
        subtitles_srt=args.subtitles,
        subtitles_vtt=args.subtitles_vtt,
        report=args.report,
        original=args.original,
        rendered_video=args.video,
        speakers=speakers,
    )
    fmt = args.format or ("zip" if args.out.lower().endswith(".zip") else "dir")
    if fmt == "zip":
        written = export_zip(manifest, sources, args.out, overwrite=args.force)
    else:
        written = export_bundle(manifest, sources, args.out, overwrite=args.force)
    print(manifest.to_markdown(), end="")
    print(f"wrote {written}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    with open(args.report_a, encoding="utf-8") as fh:
        report_a = DubReport.from_json(fh.read())
    with open(args.report_b, encoding="utf-8") as fh:
        report_b = DubReport.from_json(fh.read())
    comparison = compare_reports(report_a, report_b, args.label_a, args.label_b)
    text = comparison.to_json() if args.json else comparison.to_markdown()
    print(text, end="" if text.endswith("\n") else "\n")
    if args.report_out:
        with open(args.report_out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.report_out}")
    return 0


def _cmd_estimate(args: argparse.Namespace) -> int:
    with open(args.transcript, encoding="utf-8") as fh:
        transcript = Transcript.from_json(fh.read())
    profile = None
    if args.profile:
        with open(args.profile, encoding="utf-8") as fh:
            profile = SpeakerProfile.from_json(fh.read())
    wpm = args.wpm
    if wpm is None:
        wpm = profile.speaking_rate_wpm if profile is not None else 150.0
    cache = SynthesisCache(args.cache_dir) if args.cache_dir else None
    estimate = estimate_dub(
        transcript,
        backend=args.backend,
        target=args.target,
        words_per_minute=wpm,
        real_time_factor=args.rtf,
        price_per_char=args.price_per_char,
        price_per_minute=args.price_per_minute,
        currency=args.currency,
        cache=cache,
        profile=profile,
    )
    text = estimate.to_json() if args.format == "json" else estimate.to_markdown()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        print(text, end="" if text.endswith("\n") else "\n")
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    manifest = BatchManifest.from_json_file(args.manifest)
    results = run_batch(manifest, args.out_dir, force=args.force, stop_on_error=args.stop_on_error)
    for result in results:
        if result.status == "ok":
            print(
                f"ok {result.name} (verdict {result.verdict}, "
                f"MOS proxy {result.mos_proxy}, {result.duration_s:.1f}s)"
            )
        elif result.status == "skipped":
            print(f"skipped {result.name} (dubbed WAV already exists, use --force to redo)")
        else:
            print(f"failed {result.name}: {result.error}", file=sys.stderr)
    summary = summarize(results, args.out_dir)
    print(
        f"batch done: {summary.succeeded} ok, {summary.failed} failed, "
        f"{summary.skipped} skipped out of {summary.total}"
    )
    print(
        "reports: "
        f"{args.out_dir}/batch-report.json, {args.out_dir}/batch-report.md, "
        f"{args.out_dir}/batch-report.html"
    )
    return 1 if summary.failed else 0


def _load_profiles_paths(path: str) -> dict[str, str]:
    """Load a speaker-name -> profile-path JSON map (paths, not profiles)."""
    import json

    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except OSError as exc:
        raise VoiceCloneError(f"cannot read profiles map {path!r}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise VoiceCloneError(f"profiles map {path!r} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in raw.items()
    ):
        raise VoiceCloneError(f"profiles map {path!r} must be a JSON object of strings")
    return raw


def _cmd_watch(args: argparse.Namespace) -> int:
    config = _load_config(args)
    if config is None:
        return 2
    target = args.target or config.target_language
    if not target:
        print(
            "error: watch needs --target or target_language in dubber.toml",
            file=sys.stderr,
        )
        return 2
    if args.once and args.scans is not None:
        print("error: --once and --scans conflict; use one", file=sys.stderr)
        return 2
    extensions = None
    if args.extensions is not None:
        extensions = tuple(
            ext if ext.startswith(".") else f".{ext}"
            for ext in (part.strip().lower() for part in args.extensions.split(","))
            if ext
        )
    watch_kwargs: dict = {}
    if extensions is not None:
        watch_kwargs["extensions"] = extensions
    watch_config = WatchConfig(
        watch_dir=args.watch_dir,
        out_dir=args.out_dir,
        target=target,
        profile=args.profile,
        source=args.source or config.source_language,
        backend=args.backend,
        model=args.model,
        script=args.script,
        glossary=args.glossary,
        max_chars=args.max_chars,
        mux=not args.no_mux,
        profiles=_load_profiles_paths(args.profiles) if args.profiles else None,
        bed_level=args.bed_level if args.bed_level is not None else config.bed_level,
        enhance=args.enhance,
        state_path=args.state,
        **watch_kwargs,
    )
    max_scans = 1 if args.once else args.scans
    any_failed = False

    def on_scan(scan_no: int, scan) -> None:
        nonlocal any_failed
        for result in scan.results:
            if result.status == "ok":
                print(
                    f"ok {result.name} (verdict {result.verdict}, "
                    f"MOS proxy {result.mos_proxy}, {result.duration_s:.1f}s)"
                )
            elif result.status == "skipped":
                print(f"skipped {result.name} (dubbed WAV already exists)")
            else:
                print(f"failed {result.name}: {result.error}", file=sys.stderr)
        print(
            f"scan {scan_no}: {scan.new_files} new, {scan.succeeded} ok, "
            f"{scan.failed} failed, {scan.skipped} skipped"
        )
        if scan.failed:
            any_failed = True

    watch_loop(
        watch_config,
        interval_s=args.interval,
        max_scans=max_scans,
        on_scan=on_scan,
    )
    return 1 if any_failed else 0


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
        if args.command == "translate":
            return _cmd_translate(args)
        if args.command == "clone":
            return _cmd_clone(args)
        if args.command == "voices":
            return _cmd_voices(args)
        if args.command == "diarize":
            return _cmd_diarize(args)
        if args.command == "synthesize":
            return _cmd_synthesize(args)
        if args.command == "cache":
            return _cmd_cache(args)
        if args.command == "dub":
            return _cmd_dub(args)
        if args.command == "dub-multi":
            return _cmd_dub_multi(args)
        if args.command == "patch":
            return _cmd_patch(args)
        if args.command == "report":
            return _cmd_report(args)
        if args.command == "enhance":
            return _cmd_enhance(args)
        if args.command == "preflight":
            return _cmd_preflight(args)
        if args.command == "split":
            return _cmd_split(args)
        if args.command == "merge":
            return _cmd_merge(args)
        if args.command == "compare":
            return _cmd_compare(args)
        if args.command == "estimate":
            return _cmd_estimate(args)
        if args.command == "export":
            return _cmd_export(args)
        if args.command == "render":
            return _cmd_render(args)
        if args.command == "batch":
            return _cmd_batch(args)
        if args.command == "watch":
            return _cmd_watch(args)
        if args.command == "config":
            return _cmd_config(args)
    except (
        AssemblyError,
        BatchError,
        CacheError,
        DiarizeError,
        EstimateError,
        ExportError,
        ExtractionError,
        MergeError,
        MultiTargetError,
        PatchError,
        RenderError,
        SplitError,
        TranscriptionError,
        TranslationError,
        VoiceCloneError,
        WatchError,
        ValueError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line entry point.

Subcommands (see ``video-dubber <cmd> --help`` for details):

* ``extract`` — pull a video's audio track out to a 16 kHz mono WAV.
* ``transcribe`` — speech-to-text on a WAV (whisper CLI, or a scripted
  demo transcript with ``--backend scripted --script words.json``).
* ``vad`` — list the speech segments an energy VAD finds in a WAV.
* ``convert`` — re-serialise a transcript JSON file as SRT / WebVTT / JSON.
* ``translate`` — translate a transcript JSON file to another language
  (dictionary backend, or ``argos-translate`` when installed).
* ``clone`` — build a speaker profile from reference WAV recordings.
* ``voices`` — list the profiles in a voice library, or add one to it.
* ``diarize`` — label who speaks in each transcript segment, by matching
  the segment audio against a voice library.
* ``synthesize`` — render text as audio in a cloned voice (demo synthesis).
* ``dub`` — render a full dubbed track from a translated transcript and
  a speaker profile, with a quality report (and optional video mux).
* ``enhance`` — clean up a WAV: trim silence, cut hum, gate room noise,
  normalize loudness to a target RMS.
* ``batch`` — dub many videos/transcripts from one manifest JSON file.
"""

import argparse
import sys

from video_dubber import __version__
from video_dubber.assemble import (
    AssemblyError,
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
from video_dubber.extract import ExtractionError, extract_audio
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

    p_translate = sub.add_parser("translate", help="Translate a transcript JSON file.")
    p_translate.add_argument("transcript", help="Input transcript JSON file.")
    p_translate.add_argument("--target", required=True, help="Target language code, e.g. hi, es.")
    p_translate.add_argument("--source", help="Source language code (default: transcript's).")
    p_translate.add_argument(
        "--backend",
        choices=["dictionary", "argos"],
        default="dictionary",
        help="Translation backend (default dictionary).",
    )
    p_translate.add_argument("--glossary", help="JSON glossary file of term overrides.")
    p_translate.add_argument(
        "--max-chars",
        type=int,
        help="Split long translated segments into cues of at most this many chars.",
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

    p_diarize = sub.add_parser("diarize", help="Label who speaks in each transcript segment.")
    p_diarize.add_argument("transcript", help="Input transcript JSON file.")
    p_diarize.add_argument("--audio", required=True, help="Original audio WAV.")
    p_diarize.add_argument("--library", required=True, help="Voice library JSON path.")
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
        help="Also write <out-stem>.srt and <out-stem>.vtt of the translated transcript.",
    )
    p_dub.add_argument("--original", help="Original audio WAV (for loudness matching).")
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
        choices=["json", "md"],
        default="json",
        help="Quality report format (default json).",
    )

    p_enhance = sub.add_parser(
        "enhance", help="Clean up a WAV file (trim, de-hum, gate, normalize)."
    )
    p_enhance.add_argument("input", help="Input WAV file.")
    p_enhance.add_argument("--out", required=True, help="Output WAV path.")
    p_enhance.add_argument(
        "--target",
        type=float,
        default=DEFAULT_TARGET_DBFS,
        help=f"Target RMS in dBFS (default {DEFAULT_TARGET_DBFS}).",
    )
    p_enhance.add_argument(
        "--ceiling",
        type=float,
        default=DEFAULT_CEILING_DBFS,
        help=f"Peak ceiling in dBFS (default {DEFAULT_CEILING_DBFS}).",
    )
    p_enhance.add_argument(
        "--high-pass",
        type=float,
        default=DEFAULT_HIGH_PASS_HZ,
        help=f"High-pass cutoff in Hz, 0 disables (default {DEFAULT_HIGH_PASS_HZ}).",
    )
    p_enhance.add_argument(
        "--gate",
        type=float,
        default=DEFAULT_GATE_THRESHOLD_DBFS,
        help=f"Noise-gate threshold in dBFS (default {DEFAULT_GATE_THRESHOLD_DBFS}).",
    )
    p_enhance.add_argument(
        "--no-trim",
        action="store_true",
        help="Keep leading/trailing silence (needed for timed dub tracks).",
    )
    p_enhance.add_argument("--report-out", help="Write the enhancement report (Markdown) here.")

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


def _cmd_translate(args: argparse.Namespace) -> int:
    with open(args.transcript, encoding="utf-8") as fh:
        transcript = Transcript.from_json(fh.read())
    source = args.source or transcript.language or "en"
    if args.backend == "argos":
        translator = ArgosTranslator()
    else:
        translator = DictionaryTranslator(source=source, target=args.target)
    glossary = Glossary.from_json_file(args.glossary) if args.glossary else None
    translated = translate_transcript(
        transcript,
        translator,
        target=args.target,
        source=source,
        glossary=glossary,
        max_chars=args.max_chars,
    )
    stats = (
        f"{translated.word_count} words, {len(translated.segments)} segments, "
        f"{source} -> {args.target} via {translator.name}"
    )
    if isinstance(translator, DictionaryTranslator):
        stats += f", dictionary coverage {translator.last_coverage:.0%}"
        if translator.last_unknown:
            stats += f" (untranslated: {', '.join(translator.last_unknown[:5])})"
    print(stats)
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
    if args.library:
        import os

        library = (
            VoiceLibrary.from_json_file(args.library)
            if os.path.exists(args.library)
            else VoiceLibrary()
        )
        library.add(profile, overwrite=True)
        library.save(args.library)
        print(f"added {profile.name} to library {args.library} ({len(library)} voices)")
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
    with open(args.transcript, encoding="utf-8") as fh:
        transcript = Transcript.from_json(fh.read())
    audio = resample(load_wav(args.audio), DEFAULT_SAMPLE_RATE)
    library = VoiceLibrary.from_json_file(args.library)
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


def _cmd_synthesize(args: argparse.Namespace) -> int:
    with open(args.profile, encoding="utf-8") as fh:
        profile = SpeakerProfile.from_json(fh.read())
    if args.semitones or args.wpm:
        profile = adjust_prosody(profile, semitones=args.semitones, words_per_minute=args.wpm)
    audio = ScriptedVoiceCloner().synthesize(args.text, profile)
    save_wav(args.out, audio)
    print(
        f"synthesized {len(args.text.split())} words in {profile.name}'s voice: "
        f"{audio.duration:.2f}s -> {args.out}"
    )
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
    if args.video and not args.mux_out:
        print("error: --video needs --mux-out", file=sys.stderr)
        return 2
    if args.bed_level is not None and not args.original:
        print("error: --bed-level needs --original", file=sys.stderr)
        return 2
    if args.bed_level is not None and not 0.0 <= args.bed_level <= 1.0:
        print("error: --bed-level must be between 0 and 1", file=sys.stderr)
        return 2
    with open(args.transcript, encoding="utf-8") as fh:
        transcript = Transcript.from_json(fh.read())
    with open(args.profile, encoding="utf-8") as fh:
        profile = SpeakerProfile.from_json(fh.read())
    profiles = _load_profiles_map(args.profiles) if args.profiles else None
    original = resample(load_wav(args.original), DEFAULT_SAMPLE_RATE) if args.original else None
    reference = None
    if args.reference:
        with open(args.reference, encoding="utf-8") as fh:
            reference = Transcript.from_json(fh.read())
    dub_audio, fit_factors, placed = dub_transcript(
        transcript,
        ScriptedVoiceCloner(),
        profile,
        original,
        profiles=profiles,
        bed_level=args.bed_level,
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
        bed_level=args.bed_level,
        enhancement=enhancement,
    )
    print(f"dubbed {report.segments_dubbed} segments, {dub_audio.duration:.2f}s -> {args.out}")
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
        text = report.to_json() if args.report_format == "json" else report.to_markdown()
        with open(args.report_out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.report_out}")
    if report.speakers_dubbed:
        print(f"speakers dubbed: {report.speakers_dubbed}")
    if args.subtitles:
        import os

        stem, _ = os.path.splitext(args.out)
        for ext, text in (("srt", transcript.to_srt()), ("vtt", transcript.to_vtt())):
            path = f"{stem}.{ext}"
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            print(f"wrote {path}")
    if args.video:
        mux_audio(args.video, args.out, args.mux_out)
        print(f"muxed dub into {args.mux_out}")
    return 0


def _cmd_enhance(args: argparse.Namespace) -> int:
    audio = load_wav(args.input)
    enhanced, report = enhance_audio(
        audio,
        target_dbfs=args.target,
        ceiling_dbfs=args.ceiling,
        high_pass_hz=args.high_pass,
        gate_threshold_dbfs=args.gate,
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
    print(f"reports: {args.out_dir}/batch-report.json, {args.out_dir}/batch-report.md")
    return 1 if summary.failed else 0


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
        if args.command == "dub":
            return _cmd_dub(args)
        if args.command == "enhance":
            return _cmd_enhance(args)
        if args.command == "batch":
            return _cmd_batch(args)
    except (
        AssemblyError,
        BatchError,
        DiarizeError,
        ExtractionError,
        TranscriptionError,
        TranslationError,
        VoiceCloneError,
        ValueError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

# Video Dubbing Studio

Multilingual video dubbing with voice cloning — take a video in one
language and produce a natural-sounding dubbed version in another,
keeping the speaker's voice.

## Roadmap

- [x] Project scaffold (README roadmap, license, gitignore)
- [x] Audio extraction + transcription (speech-to-text, word timings)
- [x] Translation pipeline (source → target language, subtitle-aware)
- [x] Voice cloning (few-shot speaker embedding, TTS synthesis)
- [x] Dub assembly (lip-sync timing, loudness match, mux back to video)
- [x] CLI + batch mode, quality report (WER, MOS proxy, sync drift)
- [ ] Self-review checklist, then public

## Quick start

```bash
pip install -e .[dev]
video-dubber --help
```

## Audio extraction + transcription

Requires ffmpeg on PATH for video input; transcription uses the
`whisper` CLI (openai-whisper) when installed.

```bash
# Pull a video's audio track out to 16 kHz mono WAV
video-dubber extract clip.mp4 --out clip.wav

# Transcribe it (whisper tiny model by default)
video-dubber transcribe clip.wav --language en --out transcript.json

# Re-segment the transcript at speech boundaries (drops words that fall
# in silence gaps) and export subtitles
video-dubber transcribe clip.wav --align --out subs.srt --format srt

# Find speech segments without transcribing
video-dubber vad clip.wav

# Convert a transcript between formats
video-dubber convert transcript.json --format vtt --out subs.vtt
```

No model installed? The `--backend scripted` path takes a JSON word list
(`[{text, start, end}]`) and runs the rest of the pipeline — it is a
labelled demo/test double, not speech recognition. Try the full flow
without any model:

```bash
python examples/demo.py
```

### How it works

- **Audio** (`src/video_dubber/audio.py`): stdlib-only WAV loading
  (8/16/24/32-bit PCM, stereo mixed to mono), saving, resampling.
- **Extraction** (`extract.py`): ffprobe probing + ffmpeg extraction to
  16 kHz mono WAV, with a clear error when ffmpeg is missing.
- **VAD** (`vad.py`): energy-based speech detection with an adaptive
  threshold from the recording's noise floor. It finds *where* people
  talk; loud non-speech (music, applause) counts as speech — a known
  limit of the energy baseline, to be replaced by a neural VAD later.
- **Transcripts** (`transcript.py`): timed words, segments, JSON / SRT /
  WebVTT serialisation.
- **Transcribers** (`transcribe.py`): a `Transcriber` protocol, a Whisper
  CLI adapter (word-timestamped JSON), and the scripted demo double.

## Translation pipeline

Translate a transcript to another language, keeping every segment's
timing window so the dub still lines up with the video:

```bash
# English transcript -> Hindi (offline dictionary backend)
video-dubber translate transcript.json --target hi --out translated.json

# With a glossary pinning team/brand names, and subtitle-sized cues
video-dubber translate transcript.json --target hi \
  --glossary glossary.json --max-chars 84 --format srt --out subs.srt
```

```json
// glossary.json — terms that must render exactly this way
{"Mars Rovers": "मार्स रोवर्स", "NASA": "NASA"}
```

- **Dictionary backend** (`--backend dictionary`, default): built-in
  en→hi and en→es word/phrase tables. Offline and deterministic; unknown
  words pass through unchanged and the CLI reports dictionary coverage,
  so a demo never silently invents text.
- **Argos backend** (`--backend argos`): shells out to the
  `argos-translate` CLI (LibreTranslate models) when installed, for real
  translation quality on any language pair it has a package for.
- **Timing**: translated words share the source segment's window in
  proportion to their length, so a translated segment still starts and
  ends exactly where the source did. `--max-chars` splits long translated
  segments into subtitle-sized cues at word boundaries, preferring breaks
  after sentence punctuation.
- Try it without a model: `python examples/demo_translate.py`.

## Voice cloning

Clone a speaker from reference recordings, then render text in that
voice:

```bash
# Build a speaker profile from 1+ reference WAVs (16 kHz mono is ideal)
video-dubber clone ref1.wav ref2.wav --name Asha --out asha.json

# Render text in Asha's voice (demo synthesis)
video-dubber synthesize "hello mars, welcome to the final" \
  --profile asha.json --out hello.wav

# Same voice, one octave up / faster delivery
video-dubber synthesize "hello mars" --profile asha.json \
  --semitones 12 --wpm 180 --out hello-fast.wav
```

- **Embedding** (`voice.py`): a real, deterministic 16-dim embedding
  from stdlib spectral features (band energies, zero-crossing rate,
  spectral centroid, autocorrelation pitch). It tells speakers apart —
  same-voice similarity 1.000 vs 0.803 across voices in the demo — but it
  is a timbre fingerprint, not a neural voice encoder.
- **Synthesis** is a labelled demo double: formant-shaped harmonic tones
  at the speaker's pitch and rate, timed per word. It sounds like a
  voice, not like the person; a real TTS backend (Coqui XTTS /
  OpenVoice) slots in behind the same `VoiceCloner` protocol.
- **Prosody**: `--semitones` shifts pitch, `--wpm` the speaking rate;
  the embedding never changes, so a shifted profile still matches the
  original speaker. `time_stretch` fits synthesised speech into a
  segment window for dub assembly.
- Try it without recordings: `python examples/demo_voice.py`.

## Dub assembly

Render a full dubbed track from a translated transcript and a speaker
profile, then mux it back into the video:

```bash
# Translated transcript + cloned voice -> dubbed WAV + quality report
video-dubber dub translated.json --profile asha.json \
  --original clip.wav --out dubbed.wav \
  --report-out report.json

# And straight into the video (video stream copied untouched)
video-dubber dub translated.json --profile asha.json \
  --out dubbed.wav --video clip.mp4 --mux-out dubbed.mp4
```

- **Fit**: each segment's speech is time-stretched into its original
  window (clamped to 0.6–1.6× — beyond that a dub sounds rushed or
  dragged, and the report flags it instead of producing unintelligible
  audio).
- **Loudness**: every segment is RMS-matched to the original recording
  so the dub does not jump in volume.
- **Quality report** (`--report-out`, `--report-format json|md`):
  sync drift (mean/max), fit factors, loudness error in dB, translation
  coverage, and a verdict — `good`, or `review` when max drift exceeds
  0.25s, any fit is extreme, loudness is off by more than 3 dB, or the
  word error rate against a `--reference` transcript exceeds 0.25.
- **WER** (`--reference transcript.json`): word error rate between the
  dubbed text and a same-language ground-truth transcript of what the
  dub should say — a same-language comparison; comparing across
  languages just reports ~1.0.
- **MOS proxy**: a heuristic 1–5 score from drift, fits, loudness and
  WER. It ranks dubs and flags bad ones; it is not a listening test,
  and the report says so.
- Full pipeline in one go: `python examples/demo_dub.py`.

## Batch mode

Dub many videos or transcripts from one manifest file:

```bash
video-dubber batch manifest.json --out-dir out/
```

```json
{
  "defaults": {"profile": "asha.json", "target": "hi"},
  "jobs": [
    {"name": "clip1", "video": "clip1.mp4", "script": "clip1-words.json",
     "backend": "scripted", "reference": "clip1-truth.json"},
    {"name": "clip2", "transcript": "clip2.json", "target": "es"}
  ]
}
```

- Each job writes to `out/<name>/` — `dubbed.wav`, `transcript.json`,
  `translated.json`, `report.json`, and `dubbed.mp4` when a video was
  muxed. A job is either a `video` (extract → transcribe → translate
  → dub) or a pre-made `transcript` (translate → dub). With
  `backend: scripted` a video job needs a `script` word list; otherwise
  it uses Whisper (`--model` via `model`, default `tiny`).
- `reference` is a ground-truth transcript of the source audio; the
  job's transcription WER is scored against it and averaged into the
  batch summary.
- Re-running a batch **skips** jobs whose `dubbed.wav` already exists
  (resume behaviour); `--force` redoes them, `--stop-on-error` stops
  after the first failure. One bad job never kills the rest — failures
  are listed with their error in the summary.
- `batch-report.json` / `batch-report.md` land in the output dir with
  per-job verdicts, MOS proxies, WERs, and totals.
- Try it without any models: `python examples/demo_batch.py`.

## Status

Audio extraction + transcription are done (62 tests). Translation is done
(94 tests). Voice cloning is done (126 tests). Dub assembly is done
(147 tests). Batch mode + quality metrics are done (185 tests).
Remaining: the self-review checklist and public flip — the project
stays private per the 5–7 day pace rule (started 2026-11-08, earliest
completion 2026-11-13).

## License

MIT

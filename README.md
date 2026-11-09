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
- [x] Multi-speaker dubbing, voice library, subtitle export
- [x] Background-music bed (extract, duck, mix under the dub)
- [x] Speaker diarization (label who speaks, from the voice library)
- [x] Audio enhancement (loudness normalize, hum cut, noise gate, silence trim)
- [x] Translation fidelity (round-trip back-translation check, flags unstable segments)
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
- **Fidelity check** (`--round-trip`): after translating, each segment is
  translated back to the source language and compared with the original
  (word-level Jaccard, threshold 0.6). Segments that come back mangled
  are flagged in a JSON/Markdown report (`--fidelity-out`) so a human
  can fix them before the dub is rendered. A perfect score means the
  translation is *stable*, not that it is correct — unknown words pass
  through untouched both ways. The check also caught a real bug: the
  dictionary tokenizer shattered Devanagari words at combining marks,
  so Hindi back-translation silently returned the input unchanged.
- Try it without a model: `python examples/demo_translate.py`, and
  `python examples/demo_fidelity.py` for the round-trip check.

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
- **Voice library** (`VoiceLibrary`): keep a project's voices in one
  JSON file. `video-dubber clone ref.wav --name Asha --out asha.json
  --library voices.json` adds the new profile on the spot;
  `video-dubber voices voices.json` lists the stored voices, and
  `voices --add profile.json [--overwrite]` manages the file.
  `VoiceLibrary.match(embedding)` identifies a speaker from a fresh
  recording by cosine similarity — it returns `None` below the
  threshold instead of mislabelling an unknown voice.
- Try it without recordings: `python examples/demo_voice.py`.

## Multi-speaker dubbing

When a transcript's segments carry `speaker` labels, each speaker can
have their own voice:

```bash
# One voice per speaker (paths resolve against the map's directory)
video-dubber dub translated.json --profile narrator.json \
  --profiles voices-map.json --out dubbed.wav \
  --subtitles --report-out report.json
```

```json
// voices-map.json — speaker name -> profile file
{"Asha": "asha.json", "Ravi": "ravi.json"}
```

- A segment whose speaker is in the map is rendered in that voice;
  unlabelled segments — or speakers with no profile — fall back to the
  default `--profile` voice instead of failing the dub.
- The quality report counts `speakers_dubbed` (distinct speaker labels).
- `--subtitles` writes `dubbed.srt` + `dubbed.vtt` of the translated
  transcript next to the dub WAV; batch jobs always write
  `translated.srt` / `translated.vtt` too, and a batch job can carry a
  `"profiles": {"Asha": "asha.json"}` key with the same meaning.
- Try it without recordings: `python examples/demo_multispeaker.py`.
  A runnable batch version lives in `examples/multispeaker/` — run it
  with `video-dubber batch examples/multispeaker/manifest.json --out-dir out/`.

## Speaker diarization

Transcription gives timed words, but no speaker labels — a two-person
interview arrives as one flat stream. Diarization labels it:

```bash
# Label each segment by matching its audio against the voice library
video-dubber diarize transcript.json --audio clip.wav \
  --library voices.json --out labelled.json --report-out report.json
```

- Each segment's audio is embedded (the same spectral fingerprint
  voice cloning uses) and matched against every profile in the library;
  the best match above `--threshold` (default 0.90) becomes the
  segment's `speaker` label. The labelled transcript feeds straight
  into multi-speaker dubbing (`dub --profiles`), so the full chain is
  `transcribe -> diarize -> translate -> dub` with no hand-labelling.
- Unknown or silent segments stay unlabelled instead of being guessed,
  and segments that already carry a label are kept unless
  `--overwrite` is passed — diarization labels, it never silently
  relabels.
- This is diarization-lite: the stdlib embedding separates clearly
  different voices well, but two similar voices (or segments under
  ~0.3 s) can land close together, and then the segment is honestly
  `unknown` in the report rather than mislabelled. Measured on the
  demo double: formant-synthesized voices at 110 vs 220 Hz cross at
  0.906–0.921 similarity — above the 0.90 default — so similar demo
  voices need a higher `--threshold` (e.g. 0.95; same-voice is 0.995+).
  Pure-tone references separate further (0.805 cross-voice).
- Try it without recordings: `python examples/demo_diarize.py`.

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
- **Music bed** (`--bed-level 0.3`, needs `--original`): a dub that
  replaces the whole soundtrack with silence between lines sounds
  dead. The original's non-speech audio — music, ambience, crowd
  noise — is extracted (speech windows muted with 30 ms fades, so the
  original voice never bleeds through) and mixed under the dubbed
  speech at the given level. In batch manifests the same key is
  `bed_level` (video jobs only). The report records the bed level.
  `video_dubber.bed` also offers `duck_bed` for externally supplied
  music: attenuate a music track under the placed dubbed clips.
- **Enhancement** (`--enhance`): clean up the finished dub track —
  hum cut, room-noise gate, loudness normalized to −16 dBFS RMS with
  a −1 dBFS peak ceiling. Trimming is disabled here on purpose: the
  dub sits on a timed bed, so its duration and segment start times
  must not move. The enhancement numbers land in the quality report.
  In batch manifests the same key is `enhance`. See the next section
  for the standalone command.
- Full pipeline in one go: `python examples/demo_dub.py`. Bed demo:
  `python examples/demo_bed.py`.

## Audio enhancement

Clean up any WAV — a raw recording or a finished dub — before it ships:

```bash
video-dubber enhance raw.wav --out clean.wav --report-out enhance.md
```

- **Trim**: leading/trailing silence is dropped (50 ms margin kept), so
  files start on the first word. `--no-trim` keeps the duration
  untouched — use it for timed dub tracks.
- **High-pass** (`--high-pass 80`, Hz, 0 disables): two cascaded
  one-pole stages (12 dB/octave) remove rumble and mains hum below
  the cutoff; a 440 Hz tone passes within 0.5 dB while 50 Hz hum is
  cut to ~0.28×.
- **Noise gate** (`--gate -45`, dBFS): 10 ms frames below the threshold
  are pushed down to a −60 dBFS floor with attack/release smoothing,
  so pauses go quiet without clicking.
- **Normalize** (`--target -16`, `--ceiling -1`, dBFS): gain to the
  target RMS; the gain is capped so the peak lands on the ceiling,
  and anything still above it is tanh soft-limited instead of clipped.
- The report prints before/after RMS and peak, gain applied and
  seconds trimmed. Demo: `python examples/demo_enhance.py` — a quiet
  voice under room hum goes from −32 to −16 dBFS with 0.90 s trimmed,
  and the timed variant keeps its 2.00 s duration.

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
  `translated.json`, `translated.srt`, `translated.vtt`, `report.json`,
  and `dubbed.mp4` when a video was muxed. A job is either a `video` (extract → transcribe → translate
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
Multi-speaker dubbing, voice library and subtitle export are done
(206 tests). Background-music bed preservation is done (228 tests).
Speaker diarization is done (243 tests). Audio enhancement is done
(255 tests): loudness normalize with soft limiter, hum high-pass,
noise gate, silence trim, `enhance` CLI, `dub --enhance` and batch
`enhance`, with before/after numbers in the quality report. Remaining:
the self-review checklist and public flip — the
project stays private per the 5–7 day pace rule (started 2026-11-08,
earliest completion 2026-11-13).

## License

MIT

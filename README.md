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
- [x] Project config file (dubber.toml defaults for translate/dub/enhance/diarize)
- [x] HTML quality reports (per-segment table, verdict banner, batch summary)
- [x] Dub-synced subtitles (SRT/VTT retimed to the placed dub audio)
- [x] Segment patching (re-dub one segment in place, refresh report)
- [x] Watch-folder mode (dub new files as they land, persistent state)
- [x] Multi-target dubbing (one source → several languages, per-language bundles + summary)
- [x] Pre-flight audio checks (clipping, DC offset, silence, SNR — abort a bad source before dubbing)
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
  (word-level Jaccard, threshold 0.6, tunable with
  `--fidelity-threshold`). Segments that come back mangled
  are flagged in a JSON/Markdown report (`--fidelity-out`) so a human
  can fix them before the dub is rendered. A perfect score means the
  translation is *stable*, not that it is correct — unknown words pass
  through untouched both ways. The check also caught a real bug: the
  dictionary tokenizer shattered Devanagari words at combining marks,
  so Hindi back-translation silently returned the input unchanged.
- Try it without a model: `python examples/demo_translate.py`, and
  `python examples/demo_fidelity.py` for the round-trip check.

## Project config file (dubber.toml)

Pin the flags a project repeats on every command in a `dubber.toml`
next to the work (or `~/.config/video-dubber/dubber.toml` for user
defaults; `--config path` overrides the search):

```toml
[dubber]
target_language = "hi"
voice_library = "voices.json"
bed_level = 0.3

[dubber.enhance]
target = -16.0
ceiling = -1.0
```

```bash
video-dubber config                    # show the effective configuration
video-dubber translate transcript.json # --target comes from the file
video-dubber enhance in.wav --out out.wav  # targets come from the file
```

Explicit CLI flags always win over the file. Unknown keys, bad types
and out-of-range values (e.g. `bed_level = 1.5`) fail loudly at load
time instead of silently doing nothing, and a `--config` path that
does not exist is an error rather than a quiet fall-back to defaults.
See `examples/dubber.toml` for the full commented file, and
`python examples/demo_config.py` for a live run.

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
- `--subtitles` writes `dubbed.srt` + `dubbed.vtt` next to the dub WAV;
  batch jobs always write `translated.srt` / `translated.vtt` too, and
  a batch job can carry a `"profiles": {"Asha": "asha.json"}` key with
  the same meaning. The cues are **retimed to the dubbed audio**: a
  segment's words are rescaled proportionally into its placed span, so
  a cue ends when the dubbed speech ends — not at the original window's
  end, where it would linger over silence after a clamped stretch.
  Cues shorter than 0.5s are stretched to that readable minimum,
  never past the next cue's start.
  (`video_dubber.subtitles.retime_transcript`)
- Try it without recordings: `python examples/demo_multispeaker.py`.
  A runnable batch version lives in `examples/multispeaker/` — run it
  with `video-dubber batch examples/multispeaker/manifest.json --out-dir out/`.
  The retiming demo is `python examples/demo_subtitles.py` — one short
  word in a 4s window gets a 0.85s cue instead of a 4s one.

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
  per-job verdicts, MOS proxies, WERs, and totals. `batch-report.html`
  is the same summary as a self-contained page (failed jobs highlighted,
  failures listed).
- Try it without any models: `python examples/demo_batch.py`.

## HTML quality reports

The JSON and Markdown reports are for machines and terminals; the HTML
report is the one you open in a browser and hand to a reviewer:

```bash
video-dubber dub translated.json --profile asha.json --out dubbed.wav \
  --report-out report.html --report-format html
```

- A **verdict banner** (green `good`, amber `review`), the **MOS proxy**
  as a bar, and the headline metrics (sync drift, fit factors, loudness
  error, WER, translation coverage).
- A **per-segment table** — start/end, speaker, text, drift, fit factor —
  with rows flagged when a segment drifted over 0.10s or stretched
  outside 0.8–1.25, so the reviewer sees exactly which segment to
  re-record instead of an average.
- Self-contained (stdlib only, inline CSS, no external assets) and fully
  escaped — a segment reading `<script>alert(1)</script>` renders as
  text, never as markup.
- Demo: `python examples/demo_report.py` — the slow second segment flips
  the verdict to `review` with exactly one flagged row.
- A report saved as JSON can be re-rendered later without re-dubbing:
  `video-dubber report report.json --out report.html` (or `--format md`).

## Segment patching

One bad line should not cost a full re-dub. `patch` re-synthesises a
single segment in its speaker's voice, fits it to the same window,
loudness-matches it to the clip it replaces, and splices it into the
existing dub — only the old placed span is zeroed, so a music bed
under the rest of the track is bit-identical:

```bash
video-dubber patch --dub dubbed.wav --transcript translated.json \
  --profile asha.json --segment 2 --text "corrected line" \
  --out patched.wav --transcript-out patched.json \
  --report report.json --report-out patched-report.json --subtitles
```

- Segments are numbered 1-based among timed segments — the same
  numbering the quality report's per-segment rows use.
- The patched transcript's words are retimed proportionally across the
  segment's original window, so subtitles and reports stay consistent.
- The refreshed report recomputes every dependent aggregate (drift,
  fit factors, MOS proxy, verdict) from the updated row; a patch that
  needs an extreme stretch flips the verdict to `review` honestly
  instead of hiding it.
- Demo: `python examples/demo_patch.py` — the patched line's 1.43x fit
  flips the verdict to `review` while segment 1 stays bit-identical.

## Watch-folder mode

Drop videos into a folder and have them dubbed as they arrive — no
manifest, no re-running a batch by hand:

```bash
video-dubber watch incoming/ --out-dir out/ --target hi \
  --profile asha.json --interval 60
```

- Every scan dubs each new media file (mp4/mkv/mov/webm/avi/wav/mp3/
  flac/m4a/ogg; `--extensions` overrides the list) through the full
  pipeline — extract → transcribe → translate → dub — into
  `out/<job-name>/`, exactly like a batch job. Job names come from the
  file name, sanitised and de-duplicated (`Morning Clip.mp4` →
  `morning-clip`).
- A state file, `out/.dubber-watch.json` by default (`--state` moves
  it), records every dubbed file by size + mtime fingerprint. A
  restart never re-dubs finished work; a file edited in place is
  re-dubbed once (with `--force` semantics, so the old output does not
  skip it); a file whose dub failed is retried on the next scan instead
  of being marked done.
- `--once` scans once and exits; `--scans N` stops after N scans (both
  for scripting and CI). Without either, the watcher polls forever
  until Ctrl-C, which stops it cleanly. `--backend scripted --script
  words.json` works for offline demos, and `--glossary`, `--max-chars`,
  `--no-mux`, `--profiles`, `--bed-level` and `--enhance` all apply to
  every file. `--target` / `--source` / `--bed-level` fall back to
  `dubber.toml` when the flags are left out.
- The fingerprint is size + nanosecond mtime, not a full-file hash —
  hashing whole videos on every scan would cost more than the scan.
  The one blind spot: a file rewritten with identical size and mtime
  is not detected; re-dub it with `--force` via batch instead.
- Demo: `python examples/demo_watch.py` — two files dubbed on the
  first scan, a second scan finds nothing new, a third file dropped in
  is dubbed on the next scan and only it.

## Multi-target dubbing

One source video, several languages — a single command dubs the same
transcript into every target and writes a per-language bundle plus one
summary comparing them:

```bash
video-dubber dub-multi source.json --targets hi,es,fr \
  --profile asha.json --out-dir bundles/
```

- Per target the full pipeline runs — translate (timing windows
  preserved), dub (synthesise + fit + loudness-match, optional music
  bed via `--bed-level` and `--enhance`) — and writes
  `dub.<target>.wav`, `dub.<target>.report.json` and dub-synced
  `dub.<target>.srt` / `.vtt` (`--no-subtitles` skips the subtitle
  files). `--stem` renames the bundle prefix; `--profiles`,
  `--glossary`, `--max-chars` and `--backend` (dictionary/argos) apply
  to every target.
- The summary, `dub.multilang.json` + `dub.multilang.md`, lists each
  target's verdict, MOS proxy, mean drift, mean fit and dictionary
  coverage in one table, so the weakest language is visible at a
  glance instead of buried in per-target reports.
- A target that fails (no dictionary for that pair, synthesis error)
  is recorded in the summary and the run carries on with the next
  language; the command exits 1 when any target failed. `--stop-on-error`
  aborts on the first failure instead. Dubbing into the source language
  itself is rejected up front — it is a no-op, not a dub — and duplicate
  targets (`hi,es,hi`) are dubbed once.
- Demo: `python examples/demo_multilang.py` — one English source dubbed
  into Hindi and Spanish; the Hindi dub needs a 0.66x mean fit on the
  demo's short windows, so its verdict honestly reads `review`.

## Pre-flight audio checks

Dubbing a clipped, silent, or noisy recording wastes a whole pipeline
run and produces a bad dub. Check the source first:

```bash
video-dubber preflight source.wav
```

It measures eight things — duration, sample rate (anything but 16 kHz
is resampled, so that is a warning), clipping, DC offset, peak level,
speech coverage (VAD), SNR (speech RMS vs the gaps between segments),
and leading/trailing silence — and prints a `pass` /
`pass_with_warnings` / `fail` verdict with one plain-language line per
check. `--json` prints the report as JSON, `--report-out` writes it to
a file, and `--fail-on-warn` exits 1 on warnings too. `dub --preflight`
runs the same checks on `--original` and aborts before any synthesis
when the source fails. Demo: `python examples/demo_preflight.py` —
a clean recording passes, a clipped one fails on clipping and DC
offset, a silent one on level and speech coverage.

## Status

Audio extraction + transcription are done (62 tests). Translation is done
(94 tests). Voice cloning is done (126 tests). Dub assembly is done
(147 tests). Batch mode + quality metrics are done (185 tests).
Multi-speaker dubbing, voice library and subtitle export are done
(206 tests). Background-music bed preservation is done (228 tests).
Speaker diarization is done (243 tests). Audio enhancement is done
(255 tests): loudness normalize with soft limiter, hum high-pass,
noise gate, silence trim, `enhance` CLI, `dub --enhance` and batch
`enhance`, with before/after numbers in the quality report. Translation
fidelity is done (282 tests): round-trip back-translation check with
`translate --round-trip`, a tunable `--fidelity-threshold`, JSON/Markdown
reports, and a grapheme-aware dictionary tokenizer (the check caught
the old `\w` tokenizer shattering Devanagari words at combining marks).
Project config file is done (315 tests): `dubber.toml` defaults for
translate/dub/enhance/diarize/clone, a `config` subcommand, and an
example file. HTML quality reports are done (334 tests): a self-contained
`report.py` renderer with a verdict banner, MOS proxy bar and a
per-segment table flagging drift over 0.10s or fits outside 0.8–1.25,
`dub --report-format html`, `batch-report.html`, and an end-to-end demo
run in CI. Saved JSON reports round-trip through `DubReport.from_json`
and can be re-rendered without re-dubbing via the `report` subcommand.
Dub-synced subtitles are done (349 tests): `subtitles.retime_transcript`
rescales each segment's words into its placed span, so `dub --subtitles`
and batch `translated.srt`/`.vtt` cues end when the dubbed speech ends
instead of lingering to the original window's end after a clamped
stretch; short cues stretch to a 0.5s readable minimum without ever
overlapping the next cue. Segment patching is done (374 tests):
`patch_transcript` / `patch_dub` / `refresh_report` plus the `patch`
CLI subcommand — one segment re-synthesised, fitted and loudness-matched
to the clip it replaces, spliced in with only its placed span zeroed,
and the saved report's aggregates recomputed (the demo's patched line
needs a 1.43x fit, so its verdict honestly flips to `review`); report
rows now also record each clip's placed duration, so a later patch and
subtitle retiming use the exact span. `patch --dry-run` reports the fit,
drift and verdict without writing anything, and a CLI test pins that a
patched bedded dub keeps its music bed bit-identical outside the
patched span. Watch-folder mode is done (397 tests): `watch` polls a
folder and dubs each new file through the full pipeline, with a JSON
state file so restarts never re-dub finished work, edited files are
re-dubbed once, and failed files are retried on the next scan; `--once`
and `--scans` make single scans scriptable. Multi-target dubbing is done
(414 tests): `dub-multi` dubs one source transcript into several
languages in one run, writing a per-language bundle (dubbed WAV, quality
report, dub-synced subtitles) plus a JSON/Markdown summary comparing the
targets; a failing target is recorded and the run carries on unless
`--stop-on-error` is set. Pre-flight audio checks are done (437 tests):
`preflight.py` measures duration, sample rate, clipping, DC offset,
peak level, VAD speech coverage, SNR and edge silence, and reports a
`pass` / `pass_with_warnings` / `fail` verdict as Markdown or JSON;
`dub --preflight` aborts on a failing source before any synthesis.
Remaining: the self-review
checklist and public flip — the
project stays private per the 5–7 day pace rule (started 2026-11-08,
earliest completion 2026-11-13).

## License

MIT

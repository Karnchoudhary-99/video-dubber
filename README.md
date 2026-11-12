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
- [x] Dub comparison (A/B two quality reports metric by metric, verdict + table)
- [ ] Self-review checklist, then public
- [x] Synthesis cache (skip re-synthesising unchanged segments, LRU eviction, `cache` CLI)
- [x] Dub cost/time estimator (predicted durations, fit factors, wall time, cache savings, cost)
- [x] Segment splitting (cut long lines at silence gaps so each piece fits its window)
- [x] Export bundle (portable dir/zip: dubbed audio, subtitles, report, manifest with SHA-256, verify)
- [x] ASS subtitle export (styled .ass: one named colour per speaker, centisecond timings)
- [x] Segment merging (join over-split fragments across short same-speaker gaps)
- [x] Environment diagnostics (`dubber doctor`: tools, versions, disk space)
- [x] Per-segment loudness levelling (`level.py`: even out line volumes, `dubber level`, `dub --level`)
- [x] Dubbing style presets (documentary/conversational/energetic/narration/dramatic, `dub --style`, `dubber styles`)
- [x] Pronunciation lexicon (names/acronyms spoken as the lexicon says, `dub --lexicon`, `dubber lexicon`)

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
style = "documentary"

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
- `--subtitles` writes `dubbed.srt` + `dubbed.vtt` + `dubbed.ass` next to the dub WAV;
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

## ASS subtitle export

SRT/VTT carry one cue per segment and no styling at all, so a
two-speaker dub reads as one voice on screen. The `.ass` file
`dub --subtitles` (and `patch --subtitles`) writes next to the SRT/VTT
gives every speaker their own named style and colour — Asha white,
Ravi yellow, and so on through an 8-colour palette that cycles when a
project has more speakers than colours:

```bash
# Styled ASS from any transcript (colours assigned per speaker label)
video-dubber convert transcript.json --format ass --out subs.ass
```

- **Format**: a real Advanced SubStation Alpha file — `[Script Info]`
  (1280x720 play resolution), `[V4+ Styles]` (one `Style:` line per
  speaker: font, size, `#RRGGBB` colours converted to ASS
  `&H00BBGGRR`, bold/italic, numpad alignment, margins) and `[Events]`
  (one `Dialogue:` line per timed segment, in the speaker's style,
  with the speaker's name in the Name field). Centisecond timings are
  truncated, never rounded up, so a cue never claims to end after the
  audio does; `{`/`}` in the text are escaped so a line that says
  `{wow}` still shows on screen.
- **API**: `video_dubber.subtitles.to_ass(transcript)` (explicit styles
  via `styles={speaker: AssStyle(...)}`, falling back to `Default` for
  unlisted speakers) and `assign_speaker_styles(transcript)` for the
  palette assignment — style names are sanitised (commas become
  underscores, collisions get a `_2` suffix) so two speakers never
  share a style name. The same file feeds `dubber render --subtitles`
  directly, so the colours on screen match the colours in the file.
- Demo: `python examples/demo_ass.py` — two speakers, two styles,
  cues retimed to the dubbed audio.

## Segment merging

Splitting is the right call for a long line, but transcription and VAD
over-segment too: a hesitant speaker arrives as five 0.4 s fragments,
and dubbing each one separately means five tiny syntheses, five fit
factors, and subtitles that flash past faster than anyone can read.
`dubber merge` is split's counterpart — it joins adjacent segments
back into readable lines:

```bash
video-dubber merge fragments.json --out merged.json --report-out merge.md
# # Segment merge
# - Segments: 5 -> 2 (2 merges, 3 segments joined)
# - Longest segment: 0.80s -> 1.60s
```

- A segment joins the group before it only when all three hold: the
  same speaker label (`None` counts as a label, so two unlabelled
  lines merge but Asha never merges into Ravi), the pause between them
  is at most `--max-gap` (default 0.4s), and the combined line is at
  most `--max-duration` (default 8.0s) long. A merged group keeps every
  word in order with its original timings, so it dubs and subtitles
  exactly like the input.
- Wordless segments are copied through unchanged and break a group —
  they carry no timing, so they can neither join nor be joined across.
- The report (`--json` for machines) lists every join with the
  original segment numbers (`1+2+3`), so the caller can see what
  merged instead of trusting an average.
- Demo: `python examples/demo_merge.py` — five fragments become two
  lines, and the speaker change at the end stays separate.

## Environment diagnostics

The pipeline shells out to ffmpeg, ffprobe, whisper and
argos-translate, and a missing binary fails a run halfway through —
after the transcription, or after the dub — with an error that names
the tool but not the fix. `dubber doctor` checks everything up front:

```bash
video-dubber doctor
# # Doctor: ok_with_warnings
# - [✓] python (required): Python 3.12.3 (>= 3.11)
# - [✓] ffmpeg (required): ffmpeg version 8.1.2 ...
# - [✓] ffprobe (required): ffprobe version 8.1.2 ...
# - [✓] ffmpeg subtitles filter (required): libass subtitles filter available
# - [~] whisper (optional): not found on PATH (pip install openai-whisper ...)
# - [~] argos-translate (optional): not found on PATH (...)
# - [✓] disk space (optional): 125.8 GiB free under . (>= 1.0 GiB)
```

- Each check is **required** or **optional**: a missing ffmpeg fails
  the run (nothing works without it); a missing whisper only warns
  (the scripted backend still demos the whole pipeline). The verdict
  is `ok`, `ok_with_warnings`, or `fail` — and the command exits 1 on
  `fail` (`--fail-on-warn` exits 1 on warnings too, for CI gates).
- The subtitles-filter check reads `ffmpeg -filters` and confirms the
  libass filter `render` needs is actually in the build, so a
  minimal ffmpeg that cannot burn cues in is caught here, not at
  render time.
- `--json` prints a machine-readable report, `--report-out` writes it
  to a file, and `--out-dir` measures free space where dubs land
  (a full disk fails a render mid-write).
- Demo: `python examples/demo_doctor.py` — the full diagnosis on this
  machine, with the report asserted well-formed.

## Per-segment loudness levelling

A dub is assembled from separately synthesised lines, and even with
each line loudness-matched to the original, the dubbed lines can
still jump against each other — one whispered, the next shouted —
because the original's own levels vary. `dubber level` evens them
out on the finished track:

```bash
video-dubber level dub.wav transcript.json --out level.wav --report-out level.md
# # Segment levelling report
# - Target RMS: -16.0 dBFS
# - Max deviation from target: 13.03 dB -> 0.15 dB
# - Segments levelled: 2
# - Segments at the gain cap: 0
# - Silent segments (untouched): 0
```

- Each timed segment is gained toward a common target RMS (default
  -16 dBFS), clamped to ±12 dB (`--max-gain`) and to the peak ceiling
  (`--ceiling`, default -1 dBFS) — a segment that would need more gain
  than the cap allows is reported as `at cap` instead of being pushed
  into clipping, and the report's after-deviation says so honestly.
- The gain ramps in and out over 5 ms at each segment edge so level
  changes never click, and the track is soft-clipped at the ceiling
  (tanh) as a final safety net. Timing never moves: a levelled dub
  still lines up with its subtitles and its quality report.
- Silent segments are reported, not gained; wordless segments are
  skipped. `--json` prints the report as JSON, `--report-out` writes
  it to a file, and `dub --level` (with `--level-target` /
  `--level-max-gain`) levels the assembled dub in place — after
  `--enhance` when both are given.
- Demo: `python examples/demo_level.py` — a quiet (-29 dBFS) and a
  loud (-9 dBFS) line levelled to -16 dBFS, then the same track with
  the default 12 dB cap, where the quiet line honestly lands at
  -17 dBFS and is flagged at cap.

## Dubbing style presets

The same cloned voice can read a documentary slowly and a trailer
fast without re-cloning — a style is performance, not identity. It
scales the speaking rate and shifts the pitch (`adjust_prosody`
under the hood) and leaves the embedding untouched, so a styled
profile still matches its speaker:

```bash
# List the five presets and their numbers
video-dubber styles
#   documentary: 0.95x rate, +0 semitones — Measured, steady delivery ...
#   conversational: 1.00x rate, +0 semitones — Natural everyday speech ...
#   energetic: 1.15x rate, +2 semitones — Fast, bright delivery ...
#   narration: 0.90x rate, -1 semitones — Slow, low audiobook-style narration.
#   dramatic: 0.85x rate, -2 semitones — Deliberate, low delivery ...

# Dub in a style (also: dub-multi, patch, estimate, batch manifests,
# and a [dubber] style = "..." default in dubber.toml)
video-dubber dub translated.json --profile asha.json --style energetic \
  --out dubbed.wav --report-out report.json
# delivery style: energetic
```

- Styles matter because fit is the dub's hard constraint: a fast
  style fits long translations into short windows with less
  time-stretching, while a slow style needs wider windows — pick the
  style before estimating, and `estimate --style` predicts at the
  styled rate (its cache probe uses the styled profile, so a segment
  cached in one style is not counted as a hit for another).
- The style is recorded in the quality report (JSON, Markdown and
  HTML) and in every per-target report from `dub-multi`; `patch`
  re-reads the replacement line in the dub's style so the patched
  line does not sound like a different take. An unknown style name
  fails loudly (exit 2 on the CLI, a `StyleError` in the library)
  instead of silently dubbing in the default.
- Demo: `python examples/demo_style.py` — the same two lines dubbed
  plain (mean fit 1.06), energetic (1.19) and dramatic (0.91). The
  energetic dub's max fit is 1.27, outside the 0.8–1.25 band, so its
  verdict honestly flips to `review`: the style needs wider windows
  for these lines, and the report says so.

## Pronunciation lexicon

A synthesiser guesses at proper nouns — "NASA" comes out letter by
letter, "Karn" with the wrong vowel, "New Delhi" flattened. A lexicon
is a JSON file mapping the written form to the form the synthesiser
should speak: a respelling (`"Nguyen" -> "win"`), a spaced-out acronym
(`"NASA" -> "N A S A"`), or a simpler spelling (`"Bengaluru" ->
"bangalore"`):

```json
{"entries": {"NASA": "N A S A", "Karn": "kar-n", "New Delhi": "nyoo delhi"}}
```

```bash
# Check a lexicon, or see what it does to a line
video-dubber lexicon lexicon.json
video-dubber lexicon lexicon.json --apply "Karn reports from New Delhi for NASA"
#   Kar-n reports from Nyoo delhi for N A S A

# Dub (or synthesize) with it — also: dub-multi, batch manifests,
# and a [dubber] lexicon = "..." default in dubber.toml
video-dubber dub translated.json --profile asha.json --lexicon lexicon.json \
  --out dubbed.wav --report-out report.json
# lexicon pronunciations applied: 3
```

- Matching is case-insensitive with real word boundaries, longest
  phrase first (`"New Delhi"` wins over `"Delhi"`), and case is
  preserved where it matters: an all-caps match gets an all-caps
  pronunciation, a capitalised match gets a capitalised one. Matching
  reuses the grapheme-aware tokenizer from the translation pipeline,
  so Devanagari entries such as `"नमस्ते" -> "namaste"` match whole —
  the glossary's `\b` regex would shatter them at the combining marks.
  Entries match whole words only: `"NASA"` never rewrites `"NASAL"`,
  and `"Karn"` leaves `"Karn's"` alone.
- The lexicon rewrites only the spoken audio, never the transcript:
  subtitles keep the written words, and the quality report records
  how many pronunciations were applied (`lexicon_applied`, in JSON,
  Markdown and HTML) so a reviewer can see exactly what changed. A
  missing or invalid lexicon file is a hard error (exit 2), never a
  silent fall-back to guessing.
- Demo: `python examples/demo_lexicon.py` — one two-line transcript
  dubbed plain (mean fit 1.39) and with the lexicon (mean fit 1.19,
  3 pronunciations applied). Both verdicts honestly read `review`:
  the demo's windows are tight for these lines, and the longer
  respellings need the stronger stretch the report shows.

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

## Dub comparison

Two dubs of the same source — different voices, different backends,
before/after a fix — and the question is which one is better. Compare
their saved quality reports instead of trusting your ears alone:

```bash
video-dubber compare asha.report.json ravi.report.json \
  --label-a asha --label-b ravi
```

It scores MOS proxy, mean/max sync drift, segments over drift, mean
fit factor (distance from 1.0), extreme fits, the report verdict, and
— when both reports have them — loudness error, translation coverage
and WER, one row per metric with its winner. The overall verdict goes
to the side that wins more metrics, ties broken by the MOS proxy;
a metric only one report has is skipped, never counted as a loss.
`--json` prints the comparison as JSON, `--report-out` writes it to a
file. Demo: `python examples/demo_compare.py`.

## Synthesis cache

Synthesis is the slowest step of the pipeline, and re-dubbing an
unchanged transcript re-renders every segment. Pass `--cache-dir` to
`dub`, `patch` or `synthesize` and each unique line is rendered once,
ever — a re-dub of the same transcript in the same voice is served
entirely from the cache, and a patch re-renders only the segment it
touches:

```bash
video-dubber dub translated.json --profile asha.json --out dub.wav \
  --cache-dir ~/.cache/video-dubber/synth
# cache: 2 hit(s) / 0 miss(es) (1.00 hit rate, 2 segment(s) stored)
```

The key is a hash of everything that determines the sound — cloner,
exact text, speaker name + embedding, base pitch, speaking rate and
sample rate — so any change to the line, the voice or the rate is a
different entry, never a stale one. Entries are 16-bit WAVs under the
cache directory with a JSON index; the cache evicts least-recently-used
entries past 512 MiB / 1000 entries, and a corrupt index simply starts
empty — the cache is an optimisation, so it never breaks a dub.
`video-dubber cache [--cache-dir DIR]` prints entries, size and hit
rate; `--clear` empties it. Demo: `python examples/demo_cache.py`.

## Video render

A dub is only reviewable when you can *watch* it — with the
translated lines on screen, timed to the dubbed speech. `dubber render`
takes the original video, the dubbed WAV and a retimed SRT (the one
`dub --subtitles` writes) and produces a final MP4 with the subtitles
burned into the picture via ffmpeg's `subtitles` filter (libass), so
the file plays anywhere — no player subtitle support needed:

```bash
video-dubber render clip.mp4 dubbed.wav --subtitles dubbed.srt \
  --out rendered.mp4
# rendered rendered.mp4 (4.00s, subtitles filter, CRF 23)
# burned in 2 subtitle cue(s)
```

Style is configurable (`--font-size`, `--font`, `--bold`,
`--alignment` for the ASS numpad position, `--margin-v`), and the
encode uses libx264 (`--crf`, `--preset`). `--preview-start` /
`--preview-duration` render only a slice, so a long dub can be
spot-checked in seconds — the seek is applied to the output, not to
one input, so audio and video stay in sync in the slice. `dub
--burn-subtitles` does the same in one step when muxing: it renders
the muxed video with the retimed SRT burned in instead of a plain
mux. A render report (JSON/Markdown, `--report-out`) records the
output duration, the number of cues burned in and the filter used.
Needs an ffmpeg build with the `subtitles` filter; rendering without
`--subtitles` still works (video re-encoded with the new audio).
Demo: `python examples/demo_render.py`.

## Dub cost/time estimator

Dubbing a long video is the expensive step, and a bad fit only shows
up afterwards as stretched audio. `dubber estimate` predicts the dub
from the translated transcript alone — before any synthesis runs:

```bash
video-dubber estimate translated.json --target hi \
  --profile asha.json --cache-dir ~/.cache/video-dubber/synth \
  --price-per-char 0.002
# # Dub estimate
# - Backend: `scripted` (exact durations)
# - Segments: 2 over 3.20s of source
# - Predicted speech: 2.99s (mean fit 1.33x, max 1.61x, 1 extreme)
# - Cache: 1/2 segments already cached — 25 of 38 chars to synthesise
# - Estimated wall time: 0.2s (0.1s synthesis at 20x real-time + 0.1s overhead)
# - Estimated cost: 0.0500 USD
```

Per segment it reports the window, predicted speech duration, fit
factor (flagged when outside 0.8–1.25x, the same band the quality
report calls extreme) and whether the segment is already cached. The
duration model is exact for the `scripted` backend — it is the same
per-word formula the synthesiser uses, so the demo's prediction
matches a real render to the sample; for other backends (coqui,
openvoice, elevenlabs) the same word-rate model is the best available
guess and the estimate says so. Wall time uses a per-backend
real-time factor (rough defaults, override with `--rtf` after
measuring your own setup); unknown backends require `--rtf`. Cost is
priced on the characters that still need synthesis — cache hits are
excluded — via `--price-per-char` or `--price-per-minute`
(`--currency` for the unit). `--format json` writes a machine-readable
estimate. Probing the cache is read-only: it never loads audio or
moves hit counters. Demo: `python examples/demo_estimate.py`.

## Segment splitting

A long segment is the dub assembler's worst case: the whole line must
be stretched or squeezed into one window, so a 20-second sentence in
a 12-second window needs a 0.6x fit and the quality report flags it.
`dubber split` cuts over-long segments at real pauses instead:

```bash
video-dubber split transcript.json --out split.json --audio source.wav
# # Segment split
# - Segments: 1 -> 2 (1 split)
# - Longest segment: 9.10s -> 4.10s
#
# | Segment | Cut at | Reason |
# | --- | --- | --- |
# | 1 | 4.10s | silence |
```

A boundary is splittable when the pause there is at least `--min-gap`
(default 0.4s) — read off the word timings, and, when `--audio` is
given, confirmed against VAD silence gaps, so a cut point is found
even when the word timings are coarse (VAD edges are frame-quantised
plus hangover, so a boundary up to 150 ms before a silence interval
starts still counts). `--max-duration` (default 8.0s) sets the longest
piece; `--min-segment` (default 1.0s) keeps the splitter from leaving
a stub when a choice exists. When a piece runs over the limit with no
silence candidate, the split falls back to the longest pause inside
the piece, and only as a last resort to a forced cut at the word
nearest the limit — forced cuts are counted in the report, not hidden,
and a single long word is never cut. Speaker labels survive on every
piece, and the input transcript is untouched. `dub --split-segments`
(`--max-segment-duration`) applies the same split to the translated
transcript before synthesis, so each piece fits its window; the split
report prints first and the quality report reflects the split
segments. Demo: `python examples/demo_split.py`.

## Export bundle

A dub is only useful once it leaves this machine. `dubber export`
packs a finished dub into a portable bundle — a directory, or a single
`.zip` — with a `manifest.json` recording the provenance (source and
target languages, speakers, quality verdict, MOS proxy) and a SHA-256
inventory of every file inside, plus a human-readable `README.md`:

```bash
dubber export dubbed.wav --target hi --subtitles dubbed.srt \
  --report report.json --original clip.mp4 --out mars-cricket.hi.zip
# # Export bundle: dubbed.hi
# - Languages: en -> hi
# - Verdict: **review**
# ...
# wrote mars-cricket.hi.zip

dubber export --verify mars-cricket.hi
# bundle intact: mars-cricket.hi
```

Two files from different roles can share a basename (the dub and the
original are both often `clip.wav`), so colliding names are prefixed
with their role (`original_clip.wav`) instead of silently overwriting
each other. Verification re-hashes every file and reports anything
missing, changed, or extra — a tampered bundle fails loudly instead of
passing quietly. Demo: `python examples/demo_export.py`.

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
Dub comparison is done (452 tests): `compare_reports` scores two
saved quality reports metric by metric (MOS proxy, sync drift, fit
factors, verdict, plus loudness error, coverage and WER when both
reports have them), with a win-count verdict broken by the MOS proxy
and a Markdown table showing why the winner won; `dubber compare`
takes labels and a `--json` flag. Synthesis cache is done
(471 tests): `cache.py` content-addresses rendered segments by
cloner, text, speaker embedding, prosody and sample rate, stores them
as WAVs with a JSON index, evicts least-recently-used entries past
512 MiB / 1000 entries, and serves re-dubs and patches without
re-synthesis; `dub`/`patch`/`synthesize` take `--cache-dir` and the
`cache` subcommand reports entries, size and hit rate (the demo's
repeat render is 1 synthesis call, 1 cache hit). Video render is done
(507 tests): `render.py` burns a retimed SRT into the picture with
ffmpeg's `subtitles` filter (configurable ASS style: size, font,
bold, alignment, margin), re-encodes with libx264/AAC, and
`render_preview` slices a short spot-check with an output-side seek
so A/V stay in sync; `dubber render` takes style/preset/CRF flags
and `--report-out`, and `dub --burn-subtitles` renders the muxed
video with subtitles burned in instead of a plain mux (a preview test
caught `-shortest` cutting a sliced render short when the dub audio
ended first — duration-bounded renders now let `-t` bound the
output). Dub cost/time estimator is done
(528 tests): `estimate.py` predicts each segment's speech duration
(exact for the scripted backend — the demo's prediction matches a
real render to the sample), fit factors, synthesis wall time at the
backend's real-time factor, cache savings via a read-only
`SynthesisCache.contains` probe, and cost at a per-character or
per-minute rate; `dubber estimate` takes `--profile/--wpm/--rtf/
--cache-dir/--price-per-char/--price-per-minute/--currency/--format/
--out`, and unknown backends require `--rtf` rather than guessing.
Segment splitting is done (547 tests): `split.py` cuts segments
longer than `--max-duration` at silence gaps from the word timings
or, with `--audio`, VAD-confirmed pauses, falling back to the longest
pause and then a counted forced cut (a test caught the longest-gap
search excluding the piece's last word; a VAD edge-tolerance of
150 ms covers frame quantisation plus hangover); `dubber split`
writes the split transcript plus a Markdown/JSON report, and
`dub --split-segments` splits before synthesis so each piece fits
its window. Export bundle is done (563 tests): `export.py` builds a
manifest (provenance + SHA-256 file inventory, JSON/Markdown,
round-trippable), `export_bundle` writes a directory bundle and
`export_zip` a single `.zip`, and `verify_bundle` re-hashes a bundle
and reports missing, changed, or unlisted files; `dubber export`
takes the dubbed audio plus optional subtitles, report, original and
rendered video, and `dubber export --verify` checks a bundle
(the demo's tampered copy fails verification on both hash and size).
ASS subtitle export is done (576 tests): `subtitles.to_ass` writes a
real `.ass` file — `[Script Info]`, one named style per speaker in
`[V4+ Styles]` (colours from an 8-entry palette, names sanitised so
two speakers never collide), and one `Dialogue` line per timed segment
with centisecond timestamps truncated, never rounded up; `convert
--format ass` exports any transcript, and `dub`/`patch --subtitles`
write the `.ass` next to the `.srt`/`.vtt` (the demo's two speakers
render white and yellow, and the same file feeds `render --subtitles`
directly).
Segment merging is done (590 tests): `merge.py` joins adjacent
segments across a short pause (default 0.4s) when the speakers match
and the combined line fits one dub window (default 8.0s), keeping
every word's original timings; wordless segments pass through and
break a group, and the report lists every join by original segment
number; `dubber merge` takes `--max-gap`/`--max-duration` and a
JSON/Markdown report (the demo's five fragments become two lines,
and the speaker change stays separate).
Environment diagnostics are done (606 tests): `doctor.py` checks
python >= 3.11, ffmpeg/ffprobe presence + version, the libass
`subtitles` filter (parsed from `ffmpeg -filters`, so a minimal
ffmpeg build is caught here, not at render time), whisper and
argos-translate as optional backends, and free disk space under the
output directory; `dubber doctor` prints a Markdown/JSON report with
an `ok` / `ok_with_warnings` / `fail` verdict and exits 1 on failure
(the demo's bare machine honestly reports `ok_with_warnings` —
whisper and argos missing, everything required present).
Per-segment loudness levelling is done (624 tests): `level.py`
measures each timed segment's RMS/peak, gains it toward -16 dBFS
clamped to ±12 dB and the -1 dBFS peak ceiling, with 5 ms gain ramps
at the segment edges and a tanh soft clip as a safety net; silent
segments are reported, not gained, and timing never moves.
`dubber level` takes `--target/--max-gain/--ceiling/--json/
--report-out`, and `dub --level` levels the assembled dub in place
(a test caught the after-deviation measuring the *before* levels, so
the report now measures the levelled track; the demo's quiet line
needs +13 dB, past the default 12 dB cap, so it honestly lands at
-17 dBFS and is flagged at cap).
Dubbing style presets are done (669 tests): `style.py` defines five
named delivery styles — documentary (0.95x, +0 semitones),
conversational (1.00x, +0), energetic (1.15x, +2), narration (0.90x,
-1) and dramatic (0.85x, -2) — applied via `adjust_prosody`, so the
voice's embedding is untouched and the styled profile still matches
its speaker; `dub`/`dub-multi`/`patch`/`estimate` take `--style`,
batch manifests take a `style` key (or a `defaults` one),
`dubber.toml` pins a default, and `dubber styles` lists the presets.
The style is recorded in the quality report and in every
per-target `dub-multi` report, `estimate --style` predicts at the
styled rate with style-aware cache probes, and unknown style names
fail loudly instead of silently dubbing in the default (the demo's
energetic dub needs a 1.27x max fit, so its verdict honestly flips
to `review`).
Pronunciation lexicon is done (715 tests): `lexicon.py` maps written
forms to spoken ones (respellings, spaced-out acronyms, multi-word
phrases), matched case-insensitively with real word boundaries,
longest phrase first, on the grapheme-aware tokenizer — so Devanagari
entries match whole where the glossary's `\b` regex would shatter
them. `dub`/`dub-multi`/`synthesize` take `--lexicon`, batch
manifests take a `lexicon` key (or a `defaults` one), `dubber.toml`
pins a default, and `dubber lexicon` checks a file or applies it to
a line. The lexicon rewrites only the spoken audio, never the
transcript, and the quality report records how many pronunciations
were applied (`lexicon_applied`, per target in `dub-multi`); a bad
lexicon file fails loudly (exit 2) instead of silently dubbing
unguessed. The demo dubs one two-line transcript plain (mean fit
1.39) and with the lexicon (mean fit 1.19, 3 pronunciations
applied) — both verdicts honestly read `review`, the demo's windows
are tight for these lines.
Remaining: the self-review
checklist and public flip — the
project stays private per the 5–7 day pace rule (started 2026-11-08,
earliest completion 2026-11-13).

## License

MIT

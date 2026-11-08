# Video Dubbing Studio

Multilingual video dubbing with voice cloning — take a video in one
language and produce a natural-sounding dubbed version in another,
keeping the speaker's voice.

## Roadmap

- [x] Project scaffold (README roadmap, license, gitignore)
- [x] Audio extraction + transcription (speech-to-text, word timings)
- [x] Translation pipeline (source → target language, subtitle-aware)
- [ ] Voice cloning (few-shot speaker embedding, TTS synthesis)
- [ ] Dub assembly (lip-sync timing, loudness match, mux back to video)
- [ ] CLI + batch mode, quality report (WER, MOS proxy, sync drift)
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

## Status

Audio extraction + transcription are done (62 tests). Translation is done
(94 tests). Next: voice cloning, then dub assembly.

## License

MIT

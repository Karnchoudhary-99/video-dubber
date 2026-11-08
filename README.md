# Video Dubbing Studio

Multilingual video dubbing with voice cloning — take a video in one
language and produce a natural-sounding dubbed version in another,
keeping the speaker's voice.

## Roadmap

- [x] Project scaffold (README roadmap, license, gitignore)
- [ ] Audio extraction + transcription (speech-to-text, word timings)
- [ ] Translation pipeline (source → target language, subtitle-aware)
- [ ] Voice cloning (few-shot speaker embedding, TTS synthesis)
- [ ] Dub assembly (lip-sync timing, loudness match, mux back to video)
- [ ] CLI + batch mode, quality report (WER, MOS proxy, sync drift)
- [ ] Self-review checklist, then public

## Quick start

```bash
pip install -e .[dev]
video-dubber --help
```

## Status

Scaffold only — transcription, translation, cloning and assembly land in
the sessions ahead.

## License

MIT

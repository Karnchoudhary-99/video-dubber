"""Project scaffolding: ``dubber init`` starts a new dubbing project.

A dubbing project is a directory with a ``dubber.toml`` (defaults for
every command), a ``manifest.json`` (batch jobs), a ``voices.json``
library, and a ``source/`` folder for the media. Writing those four
files by hand is error-prone — a typo in the TOML or a manifest job
pointing at a file that does not exist only fails when the batch runs.
:func:`scaffold_project` writes all of them at once, and every file it
writes is valid on the spot: the config loads with
:func:`video_dubber.config.load_config`, the manifest loads with
:func:`video_dubber.batch.BatchManifest.from_json_file`, and the voice
library loads with :class:`video_dubber.voice.VoiceLibrary`.

The scaffold ships a runnable example job: a short synthetic tone in
``source/example.wav`` dubbed with the scripted (offline) transcriber,
so ``dubber batch manifest.json`` works immediately with no model and
no network. Pass ``source`` to also seed a job for a real video — that
job uses the Whisper transcriber and the same narrator profile.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from video_dubber.audio import make_sine, save_wav
from video_dubber.config import EnhanceConfig
from video_dubber.style import StyleError, get_style
from video_dubber.voice import SpeakerProfile, VoiceLibrary, extract_embedding

__all__ = [
    "InitError",
    "scaffold_project",
]

_CONFIG_FILE = "dubber.toml"
_MANIFEST_FILE = "manifest.json"
_LIBRARY_FILE = "voices.json"
_PROFILE_FILE = "voices/narrator.json"
_EXAMPLE_WAV = "source/example.wav"
_EXAMPLE_SCRIPT = "source/example-script.json"
_README_FILE = "README.md"
_GITIGNORE_FILE = ".gitignore"

# Files the scaffold owns: the only paths it ever writes or overwrites.
_OWNED_FILES = (
    _CONFIG_FILE,
    _MANIFEST_FILE,
    _LIBRARY_FILE,
    _PROFILE_FILE,
    _EXAMPLE_WAV,
    _EXAMPLE_SCRIPT,
    _README_FILE,
    _GITIGNORE_FILE,
)

_EXAMPLE_WORDS = (
    "Welcome to the Mars Sports League, where every match is played "
    "under the domes of the red planet."
)
_EXAMPLE_TONE_HZ = 180.0
_EXAMPLE_DURATION_S = 4.0
_NARRATOR_NAME = "narrator"

_LANGUAGE_RE = re.compile(r"^[a-z]{2}$")
_JOB_NAME_RE = re.compile(r"[^a-z0-9]+")


class InitError(Exception):
    """Raised when a project cannot be scaffolded (bad input, clash)."""


def _check_language(code: str, what: str) -> None:
    if not _LANGUAGE_RE.match(code):
        raise InitError(
            f"{what} language must be a two-letter lowercase code like 'en' or 'hi', got {code!r}"
        )


def _check_backend(backend: str) -> None:
    if backend not in ("dictionary", "argos"):
        raise InitError(f"unknown translation backend {backend!r} (dictionary or argos)")


def _job_name_for(path: Path) -> str:
    """A manifest-safe job name from a file stem: lowercase slug."""
    slug = _JOB_NAME_RE.sub("-", path.stem.lower()).strip("-")
    if not slug:
        raise InitError(f"cannot make a job name from {path.name!r}")
    return slug


def _example_script_text() -> str:
    """The example job's word list: JSON ``{text, start, end}`` words.

    Timings spread the words evenly across the example tone so the
    scripted transcriber's words land inside the tone's speech span
    (VAD drops words that fall in silence).
    """
    words = _EXAMPLE_WORDS.split()
    step = _EXAMPLE_DURATION_S / len(words)
    entries = [
        {
            "text": word,
            "start": round(i * step, 3),
            "end": round((i + 1) * step - 0.02, 3),
        }
        for i, word in enumerate(words)
    ]
    return json.dumps(entries, indent=2) + "\n"


def _config_text(
    *,
    source_language: str,
    target_language: str,
    backend: str,
    style: str | None,
) -> str:
    enhance = EnhanceConfig()
    lines = [
        "# Dubbing project config — created by `dubber init`.",
        "# Every key is optional; explicit CLI flags always win over the file.",
        "",
        "[dubber]",
        f'target_language = "{target_language}"',
        f'source_language = "{source_language}"',
        f'backend = "{backend}"',
        'voice_library = "voices.json"',
    ]
    if style is not None:
        lines.append(f'style = "{style}"')
    lines += [
        "",
        "[dubber.enhance]",
        f"target = {enhance.target}",
        f"ceiling = {enhance.ceiling}",
        f"high_pass = {enhance.high_pass}",
        f"gate = {enhance.gate}",
        "",
    ]
    return "\n".join(lines)


def _manifest_text(
    *,
    source_language: str,
    target_language: str,
    translation_backend: str,
    seeded_job: dict | None,
) -> str:
    jobs: list[dict] = [
        {
            "name": "example",
            "video": _EXAMPLE_WAV,
            "script": _EXAMPLE_SCRIPT,
            "profile": _PROFILE_FILE,
            "backend": "scripted",
            # The example source is audio-only, so there is no video
            # track to mux the dub back onto.
            "mux": False,
        }
    ]
    if seeded_job is not None:
        jobs.append(seeded_job)
    manifest = {
        "defaults": {
            "target": target_language,
            "source": source_language,
            "backend": translation_backend,
        },
        "jobs": jobs,
    }
    return json.dumps(manifest, indent=2) + "\n"


def _readme_text(
    *,
    project_name: str,
    source_language: str,
    target_language: str,
    seeded_name: str | None,
) -> str:
    lines = [
        f"# {project_name}",
        "",
        "A video-dubbing project, scaffolded by `dubber init`.",
        "",
        "## Layout",
        "",
        (
            "- `dubber.toml` — defaults for every command (target language, "
            "voice library, style, enhancement)."
        ),
        ("- `manifest.json` — batch jobs; run them all with `dubber batch manifest.json`."),
        "- `voices.json` — the voice library (`dubber voices` lists it).",
        ("- `voices/narrator.json` — the narrator profile the example job dubs with."),
        "- `source/` — put your videos here.",
        "- `out/` — batch outputs land in `out/<job-name>/`.",
        "",
        "## Try it",
        "",
        "```sh",
        "dubber batch manifest.json   # dubs the example job (offline, no model)",
        "dubber voices                # lists the library (one voice: narrator)",
        "dubber config                # shows the effective configuration",
        "```",
        "",
        "## Make it yours",
        "",
        (
            "1. Drop a video into `source/` and add a job to `manifest.json` "
            "(copy the example job, point `video` at your file)."
        ),
        (
            "2. Clone a real voice: `dubber clone --name <name> --out "
            "voices/<name>.json <reference.wav>` "
            "and point the job's `profile` at the new profile file."
        ),
        (
            "3. Real transcription needs Whisper — change the job's `backend` "
            'to `"whisper"` and drop the `script` key.'
        ),
        "",
    ]
    if seeded_name is not None:
        lines += [
            (
                f"The `{seeded_name}` job was seeded from your source file; "
                "it already uses the Whisper backend, so install Whisper "
                "before running it (or give it a `script` and switch it to "
                '`"scripted"` for an offline run).'
            ),
            "",
        ]
    lines += [
        (f"Source language: `{source_language}`, target language: `{target_language}`."),
        "",
    ]
    return "\n".join(lines)


def scaffold_project(
    root: str | Path,
    *,
    source_language: str = "en",
    target_language: str = "hi",
    backend: str = "dictionary",
    style: str | None = None,
    source: str | Path | None = None,
    force: bool = False,
) -> list[Path]:
    """Create a dubbing project directory and return the files written.

    ``root`` is created (with ``source/``, ``voices/`` and ``out/`` inside)
    when it does not exist. Refuses to touch a non-empty directory unless
    ``force`` is set; with ``force`` only the files this scaffold owns
    are overwritten, anything else in the directory is left alone.
    ``source`` seeds an extra batch job for an existing video file.
    """
    _check_language(source_language, "source")
    _check_language(target_language, "target")
    _check_backend(backend)
    if style is not None:
        try:
            get_style(style)
        except StyleError as exc:
            raise InitError(str(exc)) from exc

    root = Path(root)
    source_path = Path(source) if source is not None else None
    if source_path is not None and not source_path.is_file():
        raise InitError(f"source file not found: {source}")

    if root.exists() and not root.is_dir():
        raise InitError(f"project path exists and is not a directory: {root}")
    if root.is_dir() and any(root.iterdir()) and not force:
        blocking = sorted(p.name for p in root.iterdir())
        raise InitError(
            f"directory {root} is not empty ({', '.join(blocking)}); "
            "pass force=True to scaffold into it anyway"
        )

    # The example tone is the narrator's reference voice: one short sine
    # so the scaffold needs no real recording to be runnable.
    tone = make_sine(_EXAMPLE_TONE_HZ, _EXAMPLE_DURATION_S)
    profile = SpeakerProfile(
        name=_NARRATOR_NAME,
        embedding=extract_embedding(tone),
        base_pitch_hz=_EXAMPLE_TONE_HZ,
        speaking_rate_wpm=150.0,
        reference_files=[_EXAMPLE_WAV],
    )
    library = VoiceLibrary({profile.name: profile})

    seeded_job: dict | None = None
    if source_path is not None:
        try:
            video_rel = source_path.resolve().relative_to(root.resolve())
            video_value = video_rel.as_posix()
        except ValueError:
            video_value = str(source_path)
        seeded_job = {
            "name": _job_name_for(source_path),
            "video": video_value,
            "profile": _PROFILE_FILE,
            "backend": "whisper",
        }

    (root / "source").mkdir(parents=True, exist_ok=True)
    (root / "voices").mkdir(parents=True, exist_ok=True)
    (root / "out").mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    save_wav(str(root / _EXAMPLE_WAV), tone)
    written.append(root / _EXAMPLE_WAV)
    (root / _EXAMPLE_SCRIPT).write_text(_example_script_text(), encoding="utf-8")
    written.append(root / _EXAMPLE_SCRIPT)
    (root / _PROFILE_FILE).write_text(profile.to_json() + "\n", encoding="utf-8")
    written.append(root / _PROFILE_FILE)
    (root / _LIBRARY_FILE).write_text(library.to_json() + "\n", encoding="utf-8")
    written.append(root / _LIBRARY_FILE)
    (root / _CONFIG_FILE).write_text(
        _config_text(
            source_language=source_language,
            target_language=target_language,
            backend=backend,
            style=style,
        ),
        encoding="utf-8",
    )
    written.append(root / _CONFIG_FILE)
    (root / _MANIFEST_FILE).write_text(
        _manifest_text(
            source_language=source_language,
            target_language=target_language,
            translation_backend=backend,
            seeded_job=seeded_job,
        ),
        encoding="utf-8",
    )
    written.append(root / _MANIFEST_FILE)
    (root / _README_FILE).write_text(
        _readme_text(
            project_name=root.name or "dubbing project",
            source_language=source_language,
            target_language=target_language,
            seeded_name=seeded_job["name"] if seeded_job else None,
        ),
        encoding="utf-8",
    )
    written.append(root / _README_FILE)
    (root / _GITIGNORE_FILE).write_text("out/\n", encoding="utf-8")
    written.append(root / _GITIGNORE_FILE)
    return written

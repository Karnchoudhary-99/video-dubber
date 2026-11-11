"""Portable export bundles for finished dubs.

A dub is only useful once it leaves this machine: the dubbed audio, the
retimed subtitles, the quality report and (optionally) the original or the
rendered video, packed together so a reviewer can check what they got
without re-running anything. This module builds that package:

* :class:`ExportManifest` — provenance for the bundle: which source file,
  which languages, which speakers, the quality verdict, and a SHA-256
  inventory of every file inside (JSON + Markdown, round-trippable).
* :func:`export_bundle` — copy the files into a directory with
  ``manifest.json`` and a human-readable ``README.md``.
* :func:`export_zip` — the same bundle as a single ``.zip`` file.
* :func:`verify_bundle` — re-hash a directory bundle against its manifest
  and report anything missing, changed, or extra.

Two files from different roles can share a basename (the dub and the
original are both often ``clip.wav``), so colliding names are prefixed
with their role (``original_clip.wav``) instead of silently overwriting
each other. A bundle never contains an absolute path: the manifest
records basenames only, so it stays valid if the folder is moved.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from video_dubber import __version__

__all__ = [
    "ExportError",
    "ExportFile",
    "ExportManifest",
    "build_manifest",
    "export_bundle",
    "export_zip",
    "hash_file",
    "verify_bundle",
]

# Role -> human label, in the order files are listed in the manifest.
ROLES = {
    "dubbed_audio": "Dubbed audio",
    "subtitles_srt": "Subtitles (SRT)",
    "subtitles_vtt": "Subtitles (WebVTT)",
    "quality_report": "Quality report",
    "original": "Original media",
    "rendered_video": "Rendered video",
}


class ExportError(Exception):
    """Raised when a bundle cannot be built, written, or verified."""


def hash_file(path: str | Path) -> str:
    """SHA-256 hex digest of a file, streamed so big videos stay cheap."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ExportFile:
    """One file inside a bundle: its role, name, and checksum."""

    role: str
    filename: str  # basename inside the bundle (never a path)
    sha256: str
    size_bytes: int
    source: str = ""  # original basename on disk, for provenance

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExportFile:
        if not isinstance(data, dict):
            raise TypeError("export file entry must be a JSON object")
        try:
            return cls(
                role=str(data["role"]),
                filename=str(data["filename"]),
                sha256=str(data["sha256"]),
                size_bytes=int(data["size_bytes"]),
                source=str(data.get("source", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"not a valid export file entry: {exc}") from exc


@dataclass
class ExportManifest:
    """Provenance + file inventory for one export bundle."""

    name: str
    created_utc: str  # ISO 8601, e.g. 2026-11-11T13:05:00+00:00
    dubber_version: str
    source_language: str
    target_language: str
    source_file: str | None = None  # basename of the dubbed source, if any
    verdict: str | None = None  # "good" | "review", None when no report
    mos_proxy: float | None = None  # heuristic 1-5, None when no report
    speakers: list[str] = field(default_factory=list)
    files: list[ExportFile] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "created_utc": self.created_utc,
            "dubber_version": self.dubber_version,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "source_file": self.source_file,
            "verdict": self.verdict,
            "mos_proxy": self.mos_proxy,
            "speakers": list(self.speakers),
            "files": [entry.to_dict() for entry in self.files],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        lines = [
            f"# Export bundle: {self.name}",
            "",
            f"- Created: {self.created_utc}",
            f"- video-dubber: {self.dubber_version}",
            f"- Languages: {self.source_language} -> {self.target_language}",
        ]
        if self.source_file:
            lines.append(f"- Source file: {self.source_file}")
        if self.verdict is not None:
            lines.append(f"- Verdict: **{self.verdict}**")
        if self.mos_proxy is not None:
            lines.append(f"- MOS proxy (heuristic, 1-5): {self.mos_proxy:.2f}")
        if self.speakers:
            lines.append(f"- Speakers: {', '.join(self.speakers)}")
        lines.append("")
        lines.append("## Files")
        lines.append("")
        for entry in self.files:
            label = ROLES.get(entry.role, entry.role)
            lines.append(
                f"- {label}: `{entry.filename}` ({entry.size_bytes} bytes, "
                f"sha256 `{entry.sha256[:12]}...`)"
            )
        lines.append("")
        lines.append(
            "Verify with `dubber export --verify <bundle-dir>` "
            "or `video_dubber.export.verify_bundle`."
        )
        return "\n".join(lines) + "\n"

    @classmethod
    def from_dict(cls, data: dict) -> ExportManifest:
        """Rebuild from :meth:`to_json` output."""
        if not isinstance(data, dict):
            raise TypeError("export manifest must be a JSON object")
        try:
            return cls(
                name=str(data["name"]),
                created_utc=str(data["created_utc"]),
                dubber_version=str(data["dubber_version"]),
                source_language=str(data["source_language"]),
                target_language=str(data["target_language"]),
                source_file=data.get("source_file"),
                verdict=data.get("verdict"),
                mos_proxy=data.get("mos_proxy"),
                speakers=[str(s) for s in data.get("speakers", [])],
                files=[ExportFile.from_dict(row) for row in data.get("files", [])],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"not a valid export manifest: {exc}") from exc

    @classmethod
    def from_json(cls, text: str) -> ExportManifest:
        """Rebuild from :meth:`to_json` output."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"not valid JSON: {exc}") from exc
        return cls.from_dict(data)


def _unique_names(paths: dict[str, Path]) -> dict[str, str]:
    """Bundle filename per role, prefixing colliding basenames with the role."""
    seen: dict[str, str] = {}
    out: dict[str, str] = {}
    for role, path in paths.items():
        name = path.name
        if name in seen:
            name = f"{role}_{name}"
            if name in seen:  # pragma: no cover - same role twice is a caller bug
                raise ExportError(f"duplicate bundle filename: {name}")
        seen[name] = role
        out[role] = name
    return out


def build_manifest(
    name: str,
    *,
    dub_audio: str | Path,
    target: str,
    source: str = "en",
    subtitles_srt: str | Path | None = None,
    subtitles_vtt: str | Path | None = None,
    report: str | Path | None = None,
    original: str | Path | None = None,
    rendered_video: str | Path | None = None,
    speakers: list[str] | None = None,
    created_utc: str | None = None,
) -> tuple[ExportManifest, dict[str, Path]]:
    """Validate the dub's files and build its manifest.

    ``report`` is a :class:`~video_dubber.assemble.DubReport` JSON file;
    its verdict and MOS proxy are copied into the manifest. Returns the
    manifest plus a ``{role: path}`` map for :func:`export_bundle` /
    :func:`export_zip`. Every path must exist, or :class:`ExportError`
    names the missing one.
    """
    from video_dubber.assemble import DubReport

    if not name.strip():
        raise ExportError("bundle name must not be empty")
    if not target.strip():
        raise ExportError("target language must not be empty")

    paths: dict[str, Path] = {"dubbed_audio": Path(dub_audio)}
    if subtitles_srt is not None:
        paths["subtitles_srt"] = Path(subtitles_srt)
    if subtitles_vtt is not None:
        paths["subtitles_vtt"] = Path(subtitles_vtt)
    if report is not None:
        paths["quality_report"] = Path(report)
    if original is not None:
        paths["original"] = Path(original)
    if rendered_video is not None:
        paths["rendered_video"] = Path(rendered_video)
    for role, path in paths.items():
        if not path.is_file():
            raise ExportError(f"{ROLES.get(role, role)} not found: {path}")

    verdict = None
    mos_proxy = None
    if report is not None:
        dub_report = DubReport.from_json(Path(report).read_text(encoding="utf-8"))
        verdict = dub_report.verdict
        mos_proxy = dub_report.mos_proxy

    names = _unique_names(paths)
    files = [
        ExportFile(
            role=role,
            filename=names[role],
            sha256=hash_file(path),
            size_bytes=path.stat().st_size,
            source=path.name,
        )
        for role, path in paths.items()
    ]
    manifest = ExportManifest(
        name=name,
        created_utc=created_utc or datetime.now(UTC).isoformat(),
        dubber_version=__version__,
        source_language=source,
        target_language=target,
        source_file=Path(original).name if original is not None else None,
        verdict=verdict,
        mos_proxy=mos_proxy,
        speakers=list(speakers or []),
        files=files,
    )
    return manifest, paths


def export_bundle(
    manifest: ExportManifest,
    sources: dict[str, Path],
    out_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a directory bundle: the files + manifest.json + README.md.

    Refuses to write into a non-empty directory unless ``overwrite`` is
    set. With ``overwrite``, files from an older bundle that are not part
    of the new one are left in place (``verify_bundle`` flags them as
    unlisted). Returns the path of the written manifest.
    """
    out = Path(out_dir)
    if out.exists() and not out.is_dir():
        raise ExportError(f"export path is not a directory: {out}")
    if out.is_dir() and any(out.iterdir()) and not overwrite:
        raise ExportError(f"directory is not empty (use --force to overwrite): {out}")
    out.mkdir(parents=True, exist_ok=True)
    by_role = {entry.role: entry.filename for entry in manifest.files}
    for role, path in sources.items():
        shutil.copyfile(path, out / by_role[role])
    manifest_path = out / "manifest.json"
    manifest_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    (out / "README.md").write_text(manifest.to_markdown(), encoding="utf-8")
    return manifest_path


def export_zip(
    manifest: ExportManifest,
    sources: dict[str, Path],
    out_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the same bundle as a single ``.zip`` file (deflated)."""
    out = Path(out_path)
    if out.suffix.lower() != ".zip":
        raise ExportError(f"zip export path must end in .zip: {out}")
    if out.exists() and not overwrite:
        raise ExportError(f"file already exists (use --force to overwrite): {out}")
    if out.parent != Path("."):
        out.parent.mkdir(parents=True, exist_ok=True)
    by_role = {entry.role: entry.filename for entry in manifest.files}
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for role, path in sources.items():
            zf.write(path, arcname=by_role[role])
        zf.writestr("manifest.json", manifest.to_json() + "\n")
        zf.writestr("README.md", manifest.to_markdown())
    return out


def verify_bundle(directory: str | Path) -> list[str]:
    """Re-hash a directory bundle against its manifest.

    Returns a list of problems (empty means the bundle is intact):
    missing files, hash mismatches, and files present in the directory
    that the manifest does not list (ignoring manifest.json/README.md).
    """
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return [f"no manifest.json in {directory}"]
    try:
        manifest = ExportManifest.from_json(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return [f"manifest.json is invalid: {exc}"]
    problems: list[str] = []
    listed = {"manifest.json", "README.md"}
    for entry in manifest.files:
        listed.add(entry.filename)
        path = directory / entry.filename
        if not path.is_file():
            problems.append(f"missing: {entry.filename} ({entry.role})")
            continue
        actual = hash_file(path)
        if actual != entry.sha256:
            problems.append(
                f"changed: {entry.filename} ({entry.role}) — "
                f"sha256 {actual[:12]}... != manifest {entry.sha256[:12]}..."
            )
        if path.stat().st_size != entry.size_bytes:
            problems.append(
                f"size changed: {entry.filename} ({entry.role}) — "
                f"{path.stat().st_size} bytes, manifest says {entry.size_bytes}"
            )
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name not in listed:
            problems.append(f"unlisted file in bundle: {path.name}")
    return problems

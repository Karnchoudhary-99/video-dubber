"""Demo: scaffold a project with ``dubber init`` and run its example job.

Scaffolds a project into a temp dir, shows the created files, then
runs the scaffolded manifest through the CLI — the example job dubs
offline (synthetic tone + scripted transcriber) and lands in
``out/example/``. Also shows the refusal when the directory is not
empty, and the ``--force`` override.

No model, no network needed — run it with ``python examples/demo_init.py``.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.cli import main as cli_main


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "my-first-dub"
        print("$ dubber init my-first-dub --target es")
        assert cli_main(["init", str(root), "--target", "es"]) == 0
        print()
        print("project files:")
        for path in sorted(root.rglob("*")):
            if path.is_file():
                print(f"  {path.relative_to(root)}")
        print()
        print("$ cd my-first-dub && dubber batch manifest.json")
        import os

        os.chdir(root)
        assert cli_main(["batch", "manifest.json", "--out-dir", "out"]) == 0
        print()
        print("batch outputs:")
        for path in sorted((root / "out").rglob("*")):
            if path.is_file():
                print(f"  {path.relative_to(root)}")
        print()
        print("$ dubber init my-first-dub   # already exists")
        assert cli_main(["init", str(root)]) == 1
        print("$ dubber init my-first-dub --force   # scaffold-owned files only")
        assert cli_main(["init", str(root), "--force"]) == 0
        print()
        print("done — the example job dubbed offline, no model needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

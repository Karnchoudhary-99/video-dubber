"""Demo: project status before, during and after a batch run.

Scaffolds a project, shows its status (example job not started),
runs the example batch job, shows the status again (done, with the
report verdict and MOS proxy), then touches the source file so the
dub goes stale and shows the status a third time.

No model, no network needed — run it with ``python examples/demo_status.py``.
"""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.cli import main as cli_main


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "my-first-dub"
        assert cli_main(["init", str(root)]) == 0
        import os

        os.chdir(root)
        print("$ dubber status   # fresh project")
        assert cli_main(["status"]) == 0
        print()
        print("$ dubber batch manifest.json --out-dir out")
        assert cli_main(["batch", "manifest.json", "--out-dir", "out"]) == 0
        print()
        print("$ dubber status   # after the batch")
        assert cli_main(["status"]) == 0
        print()
        time.sleep(0.02)
        (root / "source" / "example.wav").touch()
        print("$ touch source/example.wav && dubber status   # source changed")
        assert cli_main(["status"]) == 0
        print()
        print("done — not started -> done -> stale, all from the outputs on disk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

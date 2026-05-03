from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "demo_app" / "streamlit_app.py"


def main() -> int:
    env = os.environ.copy()
    env.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.fileWatcherType",
        env["STREAMLIT_SERVER_FILE_WATCHER_TYPE"],
    ]
    command.extend(sys.argv[1:])
    return subprocess.call(command, cwd=PROJECT_ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())

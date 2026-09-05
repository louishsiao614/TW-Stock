"""以 `python run.py` 啟動台股量化研究儀表板。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app = Path(__file__).with_name("app.py")
    command = [sys.executable, "-m", "streamlit", "run", str(app)]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()

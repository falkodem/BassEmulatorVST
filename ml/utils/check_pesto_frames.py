"""Совместимый CLI; реализация находится в ml/pesto/check_pesto_frames.py."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "pesto" / "check_pesto_frames.py"), run_name="__main__")

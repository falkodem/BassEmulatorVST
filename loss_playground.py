"""Совместимый CLI; реализация находится в ml/pesto/loss_playground.py."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "ml" / "pesto" / "loss_playground.py"), run_name="__main__")

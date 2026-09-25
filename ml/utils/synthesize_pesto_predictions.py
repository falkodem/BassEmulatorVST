"""Совместимый CLI; реализация находится в ml/pitch_eval/synthesize_pesto_predictions.py."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "pitch_eval" / "synthesize_pesto_predictions.py"), run_name="__main__")

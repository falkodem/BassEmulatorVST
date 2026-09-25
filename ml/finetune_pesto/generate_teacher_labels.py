"""Совместимый CLI; реализация находится в ml/pesto/finetune/generate_teacher_labels.py."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "pesto" / "finetune" / "generate_teacher_labels.py"), run_name="__main__")

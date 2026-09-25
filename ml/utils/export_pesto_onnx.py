"""Совместимый CLI; реализация находится в ml/pesto/export_onnx.py."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "pesto" / "export_onnx.py"), run_name="__main__")

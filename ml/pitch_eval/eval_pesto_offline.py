"""Compare offline PESTO checkpoints on note-labelled guitar WAVs."""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pesto.loader import load_model

from ml.pesto.finetune.generate_teacher_labels import infer_teacher_blockwise
from ml.pitch_eval.eval_pesto_onnx import (
    activity_mask,
    aggregate,
    compute_metrics,
    note_frequency,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True,
                        help="Packaged PESTO name or checkpoint path")
    parser.add_argument("--input", type=Path, default=Path("data/v0/guitar"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--frames-per-block", type=int, default=4096)
    args = parser.parse_args()

    wav_paths = sorted(
        path for path in args.input.glob("*.wav") if " render " not in path.stem
    )
    if not wav_paths:
        raise FileNotFoundError(f"No note-labelled WAVs in {args.input}")
    output = args.output or Path("runs/pitch_eval") / f"offline_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=True)

    per_file_rows = []
    summary_rows = []
    for model_name in args.models:
        label = Path(model_name).stem if Path(model_name).is_file() else model_name
        model = load_model(model_name, step_size=10.0, sampling_rate=44100)
        model = model.to(args.device).eval()
        rows_by_gate = {"all": [], "confidence_0.5": []}
        samples_by_gate = {"all": [], "confidence_0.5": []}

        for wav_path in wav_paths:
            note, expected_hz = note_frequency(wav_path)
            audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
            if sample_rate != 44100:
                raise ValueError(f"{wav_path.name}: expected 44100 Hz, got {sample_rate}")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            active, _rms_db, _threshold_db = activity_mask(audio, 441, 10.0)

            started = time.perf_counter()
            targets = infer_teacher_blockwise(
                model, wav_path, sample_rate=sample_rate, hop_samples=441,
                frames_per_block=args.frames_per_block, device=torch.device(args.device),
            )
            elapsed = time.perf_counter() - started
            f0, confidence = targets["f0_hz"], targets["confidence"]
            if active.size != f0.size:
                raise RuntimeError(f"{wav_path.name}: activity and pitch frames differ")

            for gate, threshold in (("all", 0.0), ("confidence_0.5", 0.5)):
                metrics, samples = compute_metrics(
                    f0, confidence, active, expected_hz,
                    confidence_threshold=threshold,
                    warmup_frames=10, frame_duration_s=0.01,
                    attack_window_ms=200.0, settling_tolerance_cents=50.0,
                    settling_hold_ms=50.0, minimum_onset_gap_ms=100.0,
                )
                row = {
                    "model": label, "gate": gate, "file": wav_path.name,
                    "note": note, "expected_hz": expected_hz,
                    **metrics, "inference_seconds": elapsed,
                }
                per_file_rows.append(row)
                rows_by_gate[gate].append(row)
                samples_by_gate[gate].append(samples)
            print(f"{label}: {wav_path.name}", flush=True)

        for gate in rows_by_gate:
            summary = {"gate": gate, **aggregate(
                label, rows_by_gate[gate], samples_by_gate[gate]
            )}
            cents = np.concatenate([part.cents for part in samples_by_gate[gate]])
            offset = float(np.median(cents))
            corrected = np.abs(cents - offset)
            summary.update({
                "diagnostic_offset_cents": offset,
                "median_abs_cents_after_offset": float(np.median(corrected)),
                "within_50c_after_offset_pct": float(100.0 * np.mean(corrected <= 50.0)),
            })
            summary_rows.append(summary)

    write_csv(output / "per_file.csv", per_file_rows)
    write_csv(output / "summary.csv", summary_rows)
    (output / "config.json").write_text(json.dumps({
        "models": args.models, "input": str(args.input), "device": args.device,
        "frames_per_block": args.frames_per_block,
        "sample_rate": 44100, "hop_samples": 441,
        "activity_margin_db": 10.0, "warmup_frames": 10,
        "confidence_thresholds": {"all": 0.0, "confidence_0.5": 0.5},
    }, indent=2))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()

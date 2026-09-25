"""Synthesize audio from frame-level PESTO predictions saved by eval_pesto_onnx."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("synthesize_pesto_predictions")

REQUIRED_COLUMNS = {
    "model", "file", "frame_index", "time_s", "frame_duration_s",
    "predicted_hz", "voiced", "eligible", "rms_active", "active_voiced",
}
GATE_COLUMNS = {
    "voiced": "voiced",
    "active-voiced": "active_voiced",
    "rms": "rms_active",
}


def synthesize_curve(times: np.ndarray, predicted_hz: np.ndarray, gate: np.ndarray,
                     duration_s: float, sample_rate: int, amplitude: float,
                     pitch_scale: float, waveform: str) -> np.ndarray:
    n_samples = int(round(duration_s * sample_rate))
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    valid = np.isfinite(predicted_hz) & (predicted_hz > 0)
    gate = np.asarray(gate, dtype=bool) & valid
    if not valid.any():
        return np.zeros(n_samples, dtype=np.float32)

    sample_times = np.arange(n_samples, dtype=np.float64) / sample_rate
    f0 = np.interp(sample_times, times[valid], predicted_hz[valid]) * pitch_scale
    f0 = np.clip(f0, 0.0, 0.49 * sample_rate)
    phase = np.cumsum(2.0 * np.pi * f0 / sample_rate)
    if waveform == "sine":
        carrier = np.sin(phase)
    else:
        carrier = 2.0 * np.mod(phase / (2.0 * np.pi), 1.0) - 1.0

    # Linear interpolation gives approximately one-frame click-free fades.
    envelope = np.interp(sample_times, times, gate.astype(np.float64))
    audio = amplitude * carrier * np.clip(envelope, 0.0, 1.0)
    return np.asarray(np.clip(audio, -1.0, 1.0), dtype=np.float32)


def select_gate(group: pd.DataFrame, gate_name: str) -> np.ndarray:
    if gate_name == "none":
        return group["eligible"].to_numpy(dtype=bool)
    gate = group[GATE_COLUMNS[gate_name]].to_numpy(dtype=bool)
    return gate & group["eligible"].to_numpy(dtype=bool)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path,
                        help="Evaluation run containing frames.csv.gz")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory; default: <run>/synth")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names to synthesize; default: all")
    parser.add_argument("--files", nargs="+", default=None,
                        help="Source WAV names to synthesize; default: all")
    parser.add_argument("--gate", choices=(*GATE_COLUMNS, "none"), default="voiced")
    parser.add_argument("--waveform", choices=("sine", "sawtooth"), default="sine")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--amplitude", type=float, default=0.25)
    parser.add_argument("--pitch-scale", type=float, default=1.0,
                        help="Frequency multiplier; use 0.5 to hear the bass octave")
    args = parser.parse_args()

    frames_path = args.run / "frames.csv.gz"
    if not frames_path.is_file():
        parser.error(f"missing frame data: {frames_path}")
    if args.sample_rate <= 0:
        parser.error("--sample-rate must be positive")
    if not 0.0 < args.amplitude <= 1.0:
        parser.error("--amplitude must be in (0, 1]")
    if args.pitch_scale <= 0.0:
        parser.error("--pitch-scale must be positive")

    output = args.output or args.run / "synth"
    output.mkdir(parents=True, exist_ok=True)
    frames = pd.read_csv(frames_path)
    missing = REQUIRED_COLUMNS - set(frames.columns)
    if missing:
        parser.error(f"frames file is missing columns: {sorted(missing)}")
    frames["model"] = frames["model"].astype(str)

    if args.models:
        unknown = sorted(set(args.models) - set(frames["model"]))
        if unknown:
            parser.error(f"unknown models: {unknown}")
        frames = frames[frames["model"].isin(args.models)]
    if args.files:
        unknown = sorted(set(args.files) - set(frames["file"]))
        if unknown:
            parser.error(f"unknown files: {unknown}")
        frames = frames[frames["file"].isin(args.files)]

    manifest = []
    groups = frames.groupby(["model", "file"], sort=True)
    log.info("Synthesizing %d model/file curves into %s", groups.ngroups, output)
    for (model, filename), group in groups:
        group = group.sort_values("frame_index")
        times = group["time_s"].to_numpy(dtype=np.float64)
        predicted_hz = group["predicted_hz"].to_numpy(dtype=np.float64)
        gate = select_gate(group, args.gate)
        frame_duration_s = float(group["frame_duration_s"].iloc[0])
        duration_s = float(times[-1] + frame_duration_s)
        audio = synthesize_curve(
            times, predicted_hz, gate, duration_s, args.sample_rate,
            args.amplitude, args.pitch_scale, args.waveform,
        )

        model_dir = output / str(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        output_path = model_dir / filename
        sf.write(output_path, audio, args.sample_rate, subtype="PCM_16")
        manifest.append({
            "model": model,
            "source_file": filename,
            "output_file": str(output_path.relative_to(output)),
            "frames": len(group),
            "duration_s": duration_s,
            "gated_frames_pct": 100.0 * float(gate.mean()),
            "peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
        })
        log.info("  %-16s %-24s %7.2f s", model, filename, duration_s)

    pd.DataFrame(manifest).to_csv(output / "manifest.csv", index=False)
    (output / "config.json").write_text(json.dumps({
        "run": str(args.run),
        "frames": str(frames_path),
        "models": args.models or sorted(frames["model"].unique().tolist()),
        "files": args.files or sorted(frames["file"].unique().tolist()),
        "gate": args.gate,
        "waveform": args.waveform,
        "sample_rate": args.sample_rate,
        "amplitude": args.amplitude,
        "pitch_scale": args.pitch_scale,
        "subtype": "PCM_16",
    }, indent=2))
    log.info("Wrote %d WAV files", len(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

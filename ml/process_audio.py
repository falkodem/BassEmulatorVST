#!/usr/bin/env python3
"""
Offline inference: apply a trained model to guitar audio.
Emulates plugin behaviour (sliding window + overlap-add) without the plugin wrapper.

--run  points to a run directory (runs/v0/YYYYMMDD_HHMMSS/).
       The directory must contain best.pt and config.json.
       Architecture and all parameters are read from config.json automatically.

Output: processed/{data_version}_{architecture}_{model_version}/{filename}.wav

Usage:
  # Single file
  python ml/process_audio.py --run runs/v0/20260505_120000 --input path/to/guitar.wav

  # Folder
  python ml/process_audio.py --run runs/v0/20260505_120000 --input data/v0/guitar/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nn_architectures import REGISTRY
from train_config import TrainConfig

ROOT = Path(__file__).resolve().parent.parent


def load_run(run_dir: Path, device: torch.device):
    """Load model and config from a run directory."""
    run_dir = Path(run_dir)
    cfg_path  = run_dir / "config.json"
    ckpt_path = run_dir / "best.pt"

    for p in (cfg_path, ckpt_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    cfg = TrainConfig.load(cfg_path)

    if cfg.architecture not in REGISTRY:
        raise ValueError(
            f"Unknown architecture '{cfg.architecture}'. Available: {list(REGISTRY)}"
        )
    model = REGISTRY[cfg.architecture]().to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Run      : {run_dir}")
    print(f"Arch     : {cfg.architecture}  params={model.param_count:,}")
    print(f"Checkpoint: epoch={ckpt['epoch']}"
          f"  val_rnd={ckpt.get('val_random_loss', float('nan')):.6f}"
          f"  val_file={ckpt.get('val_file_loss', float('nan')):.6f}")
    return model, cfg


def overlap_add(
    model:       torch.nn.Module,
    audio:       np.ndarray,
    window_size: int,
    hop_size:    int,
    device:      torch.device,
) -> np.ndarray:
    """
    Process audio with sliding window + Hann overlap-add.
    Hann window at 50% overlap satisfies COLA → no amplitude distortion.
    """
    n      = len(audio)
    output = np.zeros(n, dtype=np.float32)
    norm   = np.zeros(n, dtype=np.float32)
    hann   = np.hanning(window_size).astype(np.float32)

    with torch.no_grad():
        for start in range(0, n - window_size + 1, hop_size):
            frame = audio[start : start + window_size]
            x     = torch.from_numpy(frame[None, None, :]).to(device)
            y     = model(x).cpu().numpy()[0, 0]
            output[start : start + window_size] += y * hann
            norm  [start : start + window_size] += hann

    mask          = norm > 1e-8
    output[mask] /= norm[mask]
    return output


def process_file(
    model:       torch.nn.Module,
    cfg:         TrainConfig,
    input_path:  Path,
    output_path: Path,
    device:      torch.device,
) -> None:
    audio, sr = sf.read(str(input_path), dtype="float32", always_2d=True)
    audio     = audio.mean(axis=1)
    peak      = np.abs(audio).max()
    if peak > 1e-8:
        audio /= peak                        # peak-normalize (matches training)

    result = overlap_add(model, audio, cfg.window_size, cfg.hop_size, device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), result, sr)
    print(f"  {input_path.name} -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",   required=True, help="Run directory (contains best.pt + config.json)")
    parser.add_argument("--input", required=True, help="WAV file or folder of WAV files")
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir    = Path(args.run)
    input_path = Path(args.input)

    model, cfg = load_run(run_dir, device)

    out_subdir = f"{cfg.data_version}_{cfg.architecture}_{cfg.model_version}"
    out_dir    = ROOT / "processed" / out_subdir

    if input_path.is_dir():
        files = sorted(input_path.glob("*.wav"))
        print(f"\nProcessing {len(files)} files from {input_path}")
        for f in files:
            process_file(model, cfg, f, out_dir / f.name, device)
    elif input_path.is_file():
        print()
        process_file(model, cfg, input_path, out_dir / input_path.name, device)
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)

    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()

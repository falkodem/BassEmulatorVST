"""Generate frame-aligned offline PESTO targets for streaming distillation.

The offline CQT is evaluated in overlapping blocks. Only block interiors are
kept, so internal block boundaries are numerically equivalent to whole-file
offline inference and do not introduce reflect-padding artifacts.
"""
import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pesto.loader import load_model

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ml.pesto.finetune.config import DistillConfig

log = logging.getLogger(__name__)


def _resolve_checkpoint(model_name: str) -> Path:
    path = Path(model_name)
    if path.is_file():
        return path.resolve()
    import pesto
    path = Path(pesto.__file__).parent / "weights" / f"{model_name}.ckpt"
    if not path.is_file():
        raise FileNotFoundError(f"PESTO checkpoint not found: {model_name}")
    return path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_audio_segment(stream: sf.SoundFile, start: int, stop: int) -> np.ndarray:
    stream.seek(start)
    audio = stream.read(stop - start, dtype="float32", always_2d=True)
    if audio.shape[0] != stop - start:
        raise RuntimeError(f"Short read: requested {stop - start}, got {audio.shape[0]}")
    return audio.mean(axis=1, dtype=np.float32)


@torch.inference_mode()
def infer_teacher_blockwise(
    model: torch.nn.Module,
    wav_path: Path,
    *,
    sample_rate: int,
    hop_samples: int,
    frames_per_block: int,
    device: torch.device,
    max_samples: int | None = None,
) -> dict[str, np.ndarray]:
    info = sf.info(wav_path)
    if info.samplerate != sample_rate:
        raise ValueError(f"{wav_path.name}: sample_rate={info.samplerate}, expected {sample_rate}")

    num_samples = info.frames if max_samples is None else min(info.frames, max_samples)
    num_frames = num_samples // hop_samples
    if num_frames < 1:
        raise ValueError(f"{wav_path.name}: not enough samples for one frame")

    cqt = model.preprocessor.hcqt_kernels.cqt_kernels[0]
    context_samples = cqt.kernel_width // 2
    output_dim = model.encoder.hparams["output_dim"]
    activations = np.empty((num_frames, output_dim), dtype=np.float16)
    confidence = np.empty(num_frames, dtype=np.float32)
    f0_hz = np.empty(num_frames, dtype=np.float32)

    with sf.SoundFile(wav_path) as stream:
        for first_frame in range(0, num_frames, frames_per_block):
            last_frame = min(num_frames, first_frame + frames_per_block)
            core_start = first_frame * hop_samples
            core_stop = last_frame * hop_samples
            segment_start = max(0, core_start - context_samples)
            segment_start -= segment_start % hop_samples
            segment_stop = min(num_samples, core_stop + context_samples)
            audio = _read_audio_segment(stream, segment_start, segment_stop)

            tensor = torch.from_numpy(audio).to(device)
            pred, conf, _volume, acts = model(
                tensor, sr=None, convert_to_freq=True, return_activations=True
            )
            segment_first_frame = segment_start // hop_samples
            local_start = first_frame - segment_first_frame
            local_stop = local_start + (last_frame - first_frame)
            if local_stop > acts.shape[0]:
                raise RuntimeError(
                    f"{wav_path.name}: block [{first_frame}, {last_frame}) needs "
                    f"teacher frames [{local_start}, {local_stop}), got {acts.shape[0]}"
                )

            activations[first_frame:last_frame] = (
                acts[local_start:local_stop].detach().cpu().numpy().astype(np.float16)
            )
            confidence[first_frame:last_frame] = (
                conf[local_start:local_stop].detach().cpu().numpy().astype(np.float32)
            )
            f0_hz[first_frame:last_frame] = (
                pred[local_start:local_stop].detach().cpu().numpy().astype(np.float32)
            )
            log.info("%s: teacher frames %d/%d", wav_path.name, last_frame, num_frames)

    return {
        "frame_index": np.arange(num_frames, dtype=np.int64),
        "time_s": np.arange(num_frames, dtype=np.float64) * hop_samples / sample_rate,
        "activations": activations,
        "confidence": confidence,
        "f0_hz": f0_hz,
        "num_samples": np.asarray(num_samples, dtype=np.int64),
    }


def _validate_targets(targets: dict[str, np.ndarray], wav_path: Path) -> None:
    acts = targets["activations"].astype(np.float32)
    confidence = targets["confidence"]
    if not np.isfinite(acts).all() or (acts < 0).any():
        raise ValueError(f"{wav_path.name}: invalid teacher activations")
    sums = acts.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=2e-3):
        raise ValueError(
            f"{wav_path.name}: activation sums outside tolerance: "
            f"min={sums.min():.6f}, max={sums.max():.6f}"
        )
    if not np.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any():
        raise ValueError(f"{wav_path.name}: confidence outside [0, 1]")


def generate_labels(cfg: DistillConfig, overwrite: bool = False) -> Path:
    output_dir = Path(cfg.teacher_labels_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _resolve_checkpoint(cfg.teacher_model)
    device = torch.device(
        "cuda" if cfg.teacher_device == "auto" and torch.cuda.is_available()
        else "cpu" if cfg.teacher_device == "auto"
        else cfg.teacher_device
    )
    step_size_ms = 1000.0 * cfg.chunk_size / cfg.sample_rate
    model = load_model(
        cfg.teacher_model, step_size=step_size_ms, sampling_rate=cfg.sample_rate
    ).to(device).eval()
    max_samples = (
        None if cfg.max_minutes is None
        else int(cfg.max_minutes * 60.0 * cfg.sample_rate)
    )

    manifest = {
        "format_version": 1,
        "teacher_model": cfg.teacher_model,
        "teacher_checkpoint": str(checkpoint_path),
        "teacher_checkpoint_sha256": _sha256(checkpoint_path),
        "sample_rate": cfg.sample_rate,
        "hop_samples": cfg.chunk_size,
        "step_size_ms": step_size_ms,
        "frame_alignment": "offline frame i -> streaming chunk i; no temporal offset",
        "activations_dtype": "float16",
        "confidence_dtype": "float32",
        "config": asdict(cfg),
        "files": [],
    }

    for wav_text in cfg.wav_paths:
        wav_path = Path(wav_text)
        if not wav_path.is_file():
            raise FileNotFoundError(f"Training WAV not found: {wav_path}")
        output_path = output_dir / f"{wav_path.stem}.npz"
        if output_path.exists() and not overwrite:
            log.info("Keeping existing labels: %s", output_path)
            with np.load(output_path) as existing:
                num_frames = int(existing["confidence"].shape[0])
                num_samples = int(existing["num_samples"])
        else:
            targets = infer_teacher_blockwise(
                model, wav_path,
                sample_rate=cfg.sample_rate,
                hop_samples=cfg.chunk_size,
                frames_per_block=cfg.teacher_frames_per_block,
                device=device,
                max_samples=max_samples,
            )
            _validate_targets(targets, wav_path)
            temporary = output_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **targets)
            os.replace(temporary, output_path)
            num_frames = int(targets["confidence"].shape[0])
            num_samples = int(targets["num_samples"])
            log.info("Saved %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)

        info = sf.info(wav_path)
        manifest["files"].append({
            "source_path": str(wav_path.resolve()),
            "source_frames": info.frames,
            "used_samples": num_samples,
            "num_frames": num_frames,
            "labels_file": output_path.name,
        })

    manifest_path = output_dir / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".tmp.json")
    temporary_manifest.write_text(json.dumps(manifest, indent=2))
    os.replace(temporary_manifest, manifest_path)
    log.info("Saved manifest: %s", manifest_path)
    return manifest_path


def parse_args() -> tuple[DistillConfig, bool]:
    cfg = DistillConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", nargs="+", default=None, help="override WAV paths")
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--frames-per-block", type=int, default=None)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.wav is not None:
        cfg.wav_paths = args.wav
    if args.teacher_model is not None:
        cfg.teacher_model = args.teacher_model
    if args.output_dir is not None:
        cfg.teacher_labels_dir = str(args.output_dir)
    if args.device is not None:
        cfg.teacher_device = args.device
    if args.frames_per_block is not None:
        cfg.teacher_frames_per_block = args.frames_per_block
    if args.max_minutes is not None:
        cfg.max_minutes = args.max_minutes
    cfg.__post_init__()
    return cfg, args.overwrite


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg, overwrite = parse_args()
    generate_labels(cfg, overwrite=overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train streaming PESTO from frame-aligned offline teacher targets.

The two modes are intentionally separate:
- confidence: freeze pitch encoder and optimize soft BCE
- pitch: freeze confidence classifier and optimize teacher-weighted KL only
"""
import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pesto.model import ConfidenceClassifier
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.finetune_pesto.config import DistillConfig
from ml.finetune_pesto.distillation_datamodule import PESTODistillationDataModule
from ml.finetune_pesto.distillation_module import PESTODistillationModule
from ml.finetune_pesto.vendor.networks.resnet1d import Resnet1d


log = logging.getLogger(__name__)


def _to_python(value):
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _resolve_checkpoint(model_name: str) -> Path:
    path = Path(model_name)
    if path.is_file():
        return path
    import pesto
    path = Path(pesto.__file__).parent / "weights" / f"{model_name}.ckpt"
    if not path.is_file():
        raise FileNotFoundError(f"PESTO checkpoint not found: {model_name}")
    return path


def _confidence_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    prefix = "confidence."
    return {
        key.removeprefix(prefix): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    }


def build_distillation_module(
    cfg: DistillConfig,
) -> tuple[PESTODistillationModule, dict]:
    student_path = _resolve_checkpoint(cfg.student_checkpoint)
    student_checkpoint = torch.load(
        student_path, map_location="cpu", weights_only=False
    )
    encoder_hparams = _to_python(student_checkpoint["hparams"]["encoder"])
    pitch_shift_hparams = _to_python(student_checkpoint["hparams"]["pitch_shift"])
    reduction = student_checkpoint["hparams"].get("reduction", "alwa")

    module = PESTODistillationModule(
        encoder=Resnet1d(**encoder_hparams),
        confidence=ConfidenceClassifier(),
        mode=cfg.mode,
        min_pitch_shift_steps=int(pitch_shift_hparams["min_steps"]),
        max_pitch_shift_steps=int(pitch_shift_hparams["max_steps"]),
        bins_per_semitone=cfg.bins_per_semitone,
        reduction=reduction,
        lr=cfg.confidence_lr if cfg.mode == "confidence" else cfg.pitch_lr,
        weight_decay=cfg.weight_decay,
        scheduler_epochs=cfg.epochs,
        teacher_confidence_power=cfg.teacher_confidence_power,
    )

    missing, unexpected = module.load_state_dict(
        student_checkpoint["state_dict"], strict=False
    )
    log.info(
        "Loaded student %s: missing=%d unexpected=%d",
        student_path,
        len(missing),
        len(unexpected),
    )

    student_confidence = _confidence_state(student_checkpoint)
    if not student_confidence:
        teacher_path = _resolve_checkpoint(cfg.teacher_model)
        teacher_checkpoint = torch.load(
            teacher_path, map_location="cpu", weights_only=False
        )
        student_confidence = _confidence_state(teacher_checkpoint)
        if not student_confidence:
            raise ValueError(f"No confidence weights in {teacher_path}")
        module.confidence.load_state_dict(student_confidence, strict=True)
        log.info("Initialized confidence from %s", teacher_path)

    return module, student_checkpoint


def parse_args() -> DistillConfig:
    cfg = DistillConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["confidence", "pitch"], default=None)
    parser.add_argument("--student-checkpoint", default=None)
    parser.add_argument("--teacher-labels-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--precision", default=None)
    args = parser.parse_args()

    if args.mode is not None:
        cfg.mode = args.mode
    if args.student_checkpoint is not None:
        cfg.student_checkpoint = args.student_checkpoint
    if args.teacher_labels_dir is not None:
        cfg.teacher_labels_dir = str(args.teacher_labels_dir)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.max_minutes is not None:
        cfg.max_minutes = args.max_minutes
    if args.accelerator is not None:
        cfg.accelerator = args.accelerator
    if args.precision is not None:
        cfg.precision = args.precision
    cfg.__post_init__()
    return cfg


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = parse_args()
    pl.seed_everything(0, workers=True)

    if not cfg.run_name:
        cfg.run_name = f"{datetime.now():%Y%m%d_%H%M%S}_{cfg.mode}"
    run_dir = Path(cfg.output_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2))
    log.info("Run directory: %s", run_dir)

    datamodule = PESTODistillationDataModule(
        wav_paths=cfg.wav_paths,
        validation_wav=cfg.validation_wav,
        teacher_labels_dir=cfg.teacher_labels_dir,
        validation_start_fraction=cfg.validation_start_fraction,
        sample_rate=cfg.sample_rate,
        chunk_size=cfg.chunk_size,
        precompute_batch=cfg.precompute_batch,
        validation_precompute_batch=cfg.validation_precompute_batch,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        mirror=cfg.mirror,
        mirror_fn=cfg.mirror_fn,
        model_name=cfg.teacher_model,
        max_minutes=cfg.max_minutes,
    )
    module, _ = build_distillation_module(cfg)

    last_checkpoint = ModelCheckpoint(
        dirpath=run_dir,
        filename="last",
        save_last=True,
        save_top_k=0,
        every_n_epochs=1,
    )
    best_checkpoint = ModelCheckpoint(
        dirpath=run_dir,
        filename=f"best-{cfg.mode}-epoch={{epoch:03d}}-val_loss={{val_loss:.6f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    logger = TensorBoardLogger(
        save_dir=str(run_dir.parent), name=cfg.run_name, version=""
    )
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        gradient_clip_val=cfg.grad_clip,
        log_every_n_steps=cfg.log_every_n_steps,
        callbacks=[
            last_checkpoint,
            best_checkpoint,
            LearningRateMonitor(logging_interval="epoch"),
        ],
        logger=logger,
        enable_progress_bar=True,
        default_root_dir=run_dir,
        num_sanity_val_steps=0,
    )

    log.info(
        "Starting %s distillation: epochs=%d lr=%.1e",
        cfg.mode,
        cfg.epochs,
        cfg.confidence_lr if cfg.mode == "confidence" else cfg.pitch_lr,
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=cfg.resume_from)

    final_checkpoint = run_dir / f"distilled-{cfg.mode}-{cfg.run_name}.ckpt"
    trainer.save_checkpoint(final_checkpoint)
    log.info("Saved final checkpoint: %s", final_checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


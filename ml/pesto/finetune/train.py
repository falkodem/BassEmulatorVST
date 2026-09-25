"""Entry point for fine-tuning PESTO on guitar audio in streaming mode.

Usage:
    poetry run python -m ml.pesto.finetune.train
    poetry run python -m ml.pesto.finetune.train --epochs 5 --lr 5e-6

What it does:
 1. Loads pretrained `mir-1k_g7` checkpoint to get encoder weights + hparams
 2. Builds GuitarStreamingDataModule that produces CQT frames mimicking
    plugin's runtime conditions (streaming CachedConv1d, mirror=1.0, refill)
 3. Fine-tunes with PESTO self-supervised losses (invariance + equivariance + SCE)
 4. Saves checkpoint as pesto-compatible .ckpt (loadable via `pesto.load_model`)
 5. To deploy in plugin: re-run `ml/pesto/export_onnx.py --model-name <ckpt>`
"""
import argparse
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# add project root to path so `ml.pesto.finetune.X` imports work
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ml.pesto.finetune.config import TrainConfig
from ml.pesto.finetune.streaming_datamodule import GuitarStreamingDataModule
from ml.pesto.finetune.vendor.networks.resnet1d import Resnet1d
from ml.pesto.finetune.vendor.pesto_module import PESTO, PitchShiftCQT
from ml.pesto.finetune.vendor.losses import CrossEntropyLoss, ShiftCrossEntropy, PowerSeries
from ml.pesto.finetune.vendor.loss_weighting import GradientsLossWeighting, LossWeighting
from ml.pesto.finetune.vendor.pesto_module import nn  # for transforms list

# Reuse pesto-full's ToLogMagnitude / Augmentations. They're tiny; inline them.
import torch.nn as _nn


class ToLogMagnitude(_nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, x):
        if x.size(-1) == 2:
            x = torch.view_as_complex(x)
        if x.ndim == 2:
            x.unsqueeze_(1)
        x = x.abs()
        x.clamp_(min=self.eps).log10_().mul_(20)
        return x


class BatchRandomNoise(_nn.Module):
    # Defaults from pesto-full configs/model/default.yaml (NOT from transforms.py module defaults)
    def __init__(self, min_snr: float = 0.1, max_snr: float = 2.0, p: float = 0.7):
        super().__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.p = p

    def forward(self, x):
        bs = x.size(0)
        device = x.device
        snr = torch.empty(bs, device=device).uniform_(self.min_snr, self.max_snr)
        mask = torch.rand_like(snr).le(self.p)
        snr[mask] = 0
        noise_std = snr * x.view(bs, -1).std(dim=-1)
        noise_std = noise_std.unsqueeze(-1).expand_as(x.view(bs, -1)).view_as(x)
        return x + noise_std * torch.randn_like(x)


class BatchRandomGain(_nn.Module):
    def __init__(self, min_gain: float = 0.5, max_gain: float = 1.5, p: float = 0.7):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.p = p

    def forward(self, x):
        bs = x.size(0)
        device = x.device
        vol = torch.empty(bs, device=device).uniform_(self.min_gain, self.max_gain)
        mask = torch.rand_like(vol).le(self.p)
        vol[mask] = 1
        vol = vol.unsqueeze(-1).expand_as(x.view(bs, -1)).view_as(x)
        return vol * x


log = logging.getLogger(__name__)


def _global_grad_norm(module: torch.nn.Module) -> torch.Tensor | None:
    norms = [
        p.grad.detach().norm(2)
        for p in module.parameters()
        if p.grad is not None
    ]
    if not norms:
        return None
    return torch.norm(torch.stack(norms), 2)


class GradientNormLogger(pl.Callback):
    """Log gradient norms before and after Lightning gradient clipping."""

    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        norm = _global_grad_norm(pl_module)
        if norm is not None:
            pl_module.log("train/grad_norm_pre_clip", norm, prog_bar=False, logger=True)

    def on_before_zero_grad(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: optim.Optimizer,
    ) -> None:
        norm = _global_grad_norm(pl_module)
        if norm is not None:
            pl_module.log("train/grad_norm_post_clip", norm, prog_bar=False, logger=True)



def build_pesto_module(cfg: TrainConfig) -> tuple[PESTO, dict]:
    """Load mir-1k_g7 checkpoint, build PESTO module with its encoder, then load weights."""
    # find checkpoint path
    if Path(cfg.pretrained).exists():
        ckpt_path = cfg.pretrained
    else:
        # look up in pesto package weights dir
        import pesto
        ckpt_path = Path(pesto.__file__).parent / "weights" / f"{cfg.pretrained}.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    log.info("Loading pretrained: %s", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Hparams in the checkpoint are OmegaConf ListConfig/DictConfig — convert
    # to plain python types so kwargs unpacking and json-serialization work.
    from omegaconf import OmegaConf, DictConfig, ListConfig

    def _to_python(x):
        if isinstance(x, (DictConfig, ListConfig)):
            return OmegaConf.to_container(x, resolve=True)
        return x

    encoder_hparams = _to_python(checkpoint['hparams']['encoder'])
    pitch_shift_hparams = _to_python(checkpoint['hparams']['pitch_shift'])

    # build encoder with EXACT hparams from checkpoint (extra keys swallowed by **unused)
    encoder = Resnet1d(**encoder_hparams)

    # build pitch shift with EXACT hparams from checkpoint
    pitch_shift = PitchShiftCQT(
        min_steps=pitch_shift_hparams['min_steps'],
        max_steps=pitch_shift_hparams['max_steps'],
    )

    # losses (defaults from pesto-full default config)
    inv_loss = CrossEntropyLoss(symmetric=True, detach_targets=True)
    sce_loss = ShiftCrossEntropy(
        pad_length=pitch_shift_hparams['max_steps'],
        criterion=inv_loss,
    )
    equiv_loss = PowerSeries(
        value=2 ** (1/36),                     # 1.019440644 (cubic root of semitone @ bps=3)
        power_min=1 - encoder_hparams['output_dim'],
        power_max=1,
        tau=2 ** (1/6) - 1,                    # 0.122462048
    )

    # optimizer / scheduler partials (will be instantiated by configure_optimizers)
    def opt_partial(params):
        return optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    def sched_partial(optimizer):
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    module = PESTO(
        encoder=encoder,
        optimizer_cls=opt_partial,
        scheduler_cls=sched_partial,
        equiv_loss_fn=equiv_loss,
        sce_loss_fn=sce_loss,
        inv_loss_fn=inv_loss,
        pitch_shift=pitch_shift,
        transforms=[BatchRandomNoise(), BatchRandomGain()],
        reduction=checkpoint['hparams'].get('reduction', 'alwa'),
    )

    # load weights (strict=False to skip non-trained items like preprocessor's CQT kernels)
    missing, unexpected = module.load_state_dict(checkpoint['state_dict'], strict=False)
    log.info("Loaded state_dict: missing=%d, unexpected=%d", len(missing), len(unexpected))
    if missing:
        log.info("  Missing keys (first 5): %s", missing[:5])
    if unexpected:
        log.info("  Unexpected keys (first 5): %s", unexpected[:5])

    return module, checkpoint


def parse_args() -> TrainConfig:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wav', nargs='+', default=None, help='Override wav paths')
    parser.add_argument('--epochs', type=int, default=cfg.epochs)
    parser.add_argument('--lr', type=float, default=cfg.lr)
    parser.add_argument('--batch-size', type=int, default=cfg.batch_size)
    parser.add_argument('--precompute-batch', type=int, default=cfg.precompute_batch)
    parser.add_argument('--mirror', type=float, default=cfg.mirror)
    parser.add_argument('--mirror-fn', choices=['zeros', 'refill'], default=cfg.mirror_fn)
    parser.add_argument('--num-workers', type=int, default=cfg.num_workers)
    parser.add_argument('--pretrained', default=cfg.pretrained)
    parser.add_argument('--resume-from', default=cfg.resume_from,
                        help='Resume full Lightning training state from checkpoint (e.g. runs/.../last.ckpt)')
    parser.add_argument('--output-dir', default=cfg.output_dir)
    parser.add_argument('--run-name', default=cfg.run_name)
    parser.add_argument('--accelerator', default=cfg.accelerator)
    parser.add_argument('--precision', default=cfg.precision)
    parser.add_argument('--max-minutes', type=float, default=None,
                        help='If set, truncate each wav to this many minutes (for quick tests)')
    args = parser.parse_args()

    if args.wav is not None:
        cfg.wav_paths = args.wav
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    cfg.batch_size = args.batch_size
    cfg.precompute_batch = args.precompute_batch
    cfg.mirror = args.mirror
    cfg.mirror_fn = args.mirror_fn
    cfg.num_workers = args.num_workers
    cfg.pretrained = args.pretrained
    cfg.resume_from = args.resume_from
    cfg.output_dir = args.output_dir
    cfg.run_name = args.run_name
    cfg.accelerator = args.accelerator
    cfg.precision = args.precision
    cfg.max_minutes = args.max_minutes  # type: ignore[attr-defined]
    cfg.__post_init__()
    return cfg


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = parse_args()

    # run dir
    if not cfg.run_name:
        cfg.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Run dir: %s", run_dir)

    # save config snapshot
    import json
    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump({k: (str(v) if isinstance(v, (Path,)) else v) for k, v in asdict(cfg).items()}, f, indent=2)
    log.info("Saved config snapshot: %s", config_path)

    # data
    datamodule = GuitarStreamingDataModule(
        wav_paths=cfg.wav_paths,
        sample_rate=cfg.sample_rate,
        chunk_size=cfg.chunk_size,
        precompute_batch=cfg.precompute_batch,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        random_offset=cfg.random_offset,
        harmonics=cfg.harmonics,
        fmin=cfg.fmin,
        bins_per_semitone=cfg.bins_per_semitone,
        n_bins=cfg.n_bins,
        center_bins=cfg.center_bins,
        gamma=cfg.gamma,
        mirror=cfg.mirror,
        mirror_fn=cfg.mirror_fn,
        model_name=cfg.pretrained,
        transforms=[ToLogMagnitude()],
    )

    # model
    module, _ = build_pesto_module(cfg)

    # callbacks
    loss_weights = {
        "invariance": cfg.weight_invariance,
        "equivariance": cfg.weight_equivariance,
        "shift_entropy": cfg.weight_shift_entropy,
    }
    if cfg.loss_weighting == "gradients":
        loss_weighting = GradientsLossWeighting(weights=loss_weights, ema_rate=cfg.loss_weighting_ema)
    else:
        loss_weighting = LossWeighting(weights=loss_weights)
    last_ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="last",
        save_last=True,
        save_top_k=0,
        every_n_epochs=1,
    )
    best_ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best-epoch={epoch:03d}-train_loss={train_loss:.6f}",
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval='epoch')
    grad_norm_cb = GradientNormLogger()

    logger = TensorBoardLogger(save_dir=str(run_dir.parent), name=cfg.run_name, version="")

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        gradient_clip_val=cfg.grad_clip,
        log_every_n_steps=cfg.log_every_n_steps,
        callbacks=[loss_weighting, last_ckpt_cb, best_ckpt_cb, lr_cb, grad_norm_cb],
        logger=logger,
        reload_dataloaders_every_n_epochs=1,    # rebuilds HCQT each epoch (new offset)
        enable_progress_bar=True,
        default_root_dir=str(run_dir),
        num_sanity_val_steps=0,                 # skip — we have dummy val anyway
    )

    if cfg.resume_from:
        log.info("Resuming full training state from: %s", cfg.resume_from)
    log.info("Starting fit (epochs=%d, batch=%d, lr=%.1e)", cfg.epochs, cfg.batch_size, cfg.lr)
    trainer.fit(module, datamodule=datamodule, ckpt_path=cfg.resume_from)

    # save final state under name compatible with pesto.loader
    final_ckpt = run_dir / f"finetuned-{cfg.run_name}.ckpt"
    log.info("Saving final checkpoint to %s (loadable by pesto.load_model)", final_ckpt)
    trainer.save_checkpoint(str(final_ckpt))
    log.info("Done. Re-export ONNX with:")
    log.info("  poetry run python ml/pesto/export_onnx.py --model-name %s "
             "--mirror %.2f --mirror-fn %s", final_ckpt, cfg.mirror, cfg.mirror_fn)
    return 0


if __name__ == "__main__":
    sys.exit(main())

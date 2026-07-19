"""PESTO LightningModule + PitchShiftCQT.

Vendored from pesto-full src/models/pesto.py and src/data/pitch_shift.py.

Changes vs upstream:
 - removed `remove_omegaconf_dependencies` and `omegaconf` dependency:
   we use plain dicts/dataclasses, no Hydra
 - `on_save_checkpoint` stores plain hyperparams instead of omegaconf-cleaned ones
 - imports use relative paths
 - `pytorch_lightning` instead of upstream's `lightning`
"""
import logging
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .calibration import generate_synth_data
from .loss_weighting import LossWeighting
from .losses import NullLoss
from .reduce_activations import reduce_activations


log = logging.getLogger(__name__)


# ─── PitchShiftCQT ──────────────────────────────────────────────────────────


def _randint_sampling_fn(min_value, max_value):
    def sample_randint(*size, **kwargs):
        return torch.randint(min_value, max_value + 1, size, **kwargs)
    return sample_randint


def _gaussint_sampling_fn(min_value, max_value):
    mean = (min_value + max_value) / 2
    std = (max_value - mean) / 2
    def sample_gaussint(*size, **kwargs):
        return torch.randn(size, **kwargs).add_(mean).mul_(std).long().clip(min=min_value, max=max_value)
    return sample_gaussint


class PitchShiftCQT(nn.Module):
    """In-frequency-domain pitch shift via slicing CQT bins.

    Input CQT has shape (batch, channels, n_bins_total). We pick a central
    window of size `output_height = n_bins_total - max_steps + min_steps`
    starting at `lower_bin = max_steps`. The shifted view picks the same-width
    window starting at `lower_bin - n_steps` for a random `n_steps` per sample.

    Returns (x, xt, n_steps): original window, shifted window, shift amounts.
    """
    def __init__(self, min_steps: int, max_steps: int, gaussian_sampling: bool = False):
        super().__init__()
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.sample_random_steps = (_gaussint_sampling_fn(min_steps, max_steps)
                                    if gaussian_sampling
                                    else _randint_sampling_fn(min_steps, max_steps))
        self.lower_bin = self.max_steps

    def forward(self, spectrograms: torch.Tensor):
        batch_size, _, input_height = spectrograms.size()
        output_height = input_height - self.max_steps + self.min_steps
        assert output_height > 0, (
            f"Input height {input_height} too small for shift range "
            f"[{self.min_steps}, {self.max_steps}]"
        )
        n_steps = self.sample_random_steps(batch_size, device=spectrograms.device)
        x = spectrograms[..., self.lower_bin: self.lower_bin + output_height]
        xt = self._extract_bins(spectrograms, self.lower_bin - n_steps, output_height)
        return x, xt, n_steps

    @staticmethod
    def _extract_bins(inputs: torch.Tensor, first_bin: torch.LongTensor, output_height: int):
        indices = first_bin.unsqueeze(-1) + torch.arange(output_height, device=inputs.device)
        dims = inputs.size(0), 1, output_height
        output_size = list(inputs.size())[:-1] + [output_height]
        indices = indices.view(*dims).expand(output_size)
        return inputs.gather(-1, indices)


# ─── PESTO LightningModule ───────────────────────────────────────────────────


class PESTO(pl.LightningModule):
    """Self-supervised pitch encoder.

    Three losses on CQT-frame triplets (original, augmented, pitch-shifted):
     * invariance: original vs augmented should give same activation distribution
     * shift-cross-entropy: pitch-shifted distribution should be shifted version of original
     * equivariance: scalar projection of activations should follow exact freq ratio
    """
    def __init__(self,
                 encoder: nn.Module,
                 optimizer_cls,
                 scheduler_cls=None,
                 equiv_loss_fn: nn.Module | None = None,
                 sce_loss_fn: nn.Module | None = None,
                 inv_loss_fn: nn.Module | None = None,
                 pitch_shift: PitchShiftCQT | None = None,
                 transforms: Sequence[nn.Module] | None = None,
                 reduction: str = "alwa"):
        super().__init__()
        self.encoder = encoder
        self.optimizer_cls = optimizer_cls
        self.scheduler_cls = scheduler_cls

        self.equiv_loss_fn = equiv_loss_fn or NullLoss()
        self.sce_loss_fn = sce_loss_fn or NullLoss()
        self.inv_loss_fn = inv_loss_fn or NullLoss()

        self.pitch_shift = pitch_shift or PitchShiftCQT(min_steps=0, max_steps=0)
        self.transforms = nn.Sequential(*transforms) if transforms is not None else nn.Identity()
        self.reduction = reduction

        self.loss_weighting: LossWeighting | None = None

        self.predictions = None
        self.labels = None

        # constant offset between network output and absolute MIDI semitones
        self.register_buffer('shift', torch.zeros((), dtype=torch.float), persistent=True)

        # Save for checkpoint hparams export (used by inference repo)
        self.hyperparams = dict(
            encoder=encoder.hparams,
            pitch_shift=dict(min_steps=self.pitch_shift.min_steps,
                             max_steps=self.pitch_shift.max_steps),
            reduction=reduction,
        )

    def forward(self,
                x: torch.Tensor,
                shift: bool = True,
                return_activations: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, *_ = self.pitch_shift(x)
        activations = self.encoder(x)
        preds = reduce_activations(activations, reduction=self.reduction)
        if shift:
            preds.sub_(self.shift)
        if return_activations:
            return preds, activations
        return preds

    def on_fit_start(self) -> None:
        for callback in self.trainer.callbacks:
            if isinstance(callback, LossWeighting):
                self.loss_weighting = callback
        if self.loss_weighting is None:
            self.loss_weighting = LossWeighting()
        self.loss_weighting.last_layer = self.encoder.fc.weight

    def on_validation_epoch_start(self) -> None:
        self.predictions = []
        self.labels = []
        self.estimate_shift()

    def on_validation_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        preds, labels = outputs
        self.predictions.append(preds)
        self.labels.append(labels)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, _ = batch  # labels not used in self-supervised training

        x, xt, n_steps = self.pitch_shift(x)
        xa = x.clone()

        xa = self.transforms(xa)
        xt = self.transforms(xt)

        y = self.encoder(x)
        ya = self.encoder(xa)
        yt = self.encoder(xt)

        inv_loss = self.inv_loss_fn(y, ya)
        shift_entropy_loss = self.sce_loss_fn(ya, yt, n_steps)
        equiv_loss = self.equiv_loss_fn(ya, yt, n_steps)

        total_loss = self.loss_weighting.combine_losses(
            invariance=inv_loss,
            shift_entropy=shift_entropy_loss,
            equivariance=equiv_loss,
        )

        loss_dict = dict(invariance=inv_loss,
                         equivariance=equiv_loss,
                         shift_entropy=shift_entropy_loss,
                         loss=total_loss)
        self.log_dict({f"loss/{k}/train": v for k, v in loss_dict.items()}, sync_dist=False)
        self.log("train_loss", total_loss,
                 on_step=False, on_epoch=True, prog_bar=True, logger=False, sync_dist=False)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, pitch = batch
        return self.forward(x), pitch

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Store hparams + hcqt_params on checkpoint for inference-repo loader.

        The inference repo (`pesto.load_model`) expects these top-level keys.
        `hcqt_params` is provided by the datamodule attribute `hcqt_kwargs`.
        """
        checkpoint["hparams"] = self.hyperparams
        dm = getattr(self.trainer, 'datamodule', None)
        hcqt_kwargs = getattr(dm, 'hcqt_kwargs', None) if dm is not None else None
        if hcqt_kwargs is not None:
            checkpoint['hcqt_params'] = dict(hcqt_kwargs)

    def configure_optimizers(self) -> Mapping[str, Any]:
        optimizer = self.optimizer_cls(params=self.encoder.parameters())
        out = dict(optimizer=optimizer)
        if self.scheduler_cls is not None:
            out["lr_scheduler"] = self.scheduler_cls(optimizer=optimizer)
        return out

    def estimate_shift(self) -> None:
        """Calibrate self.shift so that predictions are absolute MIDI semitones."""
        labels = torch.arange(60, 72)

        sr = 16000
        dm = self.trainer.datamodule
        batch = []
        for p in labels:
            audio = generate_synth_data(p, sr=sr)
            hcqt = dm.hcqt(audio, sr)
            batch.append(hcqt[0])

        x = torch.stack(batch, dim=0).to(self.device)
        x = dm.transforms(torch.view_as_complex(x))

        preds = self.forward(x, shift=False)

        diff = preds - labels.to(self.device)
        shift, std = diff.median(), diff.std()
        log.info(f"Estimated shift: {shift.cpu().item():.3f} (std={std.cpu().item():.3f})")
        self.shift.fill_(shift)

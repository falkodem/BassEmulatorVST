"""Lightning module for separate confidence and pitch distillation runs."""
import math
from typing import Any, Mapping

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pesto.model import ConfidenceClassifier

from ml.finetune_pesto.vendor.reduce_activations import reduce_activations


class PESTODistillationModule(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        confidence: ConfidenceClassifier,
        *,
        mode: str,
        min_pitch_shift_steps: int,
        max_pitch_shift_steps: int,
        bins_per_semitone: int,
        reduction: str,
        lr: float,
        weight_decay: float,
        scheduler_epochs: int,
        teacher_confidence_power: float = 1.0,
    ):
        super().__init__()
        if mode not in {"confidence", "pitch"}:
            raise ValueError(f"Unsupported distillation mode: {mode}")
        self.encoder = encoder
        self.confidence = confidence
        self.mode = mode
        self.min_pitch_shift_steps = min_pitch_shift_steps
        self.max_pitch_shift_steps = max_pitch_shift_steps
        self.bins_per_semitone = bins_per_semitone
        self.reduction = reduction
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_epochs = scheduler_epochs
        self.teacher_confidence_power = teacher_confidence_power
        self.register_buffer("shift", torch.zeros((), dtype=torch.float), persistent=True)

        self.hyperparams = {
            "encoder": encoder.hparams,
            "pitch_shift": {
                "min_steps": min_pitch_shift_steps,
                "max_steps": max_pitch_shift_steps,
            },
            "reduction": reduction,
        }

        train_pitch = mode == "pitch"
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(train_pitch)
        for parameter in self.confidence.parameters():
            parameter.requires_grad_(not train_pitch)

    def _crop_pitch_input(self, x: torch.Tensor) -> torch.Tensor:
        stop = x.size(-1) + self.min_pitch_shift_steps
        return x[..., self.max_pitch_shift_steps:stop]

    def _absolute_activations(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.encoder(self._crop_pitch_input(x))
        shift_bins = -round(float(self.shift.detach().cpu()) * self.bins_per_semitone)
        return torch.roll(raw, shifts=shift_bins, dims=-1)

    def _confidence_prediction(self, x: torch.Tensor) -> torch.Tensor:
        energy = torch.exp(x * (math.log(10.0) / 10.0)).squeeze(1)
        return self.confidence(energy)

    @staticmethod
    def _masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (values * weights).sum() / weights.sum().clamp_min(1e-7)

    def _confidence_loss(
        self,
        x: torch.Tensor,
        teacher_confidence: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = self._confidence_prediction(x)
        target = teacher_confidence.to(dtype=prediction.dtype)
        per_frame = F.binary_cross_entropy(prediction, target, reduction="none")
        weights = valid.to(dtype=prediction.dtype)
        loss = self._masked_mean(per_frame, weights)
        mae = self._masked_mean((prediction - target).abs(), weights)
        return loss, {
            "confidence_mae": mae,
            "student_confidence": self._masked_mean(prediction, weights),
            "teacher_confidence": self._masked_mean(target, weights),
        }

    def _pitch_loss(
        self,
        x: torch.Tensor,
        teacher_activations: torch.Tensor,
        teacher_confidence: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        student = self._absolute_activations(x).clamp_min(1e-8)
        teacher = teacher_activations.to(dtype=student.dtype).clamp_min(1e-8)
        teacher = teacher / teacher.sum(dim=-1, keepdim=True)
        per_frame = torch.sum(teacher * (teacher.log() - student.log()), dim=-1)
        confidence_weight = teacher_confidence.to(student.dtype).clamp(0.0, 1.0)
        confidence_weight = confidence_weight.pow(self.teacher_confidence_power)
        weights = confidence_weight * valid.to(student.dtype)
        loss = self._masked_mean(per_frame, weights)
        return loss, {
            "unweighted_kl": self._masked_mean(per_frame, valid.to(student.dtype)),
            "teacher_weight": self._masked_mean(
                confidence_weight, valid.to(student.dtype)
            ),
        }

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        x, teacher_activations, teacher_confidence, valid = batch
        if self.mode == "confidence":
            loss, metrics = self._confidence_loss(x, teacher_confidence, valid)
        else:
            loss, metrics = self._pitch_loss(
                x, teacher_activations, teacher_confidence, valid
            )

        batch_size = x.size(0)
        self.log(
            f"{stage}_loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log_dict(
            {f"{stage}/{name}": value for name, value in metrics.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def forward(self, x: torch.Tensor):
        activations = self._absolute_activations(x)
        pitch = reduce_activations(activations, reduction=self.reduction)
        confidence = self._confidence_prediction(x)
        return pitch, confidence, activations

    def configure_optimizers(self) -> Mapping[str, Any]:
        parameters = (
            self.confidence.parameters()
            if self.mode == "confidence"
            else self.encoder.parameters()
        )
        optimizer = torch.optim.Adam(
            parameters, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.scheduler_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["hparams"] = self.hyperparams
        datamodule = getattr(self.trainer, "datamodule", None)
        hcqt_kwargs = getattr(datamodule, "hcqt_kwargs", None)
        if hcqt_kwargs is not None:
            checkpoint["hcqt_params"] = dict(hcqt_kwargs)
        checkpoint["distillation"] = {
            "mode": self.mode,
            "teacher_confidence_power": self.teacher_confidence_power,
        }


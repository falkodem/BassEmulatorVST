"""Loss functions for PESTO self-supervised training.

Vendored from pesto-full src/losses/{base,entropy,equivariance}.py — merged
into one file. Removed `ComposeLoss` (unused, has print statements anyway).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NullLoss(nn.Module):
    """No-op loss used when a particular term is disabled."""
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return args[0].mean().mul(0)


# ─── invariance / shift-cross-entropy ────────────────────────────────────────


class CrossEntropyLoss(nn.Module):
    """Wraps nn.CrossEntropyLoss with optional symmetric mode + detached targets.

    Used as the invariance loss (compare original vs augmented view of same audio)
    and as the criterion inside ShiftCrossEntropy.
    """
    def __init__(self,
                 symmetric: bool = False,
                 detach_targets: bool = False,
                 backend: nn.Module | None = None):
        super().__init__()
        self.symmetric = symmetric
        self.detach_targets = detach_targets
        self.backend = backend if backend is not None else nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.symmetric:
            return (self.compute_loss(input, target) + self.compute_loss(target, input)) / 2
        return self.compute_loss(input, target)

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.backend(input, target.detach() if self.detach_targets else target)


class ShiftCrossEntropy(nn.Module):
    """CrossEntropy where the target is shifted by `target` semitones.

    Encourages the activation distribution of pitch-shifted input to be exactly
    the same as the original distribution, just shifted along the bin axis.
    """
    def __init__(self, pad_length: int = 5, criterion: nn.Module | None = None):
        super().__init__()
        self.pad_length = pad_length
        self.criterion = criterion if criterion is not None else CrossEntropyLoss()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x1 = F.pad(x1, (self.pad_length, self.pad_length))
        x2 = F.pad(x2, (2 * self.pad_length, 2 * self.pad_length))
        idx = target.unsqueeze(1) + torch.arange(x1.size(-1), device=target.device) + self.pad_length
        shift_x2 = torch.gather(x2, dim=1, index=idx)
        return self.criterion(x1, shift_x2)


# ─── equivariance ────────────────────────────────────────────────────────────


class HuberLoss(nn.Module):
    def __init__(self, tau: float):
        super().__init__()
        self.register_buffer("tau", torch.tensor(tau), persistent=False)

    def forward(self, x):
        x = x.abs()
        return torch.where(x.le(self.tau),
                           x ** 2 / 2,
                           self.tau ** 2 / 2 + self.tau * (x - self.tau))


class PowerSeries(nn.Module):
    """Equivariance loss: project activations to scalar via power-series weights,
    then enforce that the ratio of projections matches the pitch-shift factor.
    """
    def __init__(self, value: float, power_min: int, power_max: int, tau: float = 1.):
        super().__init__()
        self.value = value
        powers = torch.arange(power_min, power_max)
        self.register_buffer("weights", self.value ** powers, persistent=False)
        self.dim = len(self.weights)
        self.loss_fn = HuberLoss(tau)

    def forward(self,
                x1: torch.Tensor,
                x2: torch.Tensor,
                target: torch.Tensor,
                nlog_c1: torch.Tensor | None = None,
                nlog_c2: torch.Tensor | None = None) -> torch.Tensor:
        z1 = self.project(x1)
        z2 = self.project(x2)
        if nlog_c1 is not None:
            z1 = z1 * torch.exp(-nlog_c1)
        if nlog_c2 is not None:
            z2 = z2 * torch.exp(-nlog_c2)
        freq_ratios = self.value ** target.float()
        loss_12 = self.loss_fn(z2 / z1 - freq_ratios).mean()
        loss_21 = self.loss_fn(z1 / z2 - 1 / freq_ratios).mean()
        return (loss_12 + loss_21) / 2

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return x.mv(self.weights)

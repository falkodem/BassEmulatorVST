import torch
import torch.nn as nn


class MultiScaleSTFTLoss(nn.Module):
    """
    L1 loss on magnitude spectrograms at multiple STFT scales.

    Operates on waveform input (batch, 1, time). More sensitive to tonal
    accuracy than sample-level L1/L2: a phase-shifted sine has zero sample
    loss but non-zero spectral loss.

    Default scales suit window_size=1024 (minimum n_fft must fit the signal).
    """

    def __init__(self, fft_sizes: tuple[int, ...] = (128, 256, 512)):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = [s // 4 for s in fft_sizes]
        for n in fft_sizes:
            self.register_buffer(f"_win{n}", torch.hann_window(n))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Flatten channel dim: (B, 1, T) → (B, T)
        if pred.dim() == 3:
            pred   = pred.squeeze(1)
            target = target.squeeze(1)

        total = pred.new_zeros(())
        for n_fft, hop in zip(self.fft_sizes, self.hop_sizes):
            win = getattr(self, f"_win{n_fft}")
            p_mag = torch.stft(pred,   n_fft=n_fft, hop_length=hop,
                               window=win, center=False, return_complex=True).abs()
            t_mag = torch.stft(target, n_fft=n_fft, hop_length=hop,
                               window=win, center=False, return_complex=True).abs()
            total = total + torch.mean(torch.abs(p_mag - t_mag))

        return total / len(self.fft_sizes)


LOSS_REGISTRY: dict[str, type] = {
    "l1":               nn.L1Loss,
    "l2":               nn.MSELoss,
    "multi_scale_stft": MultiScaleSTFTLoss,
}


def make_loss(name: str) -> nn.Module:
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name]()

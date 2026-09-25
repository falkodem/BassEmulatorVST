"""
Input/output transforms for WindowDataset.

Waveform tensors entering the dataset are shaped (1, window_size).
Transforms convert them to the representation expected by a given architecture.

  domain="waveform"  →  (1, window_size)           — no transform, identity
  domain="stft"      →  (1, freq_bins, time_frames) — magnitude spectrogram
                         freq_bins  = n_fft // 2 + 1
                         time_frames = (window_size - n_fft) // hop + 1

Note: stft transform is applied to both guitar and bass tensors inside
WindowDataset, so the model maps spectrogram → spectrogram.
Use "l1" or "l2" loss when training in the stft domain; "multi_scale_stft"
is meant for waveform-domain models only.
"""

import torch


class STFTTransform:
    """(1, T) waveform → (1, F, t) magnitude spectrogram."""

    def __init__(self, n_fft: int, hop_length: int):
        self.n_fft      = n_fft
        self.hop_length = hop_length

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, T)  →  spec: (1, F, t)
        signal = x.squeeze(0)                                      # (T,)
        window = torch.hann_window(self.n_fft, device=signal.device)
        spec   = torch.stft(
            signal,
            n_fft      = self.n_fft,
            hop_length = self.hop_length,
            window     = window,
            center     = False,
            return_complex = True,
        )
        return spec.abs().unsqueeze(0)                             # (1, F, t)

    def output_shape(self, window_size: int) -> tuple[int, int, int]:
        freq_bins   = self.n_fft // 2 + 1
        time_frames = (window_size - self.n_fft) // self.hop_length + 1
        return (1, freq_bins, time_frames)


def make_transform(domain: str, stft_n_fft: int, stft_hop_length: int):
    """
    Returns the transform callable for the given domain, or None for waveform.
    """
    if domain == "waveform":
        return None
    if domain == "stft":
        return STFTTransform(n_fft=stft_n_fft, hop_length=stft_hop_length)
    raise ValueError(f"Unknown domain '{domain}'. Available: waveform, stft")

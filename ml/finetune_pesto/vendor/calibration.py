"""Vendored from pesto-full src/utils/calibration.py — unchanged.

Used by PESTO.estimate_shift to find the absolute-pitch offset between the
network's relative output and MIDI semitones.
"""
import torch


def mid_to_hz(pitch: int) -> float:
    return 440 * 2 ** ((pitch - 69) / 12)


def generate_synth_data(pitch: int, num_harmonics: int = 5, duration: float = 2, sr: int = 16000) -> torch.Tensor:
    f0 = mid_to_hz(pitch)
    t = torch.arange(0, duration, 1 / sr)
    harmonics = torch.stack([
        torch.cos(2 * torch.pi * k * f0 * t + torch.rand(()))
        for k in range(1, num_harmonics + 1)
    ], dim=1)
    volume = torch.rand(num_harmonics)
    volume[0] = 1
    volume *= torch.randn(())
    audio = torch.sum(volume * harmonics, dim=1)
    return audio

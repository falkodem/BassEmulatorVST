import torch
import torch.nn as nn


class WaveConvNet(nn.Module):
    """
    Stateless 1D CNN operating in the waveform domain.
    Input/output: (batch, 1, 1024) — raw audio samples.

    No recurrence, no cache — each window is processed independently.
    All kernels are odd → symmetric padding → output length == input length.
    ~85k parameters.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,  16, kernel_size=63, padding=31), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=31, padding=15), nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=15, padding=7),  nn.ReLU(),
            nn.Conv1d(16,  1, kernel_size=7,  padding=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

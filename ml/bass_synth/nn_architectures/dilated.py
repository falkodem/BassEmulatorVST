import torch
import torch.nn as nn


class DilatedConvNet(nn.Module):
    """
    Fully-dilated 1D CNN with receptive field = 1023 samples (~23 ms at 44100 Hz).
    Input/output: (batch, 1, 1024) — raw audio samples.

    Dilation schedule doubles each layer: 1, 2, 4, …, 256
    RF = 2 * (1 + 2 + 4 + … + 256) + 1 = 1023 — covers the full window with
    far fewer parameters than large-kernel convolutions.

    Residual skip (input → output) helps the model learn a correction
    on top of the guitar signal rather than the full bass waveform.

    ~28k parameters (channels=32).
    """

    DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256)

    def __init__(self, channels: int = 32):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv1d(1, channels, kernel_size=1)]
        for d in self.DILATIONS:
            layers += [
                nn.Conv1d(channels, channels, kernel_size=3, dilation=d, padding=d),
                nn.ReLU(),
            ]
        layers.append(nn.Conv1d(channels, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

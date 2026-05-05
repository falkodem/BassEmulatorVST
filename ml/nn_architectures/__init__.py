from .bassnet import WaveConvNet
from .dilated import DilatedConvNet

REGISTRY: dict[str, type] = {
    "WaveConvNet":    WaveConvNet,
    "DilatedConvNet": DilatedConvNet,
}

__all__ = ["WaveConvNet", "DilatedConvNet", "REGISTRY"]

from .yin import YinDetector
from .pesto import PestoDetector

REGISTRY: dict[str, type] = {
    "yin": YinDetector,
    "pesto": PestoDetector,
}

__all__ = ["YinDetector", "PestoDetector", "REGISTRY"]

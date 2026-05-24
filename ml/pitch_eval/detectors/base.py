"""Base classes for pitch detectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class PitchResult:
    """Result of pitch detection for one audio file.

    Attributes:
        times:      shape (T,) — time in seconds for each frame
        f0_hz:      shape (T,) — fundamental frequency in Hz; 0.0 or NaN means unvoiced/silence
        confidence: shape (T,) — voicing confidence in [0, 1]
        name:       detector name (e.g. "yin", "pesto")
        hop_ms:     hop size in milliseconds used by this detector
    """
    times: np.ndarray
    f0_hz: np.ndarray
    confidence: np.ndarray
    name: str
    hop_ms: float


class PitchDetector(ABC):
    """Abstract base class for all pitch detectors."""

    name: str

    @abstractmethod
    def estimate(self, audio: np.ndarray, sr: int) -> PitchResult:
        """Estimate pitch from a mono audio signal.

        Args:
            audio: mono waveform, shape (N,), float32 or float64
            sr:    sample rate in Hz

        Returns:
            PitchResult with times, f0_hz, confidence, name, hop_ms
        """
        ...

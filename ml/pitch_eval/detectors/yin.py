"""YIN pitch detector wrapper using librosa.yin.

Parameters mirror the C++ YinPitchDetector in src/YinPitchDetector.h:
  - fmin=70, fmax=1300 Hz
  - frame_length=1536 (matches 1024 + 512 ring-buffer overlap in C++)
  - hop_length=1024

Confidence heuristic: a frame is "voiced" when the raw YIN estimate falls
strictly inside [fmin, fmax]. librosa.yin always returns a value in that
range (clamped), so we use a small guard margin (+5 Hz on fmin, -5 Hz on
fmax) to exclude frames where YIN hit the boundary — those are typically
noise/silence frames that got clipped to the nearest valid frequency.

This is deliberately simple: confidence is binary (0 or 1) because this
tool is for comparison, not production voicing detection.
"""

import numpy as np
import librosa

from .base import PitchDetector, PitchResult

_FMIN = 70.0
_FMAX = 1300.0
_FRAME_LENGTH = 1536
_HOP_LENGTH = 1024
_VOICED_MARGIN = 5.0   # Hz margin inward from fmin/fmax to exclude clipped frames


class YinDetector(PitchDetector):
    name = "yin"

    def __init__(
        self,
        fmin: float = _FMIN,
        fmax: float = _FMAX,
        frame_length: int = _FRAME_LENGTH,
        hop_length: int = _HOP_LENGTH,
        voiced_margin: float = _VOICED_MARGIN,
    ) -> None:
        self.fmin = fmin
        self.fmax = fmax
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.voiced_margin = voiced_margin

    def estimate(self, audio: np.ndarray, sr: int) -> PitchResult:
        hop_ms = self.hop_length / sr * 1000.0

        f0_yin = librosa.yin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=sr,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )

        # Binary voiced flag: f0 strictly inside [fmin+margin, fmax-margin]
        lo = self.fmin + self.voiced_margin
        hi = self.fmax - self.voiced_margin
        voiced = (f0_yin > lo) & (f0_yin < hi)
        confidence = voiced.astype(np.float64)

        f0_out = np.where(voiced, f0_yin, 0.0)
        times = np.arange(len(f0_yin)) * self.hop_length / sr

        return PitchResult(
            times=times.astype(np.float64),
            f0_hz=f0_out.astype(np.float64),
            confidence=confidence,
            name=self.name,
            hop_ms=hop_ms,
        )

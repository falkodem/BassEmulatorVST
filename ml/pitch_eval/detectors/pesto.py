"""PESTO pitch detector wrapper.

API (pesto-pitch 2.0.1):
    pesto.predict(x: torch.Tensor, sr: int, step_size: float, ...) ->
        (timesteps: Tensor, preds_hz: Tensor, confidence: Tensor, activations: Tensor)

    - x: mono 1-D or 2-D tensor (batch, samples)
    - timesteps: shape (T,), in *milliseconds* (divide by 1000 to get seconds)
    - preds_hz:  shape (T,), Hz when convert_to_freq=True
    - confidence: shape (T,), model's pitch-confidence [0, 1] (NOT voicing)

The model is loaded once on first call and cached as a class-level attribute
so multiple files in the same run share one model instance.
"""

import numpy as np
import pesto
import torch

from .base import PitchDetector, PitchResult

_STEP_SIZE_MS = 10.0       # native hop for PESTO
_MODEL_NAME = "mir-1k_g7"  # default pretrained model shipped with pesto-pitch


class PestoDetector(PitchDetector):
    name = "pesto"

    def __init__(
        self,
        step_size_ms: float = _STEP_SIZE_MS,
        model_name: str = _MODEL_NAME,
        voiced_threshold: float = 0.5,
    ) -> None:
        self.step_size_ms = step_size_ms
        self.model_name = model_name
        self.voiced_threshold = voiced_threshold

    def estimate(self, audio: np.ndarray, sr: int) -> PitchResult:
        x = torch.from_numpy(audio.astype(np.float32))

        # pesto.predict returns 4 tensors; all on CPU by default
        timesteps, preds_hz, confidence, _activations = pesto.predict(
            x,
            sr=sr,
            step_size=self.step_size_ms,
            model_name=self.model_name,
            convert_to_freq=True,
            no_grad=True,
            inference_mode=True,
        )

        times = (timesteps.cpu().numpy() / 1000.0).astype(np.float64)
        f0_hz = preds_hz.cpu().numpy().astype(np.float64)
        conf = confidence.cpu().numpy().astype(np.float64)

        # Zero out f0 for unvoiced frames
        f0_out = np.where(conf >= self.voiced_threshold, f0_hz, 0.0)

        return PitchResult(
            times=times,
            f0_hz=f0_out,
            confidence=conf,
            name=self.name,
            hop_ms=self.step_size_ms,
        )

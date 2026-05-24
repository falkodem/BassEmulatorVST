"""Export pretrained PESTO to ONNX for real-time pitch detection in the JUCE plugin.

`pesto-pitch` 2.0.1 ships no ONNX export — its `utils/export.py` only writes
CSV/NPZ/PNG *result* files.  This script builds the ONNX graph manually:

    audio waveform (mono, fixed sample rate)  ->  PESTO  ->  (f0_hz, confidence)

The whole PESTO pipeline is pure-torch and traceable:
  HarmonicCQT (nn.Conv1d kernels)  ->  Resnet1d encoder  ->  reduce_activations,
plus a ConfidenceClassifier branch.  The only non-ONNX op is `torch.view_as_complex`
in `Preprocessor.forward`; we monkeypatch it with an equivalent magnitude
computation sqrt(re^2 + im^2) that needs no complex dtype.

The CQT kernels are baked for a single sample rate (default 44100, the project
standard), so the exported model MUST be fed audio at exactly that rate.

The sample axis is exported as a dynamic axis: the same model accepts a whole
file (for verification here) or a fixed analysis window (for the plugin).

Outputs (into models/, which is gitignored):
  models/pesto.onnx            — the ONNX graph
  models/pesto_onnx_meta.json  — I/O spec for the C++ side (Block 3)

Usage
-----
    poetry run python ml/utils/export_pesto_onnx.py
    poetry run python ml/utils/export_pesto_onnx.py --sample-rate 44100 --opset 17
    poetry run python ml/utils/export_pesto_onnx.py --verify-audio data/v0/guitar/E2_260427_1446.wav
"""
import argparse
import json
import logging
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pesto.loader import load_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("export_pesto_onnx")

ROOT = Path(__file__).resolve().parents[2]


def _preprocessor_forward_no_complex(self, x, sr=None):
    """Drop-in replacement for pesto `Preprocessor.forward` without `torch.view_as_complex`.

    Original: `view_as_complex(hcqt).permute(0, 3, 1, 2)` then `to_log` (complex
    `.abs()`).  Here we compute the magnitude sqrt(re^2 + im^2) directly from the
    (..., 2) real/imag layout — numerically identical, but ONNX can trace it.
    """
    hcqt = self.hcqt(x, sr=sr)                                  # (batch, harmonics, freqs, time, 2)
    mag = torch.sqrt(hcqt[..., 0] ** 2 + hcqt[..., 1] ** 2)     # (batch, harmonics, freqs, time)
    mag = mag.permute(0, 3, 1, 2)                               # (batch, time, harmonics, freqs)
    return self.to_log(mag)                                     # log-magnitude, matches original


class PestoOnnxWrapper(nn.Module):
    """Wraps PESTO so `forward(audio)` -> (f0_hz, confidence), both shape (num_frames,).

    Input is a 1-D mono waveform at the baked-in sample rate; `sr=None` makes the
    preprocessor reuse its pre-built CQT kernels instead of rebuilding them.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, audio: torch.Tensor):
        preds, confidence, _vol = self.model(
            audio, sr=None, convert_to_freq=True, return_activations=False
        )
        return preds, confidence


def build_model(model_name: str, step_size: float, sample_rate: int) -> nn.Module:
    log.info("Loading PESTO '%s' (step_size=%.1f ms, sr=%d Hz)", model_name, step_size, sample_rate)
    model = load_model(model_name, step_size=step_size, sampling_rate=sample_rate)
    model.eval()
    # Bypass torch.view_as_complex (not ONNX-exportable) in the CQT preprocessor.
    model.preprocessor.forward = types.MethodType(_preprocessor_forward_no_complex, model.preprocessor)
    return model


def export_onnx(model: nn.Module, sample_rate: int, opset: int, output_path: Path) -> None:
    wrapper = PestoOnnxWrapper(model).eval()

    # 1 s of audio is plenty to produce many CQT frames during tracing.
    dummy = torch.randn(sample_rate, dtype=torch.float32)
    with torch.no_grad():
        f0, conf = wrapper(dummy)
    log.info("Wrapper sanity OK: %d samples -> %d frames (f0 %s, conf %s)",
             dummy.numel(), f0.numel(), tuple(f0.shape), tuple(conf.shape))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(output_path),
        input_names=["audio"],
        output_names=["f0_hz", "confidence"],
        dynamic_axes={
            "audio": {0: "num_samples"},
            "f0_hz": {0: "num_frames"},
            "confidence": {0: "num_frames"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    log.info("Exported ONNX -> %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)


def verify(model: nn.Module, onnx_path: Path, sample_rate: int, audio_path: Path) -> bool:
    """Compare ONNX output against torch reference on a real recording."""
    import librosa
    import onnxruntime as ort

    audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    audio = audio.astype(np.float32)
    log.info("Verifying on %s (%d samples, %.2f s)", audio_path.name, audio.size, audio.size / sample_rate)

    with torch.no_grad():
        ref_f0, ref_conf, _ = model(
            torch.from_numpy(audio), sr=None, convert_to_freq=True, return_activations=False
        )
    ref_f0 = ref_f0.numpy()
    ref_conf = ref_conf.numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_f0, onnx_conf = sess.run(None, {"audio": audio})

    if onnx_f0.shape != ref_f0.shape or onnx_conf.shape != ref_conf.shape:
        log.error("Shape mismatch: f0 onnx %s vs ref %s | conf onnx %s vs ref %s",
                  onnx_f0.shape, ref_f0.shape, onnx_conf.shape, ref_conf.shape)
        return False

    f0_abs = np.max(np.abs(onnx_f0 - ref_f0))
    # cents error only where both are voiced-ish frequencies
    mask = (ref_f0 > 1.0) & (onnx_f0 > 1.0)
    cents = 1200.0 * np.abs(np.log2(onnx_f0[mask] / ref_f0[mask])) if mask.any() else np.array([0.0])
    conf_abs = np.max(np.abs(onnx_conf - ref_conf))

    log.info("f0:   max abs %.4e Hz | max %.4f cents | mean %.4f cents",
             f0_abs, cents.max(), cents.mean())
    log.info("conf: max abs %.4e", conf_abs)

    ok = cents.max() < 1.0 and conf_abs < 1e-3
    log.info("Verification %s", "PASSED" if ok else "FAILED")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Export pretrained PESTO to ONNX.")
    parser.add_argument("--model-name", default="mir-1k_g7", help="PESTO checkpoint name")
    parser.add_argument("--step-size", type=float, default=10.0, help="hop size in ms")
    parser.add_argument("--sample-rate", type=int, default=44100, help="baked-in sample rate")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "pesto.onnx")
    parser.add_argument("--verify-audio", type=Path, default=None,
                        help="WAV to verify against (default: first file in data/v0/guitar/)")
    parser.add_argument("--no-verify", action="store_true", help="skip numeric verification")
    args = parser.parse_args()

    model = build_model(args.model_name, args.step_size, args.sample_rate)
    export_onnx(model, args.sample_rate, args.opset, args.output)

    hop_samples = int(args.step_size * args.sample_rate / 1000 + 0.5)
    meta = {
        "model_name": args.model_name,
        "sample_rate": args.sample_rate,
        "step_size_ms": args.step_size,
        "hop_samples": hop_samples,
        "opset": args.opset,
        "input": {"name": "audio", "shape": ["num_samples"], "dtype": "float32",
                  "note": "mono waveform, MUST be at sample_rate Hz"},
        "outputs": [
            {"name": "f0_hz", "shape": ["num_frames"], "dtype": "float32",
             "note": "fundamental frequency in Hz (convert_to_freq=True)"},
            {"name": "confidence", "shape": ["num_frames"], "dtype": "float32",
             "note": "voicing confidence in [0, 1]"},
        ],
    }
    meta_path = args.output.with_name("pesto_onnx_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Wrote I/O spec -> %s", meta_path)

    if args.no_verify:
        return 0

    audio_path = args.verify_audio
    if audio_path is None:
        candidates = sorted((ROOT / "data" / "v0" / "guitar").glob("*.wav"))
        if not candidates:
            log.warning("No verification audio found in data/v0/guitar/ — skipping verify")
            return 0
        audio_path = candidates[0]

    return 0 if verify(model, args.output, args.sample_rate, audio_path) else 1


if __name__ == "__main__":
    sys.exit(main())

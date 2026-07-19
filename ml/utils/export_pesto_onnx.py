"""Export pretrained PESTO to ONNX in **streaming** mode for realtime use in the JUCE plugin.

Per PESTO v2 (arxiv 2508.01488) the proper realtime usage is:
  * `load_model(..., streaming=True, mirror=1.0)` — switches CQT from `RegularCQT`
    (reflect-pad both sides, designed for offline) to `StreamingCQT` with
    `CachedConv1d` (left pad = cache of real previous samples, right pad = mirror_fn fake).
  * The model becomes stateful: after each forward, `CachedConv1d.cache.pad` holds the
    last `padding` input samples for the next call.
  * ONNX Runtime is stateless, so we wrap the model in `StatelessPESTO`
    (taken from `realtime/onnx_wrapper.py` of the PESTO repo) which exposes
    that cache as an explicit input/output tensor of the ONNX graph.

ONNX I/O after this script:
    inputs:  audio (1, chunk_size), cache (1, cache_size)
    outputs: pred, confidence, volume, activations, cache_out (1, cache_size)

The model receives `chunk_size = hop_samples` (441 = 10 ms @ 44.1 kHz) on each call
and emits **one** frame — vs ~26 frames with the previous offline export.

`mirror_fn=refill` (RefillPad1d) is preferred over the default `zeros` for
quasi-periodic signals (guitar): on the right edge it duplicates the last
samples of [cache + chunk] instead of zero-padding. `HarmonicCQT` does not
expose `mirror_fn`, so we patch the `mirror` submodule of each `CachedConv1d`
in-place after `load_model`.

The CQT kernels are baked for a fixed sample rate. The plugin MUST feed audio
at exactly `--sample-rate` (default 44100).

Outputs (into models/, gitignored):
  models/pesto.onnx
  models/pesto_onnx_meta.json

Usage
-----
    poetry run python ml/utils/export_pesto_onnx.py
    poetry run python ml/utils/export_pesto_onnx.py --sample-rate 44100 --opset 17
    poetry run python ml/utils/export_pesto_onnx.py --verify-audio data/v0/guitar/E2_260427_1446.wav
"""
import argparse
import json
import logging
import math
import sys
import types
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from pesto.loader import load_model
from pesto.utils.cached_conv import CachedConv1d, RefillPad1d

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("export_pesto_onnx")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEP_SIZE = 10.0
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_MIRROR = 0.8
DEFAULT_MIRROR_FN = "zeros"


def _load_train_config(model_name: str, config_path: Path | None) -> tuple[dict, Path | None]:
    """Load an explicit config or discover config.json beside a custom checkpoint."""
    if config_path is None:
        model_path = Path(model_name)
        if model_path.is_file():
            candidate = model_path.parent / "config.json"
            config_path = candidate if candidate.is_file() else None
    elif not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    if config_path is None:
        return {}, None

    with config_path.open() as f:
        config = json.load(f)
    log.info("Loaded training config: %s", config_path)
    return config, config_path


def _prefer_cli(cli_value, config: dict, key: str, default):
    return cli_value if cli_value is not None else config.get(key, default)


def _load_confidence_state(model_name: str) -> dict[str, torch.Tensor]:
    """Load only ConfidenceClassifier weights from a PESTO checkpoint."""
    checkpoint_path = Path(model_name)
    if not checkpoint_path.is_file():
        import pesto
        checkpoint_path = Path(pesto.__file__).parent / "weights" / f"{model_name}.ckpt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Confidence checkpoint not found: {model_name}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    prefix = "confidence."
    confidence_state = {
        key.removeprefix(prefix): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    }
    if not confidence_state:
        raise ValueError(f"Checkpoint has no confidence weights: {checkpoint_path}")
    return confidence_state


def _preprocessor_forward_no_complex(self, x, sr=None):
    """Drop-in replacement for pesto `Preprocessor.forward` without `torch.view_as_complex`.

    Original computes `view_as_complex(hcqt).abs()` which uses a complex dtype
    op that ONNX cannot trace. The HCQT already outputs (..., 2) real/imag pairs,
    so we compute the magnitude sqrt(re^2 + im^2) directly — numerically identical.
    """
    hcqt = self.hcqt(x, sr=sr)                                  # (B, harmonics, freqs, time, 2)
    mag = torch.sqrt(hcqt[..., 0] ** 2 + hcqt[..., 1] ** 2)     # (B, harmonics, freqs, time)
    mag = mag.permute(0, 3, 1, 2)                               # (B, time, harmonics, freqs)
    return self.to_log(mag)


def _patch_mirror_to_refill(model: nn.Module) -> int:
    """Replace each CachedConv1d.mirror (ZeroPad1d by default) with RefillPad1d.

    `HarmonicCQT.__init__` does not forward `mirror_fn` down to `CachedConv1d`,
    so the only way to switch to refill-pad on the right edge is to walk the
    model post-construction and swap the submodule. We extract the original
    right-padding amount from `mirror.padding[1]` (set by CachedConv1d as
    `(0, mirrored_samples)`) and rebuild with `RefillPad1d`.

    Returns the number of patched modules.
    """
    patched = 0
    for name, m in model.named_modules():
        if not isinstance(m, CachedConv1d):
            continue
        mir = m.mirror
        pad_attr = getattr(mir, "padding", None)
        if pad_attr is None:
            log.warning("Module %s mirror has no .padding, skipping", name)
            continue
        # padding is typically a tuple (left, right) for 1d pad layers
        if isinstance(pad_attr, int):
            right = pad_attr
        else:
            right = pad_attr[1] if len(pad_attr) >= 2 else pad_attr[0]
        if right == 0:
            log.info("Module %s has zero right-mirror, leaving as-is", name)
            continue
        m.mirror = RefillPad1d((0, right))
        patched += 1
        log.info("Patched %s: ZeroPad1d(0, %d) -> RefillPad1d((0, %d))",
                 name, right, right)
    return patched


class StatelessPESTO(nn.Module):
    """Externalize PESTO cache state as explicit input/output for ONNX Runtime.

    Adapted from `realtime/onnx_wrapper.py` in the PESTO repository. Collects
    all `CachedConv1d.cache.pad` buffers into a single flattened tensor on
    output, and scatters an input cache tensor back into them before forward.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.cache_size_dict: Dict[str, Tuple[int, ...]] = {}
        for name, m in self.model.named_modules():
            if isinstance(m, CachedConv1d) and hasattr(m.cache, "pad"):
                self.cache_size_dict[name + "-cache"] = tuple(m.cache.pad.shape)
        self.cache_size = sum(math.prod(s) for s in self.cache_size_dict.values())

    def init_cache(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        if self.cache_size == 0:
            return torch.empty(batch_size, 0, device=device)
        return torch.zeros(batch_size, self.cache_size, device=device)

    def _set_caches(self, cache: torch.Tensor) -> None:
        """Write the flat input cache back into each CachedConv1d.cache.pad."""
        if cache.numel() == 0:
            return
        cache_flat = cache[0]
        ptr = 0
        for name, m in self.model.named_modules():
            if isinstance(m, CachedConv1d) and hasattr(m.cache, "pad"):
                shape = self.cache_size_dict[name + "-cache"]
                n = math.prod(shape)
                m.cache.pad = cache_flat[ptr:ptr + n].view(shape)
                ptr += n

    def _gather_caches(self, batch_size: int) -> torch.Tensor:
        """Read each CachedConv1d.cache.pad into a single flat tensor."""
        parts = []
        for name, m in self.model.named_modules():
            if isinstance(m, CachedConv1d) and hasattr(m.cache, "pad"):
                parts.append(m.cache.pad.flatten())
        if not parts:
            return torch.empty(batch_size, 0)
        cat = torch.cat(parts, dim=0)
        return cat.unsqueeze(0).expand(batch_size, -1)

    def forward(self, audio: torch.Tensor, cache: torch.Tensor):
        """audio: (B, chunk_size), cache: (B, cache_size) -> (f0, conf, vol, acts, cache_out)."""
        self._set_caches(cache)
        preds, confidence, vol, activations = self.model(
            audio, sr=None, convert_to_freq=True, return_activations=True
        )
        cache_out = self._gather_caches(audio.size(0))
        return preds, confidence, vol, activations, cache_out


def build_model(model_name: str,
                step_size: float,
                sample_rate: int,
                max_batch_size: int,
                mirror: float,
                mirror_fn: str,
                confidence_model: str | None = None) -> nn.Module:
    log.info("Loading PESTO '%s' (step_size=%.1f ms, sr=%d Hz, streaming, "
             "mirror=%.2f, mirror_fn=%s)",
             model_name, step_size, sample_rate, mirror, mirror_fn)
    model = load_model(
        model_name,
        step_size=step_size,
        sampling_rate=sample_rate,
        streaming=True,
        max_batch_size=max_batch_size,
        mirror=mirror,
    )
    if confidence_model:
        confidence_state = _load_confidence_state(confidence_model)
        model.confidence.load_state_dict(confidence_state, strict=True)
        log.info("Loaded confidence weights from '%s'", confidence_model)
    model.eval()
    model.preprocessor.forward = types.MethodType(_preprocessor_forward_no_complex, model.preprocessor)
    if mirror_fn == "refill":
        n_patched = _patch_mirror_to_refill(model)
        log.info("Patched %d CachedConv1d mirror(s) to RefillPad1d", n_patched)
    else:
        log.info("Using default mirror_fn='zeros' (no patching)")
    return model


def export_onnx(model: nn.Module,
                chunk_size: int,
                opset: int,
                output_path: Path) -> Tuple[int, int]:
    """Export streaming PESTO as ONNX. Returns (chunk_size, cache_size)."""
    wrapper = StatelessPESTO(model).eval()
    cache_size = wrapper.cache_size
    log.info("Cache size: %d float32 (%d shapes: %s)",
             cache_size, len(wrapper.cache_size_dict),
             {k: v for k, v in wrapper.cache_size_dict.items()})

    dummy_audio = torch.randn(1, chunk_size, dtype=torch.float32).clip(-1, 1)
    dummy_cache = wrapper.init_cache(batch_size=1)
    with torch.no_grad():
        preds, conf, vol, acts, cache_out = wrapper(dummy_audio, dummy_cache)
    log.info("Wrapper sanity: audio=%s cache=%s -> pred=%s conf=%s vol=%s acts=%s cache_out=%s",
             tuple(dummy_audio.shape), tuple(dummy_cache.shape),
             tuple(preds.shape), tuple(conf.shape), tuple(vol.shape),
             tuple(acts.shape), tuple(cache_out.shape))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_audio, dummy_cache),
        str(output_path),
        input_names=["audio", "cache"],
        output_names=["f0_hz", "confidence", "volume", "activations", "cache_out"],
        dynamic_axes={
            "audio":      {0: "batch"},
            "cache":      {0: "batch"},
            "f0_hz":      {0: "batch", 1: "frames"},
            "confidence": {0: "batch", 1: "frames"},
            "volume":     {0: "batch", 1: "frames"},
            "activations": {0: "batch", 1: "frames"},
            "cache_out":  {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    log.info("Exported ONNX -> %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return chunk_size, cache_size


def verify_streaming(model: nn.Module,
                     onnx_path: Path,
                     sample_rate: int,
                     chunk_size: int,
                     cache_size: int,
                     audio_path: Path) -> bool:
    """Run ONNX chunk-by-chunk and compare with the same streaming torch model.

    Both should give bitwise-close results since the ONNX graph IS the same
    torch model + cache externalization. Catches export bugs (missing ops,
    state mismatch, etc.), not algorithmic regressions vs offline.
    """
    import librosa
    import onnxruntime as ort

    audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    audio = audio.astype(np.float32)
    n_chunks = audio.size // chunk_size
    audio = audio[: n_chunks * chunk_size]
    log.info("Verifying on %s (%d samples = %d chunks of %d)",
             audio_path.name, audio.size, n_chunks, chunk_size)

    # Torch reference: feed same chunks, model maintains its own cache internally
    model.eval()
    # Reset internal cache by re-zeroing all CachedConv1d.cache.pad
    for m in model.modules():
        if isinstance(m, CachedConv1d) and hasattr(m.cache, "pad"):
            m.cache.pad = torch.zeros_like(m.cache.pad)

    ref_f0, ref_conf = [], []
    with torch.no_grad():
        for i in range(n_chunks):
            chunk = torch.from_numpy(audio[i * chunk_size:(i + 1) * chunk_size]).unsqueeze(0)
            f0, conf, _vol = model(chunk, sr=None, convert_to_freq=True, return_activations=False)
            ref_f0.append(f0.item())
            ref_conf.append(conf.item())
    ref_f0 = np.array(ref_f0, dtype=np.float32)
    ref_conf = np.array(ref_conf, dtype=np.float32)

    # ONNX run with externalized cache
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    cache_state = np.zeros((1, cache_size), dtype=np.float32)
    onnx_f0, onnx_conf = [], []
    for i in range(n_chunks):
        chunk = audio[i * chunk_size:(i + 1) * chunk_size].reshape(1, -1)
        out = sess.run(None, {"audio": chunk, "cache": cache_state})
        f0, conf, _vol, _acts, cache_state = out
        onnx_f0.append(float(f0.ravel()[0]))
        onnx_conf.append(float(conf.ravel()[0]))
    onnx_f0 = np.array(onnx_f0, dtype=np.float32)
    onnx_conf = np.array(onnx_conf, dtype=np.float32)

    f0_abs = float(np.max(np.abs(onnx_f0 - ref_f0)))
    mask = (ref_f0 > 1.0) & (onnx_f0 > 1.0)
    if mask.any():
        cents = 1200.0 * np.abs(np.log2(onnx_f0[mask] / ref_f0[mask]))
        log.info("f0:   max abs %.4e Hz | max %.4f cents | mean %.4f cents",
                 f0_abs, float(cents.max()), float(cents.mean()))
    else:
        cents = np.array([0.0])
        log.warning("No voiced frames detected — cents check skipped")
    conf_abs = float(np.max(np.abs(onnx_conf - ref_conf)))
    log.info("conf: max abs %.4e", conf_abs)
    log.info("voiced ratio (ref): %.2f", float((ref_conf >= 0.5).mean()))

    ok = (cents.max() < 1.0) and (conf_abs < 1e-3)
    log.info("Verification %s", "PASSED" if ok else "FAILED")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Export streaming PESTO to ONNX.")
    parser.add_argument("--model-name", default="mir-1k_g7", help="PESTO checkpoint name")
    parser.add_argument("--train-config", type=Path, default=None,
                        help="training config.json; default: discover beside --model-name checkpoint")
    parser.add_argument("--confidence-model", default="",
                        help="optional checkpoint name/path providing confidence.* weights; "
                             "empty keeps the confidence loaded with --model-name")
    parser.add_argument("--step-size", type=float, default=None,
                        help="hop size in ms; overrides training config chunk_size")
    parser.add_argument("--sample-rate", type=int, default=None,
                        help="baked-in sample rate; overrides training config")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--mirror", type=float, default=None,
                        help="fraction of right-edge fake samples (1.0 = zero added latency, "
                             "0.5 = ~44 ms latency / best accuracy). 0.8 is the elbow of the "
                             "latency/accuracy curve on guitar recordings.")
    parser.add_argument("--mirror-fn", choices=["zeros", "refill"], default=None,
                        help="how to fill the right edge fake zone. 'zeros' (default) works "
                             "best on the pretrained checkpoint; 'refill' is intended for "
                             "future realtime-finetuned models (see ROADMAP).")
    parser.add_argument("--max-batch-size", type=int, default=1, help="ONNX max batch dim")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "pesto.onnx")
    parser.add_argument("--verify-audio", type=Path, default=None,
                        help="WAV to verify against (default: first file in data/v0/guitar/)")
    parser.add_argument("--no-verify", action="store_true", help="skip numeric verification")
    args = parser.parse_args()

    train_config, train_config_path = _load_train_config(args.model_name, args.train_config)
    sample_rate = int(_prefer_cli(args.sample_rate, train_config, "sample_rate", DEFAULT_SAMPLE_RATE))
    mirror = float(_prefer_cli(args.mirror, train_config, "mirror", DEFAULT_MIRROR))
    mirror_fn = str(_prefer_cli(args.mirror_fn, train_config, "mirror_fn", DEFAULT_MIRROR_FN))

    if args.step_size is not None:
        step_size = args.step_size
        chunk_size = int(step_size * sample_rate / 1000 + 0.5)
    elif "chunk_size" in train_config:
        chunk_size = int(train_config["chunk_size"])
        step_size = 1000.0 * chunk_size / sample_rate
    else:
        step_size = float(train_config.get("step_size_ms", DEFAULT_STEP_SIZE))
        chunk_size = int(step_size * sample_rate / 1000 + 0.5)

    log.info("Resolved export config: sr=%d, step=%.6f ms, chunk=%d, mirror=%.2f, mirror_fn=%s",
             sample_rate, step_size, chunk_size, mirror, mirror_fn)

    model = build_model(args.model_name, step_size, sample_rate,
                        args.max_batch_size, mirror, mirror_fn,
                        confidence_model=args.confidence_model or None)
    _, cache_size = export_onnx(model, chunk_size, args.opset, args.output)

    meta = {
        "model_name": args.model_name,
        "confidence_model": args.confidence_model or None,
        "train_config": str(train_config_path) if train_config_path else None,
        "sample_rate": sample_rate,
        "step_size_ms": step_size,
        "hop_samples": chunk_size,
        "chunk_size": chunk_size,
        "cache_size": cache_size,
        "mirror": mirror,
        "mirror_fn": mirror_fn,
        "max_batch_size": args.max_batch_size,
        "opset": args.opset,
        "inputs": [
            {"name": "audio", "shape": ["batch", chunk_size], "dtype": "float32",
             "note": f"mono waveform chunk, {chunk_size} samples @ {sample_rate} Hz"},
            {"name": "cache", "shape": ["batch", cache_size], "dtype": "float32",
             "note": "streaming state from previous call; zeros on first call"},
        ],
        "outputs": [
            {"name": "f0_hz", "shape": ["batch", "frames"], "dtype": "float32",
             "note": "F0 in Hz, 1 frame per call when chunk_size == hop_samples"},
            {"name": "confidence", "shape": ["batch", "frames"], "dtype": "float32",
             "note": "voicing confidence in [0, 1]"},
            {"name": "volume", "shape": ["batch", "frames"], "dtype": "float32",
             "note": "frame energy"},
            {"name": "activations", "shape": ["batch", "frames", "bins"], "dtype": "float32",
             "note": "raw pitch activation logits"},
            {"name": "cache_out", "shape": ["batch", cache_size], "dtype": "float32",
             "note": "updated streaming state; feed back as 'cache' on next call"},
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

    return 0 if verify_streaming(model, args.output, sample_rate,
                                 chunk_size, cache_size, audio_path) else 1


if __name__ == "__main__":
    sys.exit(main())

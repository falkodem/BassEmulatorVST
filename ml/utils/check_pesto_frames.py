"""Verify actual output frame count from the exported PESTO ONNX model.

Usage:
    poetry run python ml/utils/check_pesto_frames.py

Checks what num_frames PESTO emits for our plugin's kWindowSamples=11025,
so we can confirm kFramesPerWindow in PestoPitchDetector.h is set correctly.
"""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "pesto.onnx"

def main():
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not installed. Run: poetry add onnxruntime")
        return 1

    if not MODEL_PATH.exists():
        print(f"ERROR: model not found at {MODEL_PATH}")
        print("Run: poetry run python ml/utils/export_pesto_onnx.py")
        return 1

    sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])

    # Check model I/O metadata
    print("=== Model Inputs ===")
    for inp in sess.get_inputs():
        print(f"  {inp.name}: shape={inp.shape}, dtype={inp.type}")

    print("=== Model Outputs ===")
    for out in sess.get_outputs():
        print(f"  {out.name}: shape={out.shape}, dtype={out.type}")

    print()

    # Run with kWindowSamples = 11025 (as used in the plugin)
    kWindowSamples = 11025
    kHopSamples    = 441

    audio = np.zeros(kWindowSamples, dtype=np.float32)
    f0_hz, confidence = sess.run(None, {"audio": audio})

    print(f"Input:  {kWindowSamples} samples ({kWindowSamples/44100*1000:.1f} ms)")
    print(f"Output: f0_hz.shape     = {f0_hz.shape}")
    print(f"        confidence.shape = {confidence.shape}")
    print()

    n_frames = f0_hz.shape[0]
    formula_n_hop = kWindowSamples // kHopSamples           # = 25
    formula_n_hop_plus1 = kWindowSamples // kHopSamples + 1 # = 26

    print(f"kWindowSamples / kHopSamples         = {formula_n_hop}  (kFramesPerWindow should be THIS if n_frames == {formula_n_hop})")
    print(f"kWindowSamples / kHopSamples + 1     = {formula_n_hop_plus1}  (kFramesPerWindow should be THIS if n_frames == {formula_n_hop_plus1})")
    print()

    if n_frames == formula_n_hop:
        print(f">>> RESULT: PESTO outputs {n_frames} frames — kFramesPerWindow should be {formula_n_hop}")
        print(f"    Current value in PestoPitchDetector.h: 26")
        if n_frames != 26:
            print(f"    *** BUG: kFramesPerWindow=26 is WRONG — change to {n_frames} ***")
        else:
            print(f"    OK: matches current setting.")
    elif n_frames == formula_n_hop_plus1:
        print(f">>> RESULT: PESTO outputs {n_frames} frames — kFramesPerWindow should be {formula_n_hop_plus1}")
        print(f"    Current value in PestoPitchDetector.h: 26")
        if n_frames != 26:
            print(f"    *** BUG: kFramesPerWindow=26 is WRONG — change to {n_frames} ***")
        else:
            print(f"    OK: matches current setting.")
    else:
        print(f">>> RESULT: PESTO outputs {n_frames} frames — unexpected value!")
        print(f"    Update kFramesPerWindow in PestoPitchDetector.h to {n_frames}")

    # Also check with real guitar-like audio (sine at E2 = 82.4 Hz)
    print()
    print("--- Running with sine at E2 (82.4 Hz) ---")
    t = np.linspace(0, kWindowSamples / 44100, kWindowSamples, endpoint=False).astype(np.float32)
    sine_e2 = (0.5 * np.sin(2 * np.pi * 82.4 * t)).astype(np.float32)
    f0_hz2, conf2 = sess.run(None, {"audio": sine_e2})
    print(f"f0_hz[-5:] = {f0_hz2[-5:]}")
    print(f"conf[-5:]  = {conf2[-5:]}")
    print(f"max conf   = {conf2.max():.4f}, mean conf = {conf2.mean():.4f}")
    voiced = conf2 >= 0.5
    print(f"voiced frames (conf >= 0.5): {voiced.sum()} / {len(conf2)}")
    if voiced.any():
        print(f"median f0 (voiced) = {np.median(f0_hz2[voiced]):.2f} Hz (expected ~82.4 Hz)")
    else:
        print("WARNING: no voiced frames detected on E2 sine — model or input may be wrong")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

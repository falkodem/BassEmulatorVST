"""Synthesize sawtooth audio from F0 curves for auditory pitch-detector evaluation.

Usage
-----
    python ml/pitch_eval/synthesize.py \\
        --run runs/pitch_eval/20260513_215529   # required: pitch_eval run directory
        --detectors yin,pesto                   # default: all detectors found in CSV
        --amp 0.3                               # sawtooth amplitude 0..1, default 0.3
        --lpf-cutoff 3000                       # low-pass cutoff Hz; 0 disables

For each <stem> subdirectory inside <run>/, reads f0_curves.csv and writes
synth_<detector>.wav next to audio.wav.  Output: mono, 44100 Hz, float32.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_SR = 44100
_FADE_MS = 10.0  # voiced/unvoiced boundary fade length, milliseconds


# ---------------------------------------------------------------------------
# Core synthesis
# ---------------------------------------------------------------------------

def synth_sawtooth(
    times: np.ndarray,
    f0: np.ndarray,
    n_samples: int,
    sr: int,
    amp: float,
    lpf_cutoff: float = 0.0,
) -> np.ndarray:
    """Synthesize a naive sawtooth wave driven by a per-frame F0 curve.

    Args:
        times:      1-D array of frame timestamps (seconds), shape (N,)
        f0:         1-D array of F0 values (Hz), shape (N,); 0 = unvoiced
        n_samples:  output length in samples (must match original audio for alignment)
        sr:         sample rate in Hz
        amp:        peak amplitude of the sawtooth, 0..1
        lpf_cutoff: low-pass filter cutoff in Hz; <= 0 disables filtering

    Returns:
        output: np.ndarray, shape (n_samples,), float32, clipped to [-1, 1]
    """
    if n_samples == 0:
        return np.zeros(0, dtype=np.float32)

    sample_times = np.arange(n_samples) / sr  # (n_samples,)
    voiced_mask = f0 > 0.0

    # If no voiced frames at all, return silence
    if not np.any(voiced_mask):
        log.debug("No voiced frames — returning silence")
        return np.zeros(n_samples, dtype=np.float32)

    # -----------------------------------------------------------------------
    # 1. Per-sample F0: interpolate only over voiced timestamps so that
    #    frequency does not collapse to 0 in unvoiced gaps.
    # -----------------------------------------------------------------------
    voiced_times = times[voiced_mask]
    voiced_f0 = f0[voiced_mask]

    # np.interp clamps at boundary values — fine for our purposes
    f0_per_sample = np.interp(sample_times, voiced_times, voiced_f0)

    # -----------------------------------------------------------------------
    # 2. Phase accumulation → naive sawtooth
    # -----------------------------------------------------------------------
    phase = np.cumsum(2.0 * np.pi * f0_per_sample / sr)  # (n_samples,)
    sawtooth = 2.0 * (np.mod(phase / (2.0 * np.pi), 1.0)) - 1.0  # [-1, 1]

    # -----------------------------------------------------------------------
    # 2b. Low-pass filter: a naive sawtooth has slow 1/n harmonic rolloff and
    #     sounds harsh. Zero-phase Butterworth keeps sample alignment intact.
    # -----------------------------------------------------------------------
    if 0.0 < lpf_cutoff < sr / 2.0:
        sos = butter(4, lpf_cutoff, btype="low", fs=sr, output="sos")
        # sosfiltfilt needs the signal longer than its edge padding
        if len(sawtooth) > 3 * sos.shape[0] * 2:
            sawtooth = sosfiltfilt(sos, sawtooth)

    # -----------------------------------------------------------------------
    # 3. Per-sample voiced envelope: build binary mask, then smooth with
    #    a Hann window (~10 ms) to create click-free fades on boundaries.
    # -----------------------------------------------------------------------
    # Nearest-neighbor lookup: for each output sample, find the closest frame
    voiced_frame_mask = voiced_mask.astype(np.float64)

    # Interpolate the 0/1 voiced flag onto per-sample grid
    # (nearest-neighbor via np.interp over full frame array)
    envelope = np.interp(sample_times, times, voiced_frame_mask)

    # Smooth with a Hann window of fade_samples length to get fade in/out
    fade_samples = int(round(_FADE_MS * 1e-3 * sr))
    if fade_samples > 1:
        # Use a causal-symmetric Hann window (odd length for symmetry)
        win_len = fade_samples * 2 + 1
        hann = np.hanning(win_len)
        hann /= hann.sum()  # normalize so DC gain = 1
        envelope = np.convolve(envelope, hann, mode="same")

    # Clamp to [0, 1] after convolution ringing
    envelope = np.clip(envelope, 0.0, 1.0)

    # -----------------------------------------------------------------------
    # 4. Mix and clip
    # -----------------------------------------------------------------------
    output = sawtooth * envelope * amp
    output = np.clip(output, -1.0, 1.0)
    return output.astype(np.float32)


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def read_f0_curves(csv_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read f0_curves.csv and return {detector_name: (times, f0_hz)}.

    Uses stdlib csv — no pandas dependency required.
    """
    data: dict[str, tuple[list, list]] = {}

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            det = row["detector"]
            t = float(row["time_s"])
            f0 = float(row["f0_hz"])
            if det not in data:
                data[det] = ([], [])
            data[det][0].append(t)
            data[det][1].append(f0)

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for det, (ts, f0s) in data.items():
        times = np.array(ts, dtype=np.float64)
        f0 = np.array(f0s, dtype=np.float64)
        # Sort by time just in case
        order = np.argsort(times)
        result[det] = (times[order], f0[order])

    return result


# ---------------------------------------------------------------------------
# Per-stem processing
# ---------------------------------------------------------------------------

def process_stem(
    stem_dir: Path,
    detectors: list[str] | None,
    amp: float,
    lpf_cutoff: float,
) -> None:
    """Read f0_curves.csv and write synth_<detector>.wav for each detector."""
    csv_path = stem_dir / "f0_curves.csv"
    audio_path = stem_dir / "audio.wav"

    if not csv_path.exists():
        log.warning("No f0_curves.csv in %s — skipping", stem_dir)
        return
    if not audio_path.exists() and not audio_path.is_symlink():
        log.warning("No audio.wav in %s — skipping", stem_dir)
        return

    # Read audio metadata for exact sample count
    audio_info = sf.info(str(audio_path))
    n_samples = audio_info.frames
    sr = audio_info.samplerate
    if sr != _SR:
        log.warning(
            "%s: sample rate %d != expected %d — using actual sr for synthesis",
            stem_dir.name, sr, _SR,
        )

    log.info("Stem: %s  (%d samples, %.2f s)", stem_dir.name, n_samples, n_samples / sr)

    # Parse CSV
    curves = read_f0_curves(csv_path)

    # Resolve detector list
    available = list(curves.keys())
    if detectors is None:
        target_detectors = available
    else:
        target_detectors = detectors
        unknown = [d for d in target_detectors if d not in available]
        if unknown:
            log.warning(
                "Stem %s: detectors not found in CSV: %s. Available: %s",
                stem_dir.name, unknown, available,
            )

    for det in target_detectors:
        if det not in curves:
            continue

        times, f0 = curves[det]
        out_path = stem_dir / f"synth_{det}.wav"

        log.info("  Synthesizing %s → %s", det, out_path.name)
        audio_out = synth_sawtooth(times, f0, n_samples, sr, amp, lpf_cutoff)

        sf.write(str(out_path), audio_out, sr, subtype="FLOAT")

        peak = float(np.max(np.abs(audio_out)))
        voiced_frac = float(np.mean(f0 > 0))
        log.info(
            "    peak=%.4f  voiced_frames=%.1f%%  wrote %d samples",
            peak, voiced_frac * 100, len(audio_out),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize sawtooth WAVs from pitch-eval F0 curves."
    )
    parser.add_argument(
        "--run",
        required=True,
        help="Path to a pitch_eval run directory (contains <stem>/f0_curves.csv).",
    )
    parser.add_argument(
        "--detectors",
        default=None,
        help="Comma-separated detector names to synthesize. Default: all found in CSV.",
    )
    parser.add_argument(
        "--amp",
        type=float,
        default=0.3,
        help="Sawtooth amplitude, 0..1 (default: 0.3).",
    )
    parser.add_argument(
        "--lpf-cutoff",
        type=float,
        default=3000.0,
        help="Low-pass filter cutoff in Hz to tame the harsh sawtooth. "
             "0 disables filtering (default: 3000).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_dir():
        log.error("Run directory not found: %s", run_dir)
        sys.exit(1)

    detectors: list[str] | None = None
    if args.detectors is not None:
        detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]

    if not (0.0 < args.amp <= 1.0):
        log.error("--amp must be in (0, 1], got %s", args.amp)
        sys.exit(1)

    if args.lpf_cutoff < 0.0:
        log.error("--lpf-cutoff must be >= 0, got %s", args.lpf_cutoff)
        sys.exit(1)

    log.info("Run: %s", run_dir)
    log.info("Detectors: %s", detectors if detectors else "all")
    log.info("Amplitude: %.3f", args.amp)
    log.info("LPF cutoff: %s", f"{args.lpf_cutoff:.0f} Hz" if args.lpf_cutoff > 0 else "off")

    # Collect stem subdirectories (dirs that contain f0_curves.csv)
    stem_dirs = sorted(
        p for p in run_dir.iterdir()
        if p.is_dir() and (p / "f0_curves.csv").exists()
    )

    if not stem_dirs:
        log.error("No stem directories with f0_curves.csv found in %s", run_dir)
        sys.exit(1)

    log.info("Found %d stem(s)", len(stem_dirs))

    for stem_dir in stem_dirs:
        process_stem(stem_dir, detectors, args.amp, args.lpf_cutoff)

    log.info("Done.")


if __name__ == "__main__":
    main()

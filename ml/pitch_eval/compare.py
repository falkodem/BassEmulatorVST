"""Utilities for comparing pitch detector outputs.

Functions
---------
align_to_grid(result, hop_ms, duration_s) -> np.ndarray
    Interpolate a PitchResult onto a uniform time grid with given hop.
    Voiced frames (f0 > 0) are linearly interpolated; unvoiced frames -> NaN.

pairwise_disagreement(a, b) -> dict
    Compute disagreement metrics between two aligned F0 arrays (same grid).

compute_signal_activity(audio, sr, hop_ms, frame_ms, noise_percentile, margin_db)
    Frame-level signal activity from RMS energy with adaptive noise floor.

detector_vs_activity(f0_grid, signal_active) -> dict
    Per-detector recall/precision against signal activity ground truth.
"""

import logging
import math

import librosa
import numpy as np

from .detectors.base import PitchResult

log = logging.getLogger(__name__)


def compute_signal_activity(
    audio: np.ndarray,
    sr: int,
    hop_ms: float = 10.0,
    frame_ms: float = 30.0,
    noise_percentile: float = 10.0,
    margin_db: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Frame-level signal activity from RMS energy with adaptive noise floor.

    Algorithm:
        1. Compute frame RMS on a sliding window of `frame_ms` ms, hop `hop_ms` ms.
        2. Convert to dB: rms_db = 20 * log10(rms + 1e-10).
        3. Estimate noise floor as `noise_percentile`-th percentile of rms_db.
        4. threshold_db = noise_floor + margin_db.
        5. active = rms_db > threshold_db.

    Args:
        audio:            1-D audio samples (float32/float64)
        sr:               sample rate, Hz
        hop_ms:           hop between frames, milliseconds (default 10.0)
        frame_ms:         RMS window length, milliseconds (default 30.0)
        noise_percentile: percentile used to estimate noise floor (default 10.0)
        margin_db:        dB above noise floor to set activity threshold (default 10.0)

    Returns:
        times:  np.ndarray shape (T,), seconds — centers of frames on uniform hop grid
        active: np.ndarray shape (T,), bool — True where signal is active
    """
    hop_samples = int(round(sr * hop_ms / 1000.0))
    frame_samples = int(round(sr * frame_ms / 1000.0))

    # librosa.feature.rms returns shape (1, T)
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_samples,
        hop_length=hop_samples,
        center=True,
    )[0]  # shape (T,)

    rms_db = 20.0 * np.log10(rms + 1e-10)

    noise_floor = float(np.percentile(rms_db, noise_percentile))
    threshold_db = noise_floor + margin_db

    active = rms_db > threshold_db

    # Frame center times (same convention as librosa: frame i is centered at
    # i * hop_samples / sr when center=True)
    n_frames = len(rms)
    times = np.arange(n_frames) * hop_samples / sr

    return times, active


def align_to_grid(
    result: PitchResult,
    hop_ms: float = 10.0,
    duration_s: float = None,
) -> np.ndarray:
    """Interpolate PitchResult onto a uniform time grid.

    Voiced frames (f0_hz > 0) are interpolated with np.interp.
    Frames on the output grid that fall outside any voiced segment are NaN.

    Args:
        result:     PitchResult from any detector
        hop_ms:     target grid hop in milliseconds
        duration_s: total duration; defaults to result.times[-1]

    Returns:
        f0_grid: np.ndarray shape (M,), Hz or NaN, on grid spacing hop_ms/1000 s
    """
    if duration_s is None:
        duration_s = float(result.times[-1])

    hop_s = hop_ms / 1000.0
    grid = np.arange(0.0, duration_s + hop_s * 0.5, hop_s)

    # Mark voiced frames
    voiced_mask = result.f0_hz > 0.0

    if not np.any(voiced_mask):
        return np.full(len(grid), np.nan)

    # Interpolate only over voiced segments
    # Strategy: interpolate the full curve, then NaN-out frames that land in
    # unvoiced gaps (i.e. where both nearest neighbors are unvoiced).
    f0_interp = np.interp(grid, result.times, result.f0_hz)

    # For each grid point determine if it lies in a voiced segment:
    # we say it's voiced if the nearest source frame is voiced.
    nearest_idx = np.searchsorted(result.times, grid, side="left").clip(0, len(result.times) - 1)
    # Also check the previous index
    prev_idx = (nearest_idx - 1).clip(0, len(result.times) - 1)

    nearest_voiced = voiced_mask[nearest_idx]
    prev_voiced = voiced_mask[prev_idx]

    # A grid point is voiced if at least one of the two surrounding source
    # frames is voiced (loose criterion — avoids chopping at edges).
    on_voiced = nearest_voiced | prev_voiced

    f0_grid = np.where(on_voiced, f0_interp, np.nan)
    return f0_grid


def pairwise_disagreement(a: np.ndarray, b: np.ndarray) -> dict:
    """Compute disagreement metrics between two aligned F0 arrays.

    Both arrays must be on the same time grid (same length).
    NaN entries represent unvoiced frames.

    Returns dict with keys:
        n_frames_total          — total grid frames
        n_both_voiced           — frames where both detectors are voiced
        frac_both_voiced        — n_both_voiced / n_frames_total
        pct_disagree_50c        — % of voiced frames with |deviation| > 50 cents
        mean_abs_cents          — mean |deviation| in cents on voiced frames
        median_abs_cents        — median |deviation| in cents on voiced frames
    """
    assert len(a) == len(b), "Arrays must be the same length (same grid)"

    both_voiced = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    n_total = len(a)
    n_both = int(np.sum(both_voiced))

    if n_both == 0:
        return {
            "n_frames_total": n_total,
            "n_both_voiced": 0,
            "frac_both_voiced": 0.0,
            "pct_disagree_50c": float("nan"),
            "mean_abs_cents": float("nan"),
            "median_abs_cents": float("nan"),
        }

    a_v = a[both_voiced]
    b_v = b[both_voiced]

    # Guard against zeros/negatives slipping through
    valid = (a_v > 0) & (b_v > 0)
    a_v = a_v[valid]
    b_v = b_v[valid]

    if len(a_v) == 0:
        return {
            "n_frames_total": n_total,
            "n_both_voiced": n_both,
            "frac_both_voiced": n_both / n_total,
            "pct_disagree_50c": float("nan"),
            "mean_abs_cents": float("nan"),
            "median_abs_cents": float("nan"),
        }

    cents_diff = np.abs(1200.0 * np.log2(a_v / b_v))

    return {
        "n_frames_total": n_total,
        "n_both_voiced": n_both,
        "frac_both_voiced": n_both / n_total,
        "pct_disagree_50c": float(100.0 * np.mean(cents_diff > 50.0)),
        "mean_abs_cents": float(np.mean(cents_diff)),
        "median_abs_cents": float(np.median(cents_diff)),
    }


# Flag to emit the length-mismatch warning only once per process lifetime
_length_mismatch_warned = False


def detector_vs_activity(
    f0_grid: np.ndarray,
    signal_active: np.ndarray,
) -> dict:
    """Per-detector recall/precision against signal activity ground truth.

    Both arrays must be on the same time grid (same length).  If lengths
    differ, both are truncated to min(len_a, len_b) with a one-time warning.

    "Voiced" is defined as (f0_grid > 0) & np.isfinite(f0_grid).  NaN from
    align_to_grid means "unvoiced".

    Args:
        f0_grid:       np.ndarray shape (M,) — output of align_to_grid
        signal_active: np.ndarray shape (M,), bool — GT activity on same grid

    Returns dict with keys:
        n_active            — frames where signal is active (GT)
        n_voiced            — frames where detector said voiced
        n_active_and_voiced — true positives
        recall              — n_active_and_voiced / n_active   (NaN if n_active==0)
        precision           — n_active_and_voiced / n_voiced   (NaN if n_voiced==0)
        f1                  — harmonic mean of recall/precision (NaN if both==0)
    """
    global _length_mismatch_warned

    if len(f0_grid) != len(signal_active):
        if not _length_mismatch_warned:
            log.warning(
                "detector_vs_activity: length mismatch f0_grid=%d signal_active=%d; "
                "truncating to min. (Suppressing further warnings.)",
                len(f0_grid),
                len(signal_active),
            )
            _length_mismatch_warned = True
        min_len = min(len(f0_grid), len(signal_active))
        f0_grid = f0_grid[:min_len]
        signal_active = signal_active[:min_len]

    voiced = np.isfinite(f0_grid) & (f0_grid > 0)
    active = signal_active.astype(bool)

    n_active = int(np.sum(active))
    n_voiced = int(np.sum(voiced))
    n_tp = int(np.sum(active & voiced))

    recall = float(n_tp / n_active) if n_active > 0 else float("nan")
    precision = float(n_tp / n_voiced) if n_voiced > 0 else float("nan")

    if n_active == 0 and n_voiced == 0:
        f1 = float("nan")
    elif math.isnan(recall) or math.isnan(precision) or (recall + precision) == 0.0:
        f1 = float("nan")
    else:
        f1 = 2.0 * recall * precision / (recall + precision)

    return {
        "n_active": n_active,
        "n_voiced": n_voiced,
        "n_active_and_voiced": n_tp,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }

#!/usr/bin/env python3
"""
Slices guitar/bass WAV pairs into overlapping windows for training.

For each pair in index.csv:
  - Loads audio (expected 44100 Hz), resamples if needed
  - Normalizes BOTH signals by the guitar peak (joint normalization) so that
    the amplitude ratio guitar/bass is preserved — the model can learn gain.
  - Slices into overlapping windows (WINDOW_SIZE=1024, HOP_SIZE=512)
  - Drops silent windows (by guitar RMS)

Output:
  data/v0/windows/guitar.npy   shape (N, 1024)
  data/v0/windows/bass.npy     shape (N, 1024)
  data/v0/windows/meta.csv     window -> source file mapping

Usage:
  poetry run python ml/slice_dataset.py
"""

import csv
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# ── Config ────────────────────────────────────────────────────────────────────

TARGET_SR   = 44_100
WINDOW_SIZE = 1024
HOP_SIZE    = 512
SILENCE_RMS = 1e-3   # windows with guitar RMS below this are dropped

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR   = Path(__file__).resolve().parent.parent / "data" / "v0"
INDEX_CSV  = DATA_DIR / "index.csv"
GUITAR_DIR = DATA_DIR / "guitar"
BASS_DIR   = DATA_DIR / "bass"
OUT_DIR    = DATA_DIR / "windows"

META_FIELDS = ["window_idx", "source_file", "note", "window_in_file"]

# ── Audio helpers ─────────────────────────────────────────────────────────────

def load_mono(path: Path) -> tuple:
    """Load WAV as float32 mono. Returns (audio, sample_rate)."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sr


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    g = gcd(src_sr, dst_sr)
    return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)


def joint_normalize(g: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale both signals by the guitar peak — preserves guitar/bass amplitude ratio."""
    peak = np.abs(g).max()
    if peak > 1e-8:
        g = g / peak
        b = b / peak
    return g, b


def slice_windows(audio: np.ndarray) -> np.ndarray:
    """Return array of shape (N, WINDOW_SIZE) using vectorised indexing."""
    n = (len(audio) - WINDOW_SIZE) // HOP_SIZE + 1
    if n <= 0:
        return np.empty((0, WINDOW_SIZE), dtype=np.float32)
    idx = np.arange(n)[:, None] * HOP_SIZE + np.arange(WINDOW_SIZE)
    return audio[idx]


def window_rms(windows: np.ndarray) -> np.ndarray:
    return np.sqrt((windows ** 2).mean(axis=1))

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with INDEX_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    all_guitar = []
    all_bass   = []
    meta_rows  = []

    for row in rows:
        g_path = GUITAR_DIR / row["filename_guitar"]
        b_path = BASS_DIR   / row["filename_bass"]

        if not g_path.exists() or not b_path.exists():
            print(f"  [skip] missing file: {row['filename_guitar']}")
            continue

        g_audio, g_sr = load_mono(g_path)
        b_audio, b_sr = load_mono(b_path)

        g_audio = resample(g_audio, g_sr, TARGET_SR)
        b_audio = resample(b_audio, b_sr, TARGET_SR)

        # Trim to same length (octaver render may differ by a few samples)
        n = min(len(g_audio), len(b_audio))
        g_audio, b_audio = g_audio[:n], b_audio[:n]
        g_audio, b_audio = joint_normalize(g_audio, b_audio)

        g_wins = slice_windows(g_audio)
        b_wins = slice_windows(b_audio)

        # Drop silent windows
        mask   = window_rms(g_wins) >= SILENCE_RMS
        g_wins = g_wins[mask]
        b_wins = b_wins[mask]

        n_wins   = len(g_wins)
        note     = row.get("note", "?")
        kept_pct = 100 * n_wins / max(len(mask), 1)

        all_guitar.append(g_wins)
        all_bass.append(b_wins)
        meta_rows.extend(
            {"source_file": row["filename_guitar"], "note": note, "window_in_file": i}
            for i in range(n_wins)
        )

        src_info = f"(src {g_sr} Hz)" if g_sr != TARGET_SR else ""
        print(f"  [ok]  {note:5s}  {n_wins:6,} windows  ({kept_pct:.0f}% kept) {src_info}")

    if not all_guitar:
        print("No data processed.")
        return

    guitar_arr = np.concatenate(all_guitar, axis=0)
    bass_arr   = np.concatenate(all_bass,   axis=0)

    for i, row in enumerate(meta_rows):
        row["window_idx"] = i

    np.save(OUT_DIR / "guitar.npy", guitar_arr)
    np.save(OUT_DIR / "bass.npy",   bass_arr)

    meta_path = OUT_DIR / "meta.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(meta_rows)

    duration_min = guitar_arr.shape[0] * HOP_SIZE / TARGET_SR / 60
    print(f"\nDone: {guitar_arr.shape[0]:,} windows  "
          f"shape={guitar_arr.shape}  ~{duration_min:.1f} min")
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

"""CLI entry point for pitch detector comparison.

Usage
-----
    python ml/pitch_eval/run_eval.py \\
        --input data/v0/guitar/some_file.wav   # file or directory
        --detectors yin,pesto                  # default: all in REGISTRY
        --output runs/pitch_eval/              # default: runs/pitch_eval/<timestamp>/

Outputs per file (in <output>/<filename_stem>/):
    f0_curves.csv   — long-format table: time_s, detector, f0_hz, confidence
    f0_curves.png   — overlay F0 plot, log-Y axis, guitar note guide lines
    audio.wav       — symlink (Linux) or copy to the source WAV

Aggregate outputs in <output>/:
    summary.csv     — per-pair metrics for every file
    summary.md      — human-readable table + top-5 worst files
"""

import argparse
import csv
import datetime
import logging
import math
import os
import shutil
import sys
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow running as a script from the repo root without installing the package
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.pitch_eval.detectors import REGISTRY
from ml.pitch_eval.compare import (
    align_to_grid,
    compute_signal_activity,
    detector_vs_activity,
    pairwise_disagreement,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Guitar reference notes for guide lines (Hz)
_GUITAR_NOTES = {
    "E2": 82.41,
    "A2": 110.00,
    "D3": 146.83,
    "G3": 196.00,
    "B3": 246.94,
    "E4": 329.63,
}

_GRID_HOP_MS = 10.0  # common grid for aggregate metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_wav_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.wav"))


def save_csv(rows: list[dict], filepath: Path) -> None:
    if not rows:
        return
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(filepath, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_f0_curves(results, audio_path: Path, sr: int, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, res in enumerate(results):
        voiced = res.f0_hz > 0
        # Plot only voiced frames for clarity; scatter with small dots
        if np.any(voiced):
            ax.scatter(
                res.times[voiced],
                res.f0_hz[voiced],
                s=2,
                alpha=0.7,
                label=res.name,
                color=colors[i % len(colors)],
            )

    # Guide lines for guitar open strings
    for note_name, freq in _GUITAR_NOTES.items():
        ax.axhline(freq, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)
        ax.text(0.0, freq * 1.02, note_name, fontsize=7, color="gray", va="bottom")

    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("F0 (Hz)")
    ax.set_title(f"{audio_path.name}  |  sr={sr} Hz")
    ax.legend(loc="upper right", markerscale=4)

    # Y-axis ticks at note frequencies
    note_freqs = list(_GUITAR_NOTES.values())
    ax.set_yticks(note_freqs)
    ax.set_yticklabels([f"{f:.0f}" for f in note_freqs])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    plt.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(
    wav_path: Path, detectors, out_dir: Path
) -> tuple[list[dict], list[dict]]:
    """Run all detectors on one WAV, save outputs, return (summary_rows, per_detector_rows)."""
    log.info("Processing %s", wav_path.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = librosa.load(str(wav_path), sr=44100, mono=True)
    duration_s = len(audio) / sr

    activity_times, activity_active = compute_signal_activity(audio, sr)

    results = []
    for det in detectors:
        log.info("  Running %s ...", det.name)
        result = det.estimate(audio, sr)
        results.append(result)

    # --- f0_curves.csv (long format) ---
    csv_rows = []
    for res in results:
        for t, f0, conf in zip(res.times, res.f0_hz, res.confidence):
            # Nearest-neighbor lookup into activity_times
            idx = int(np.searchsorted(activity_times, float(t)))
            idx = min(idx, len(activity_active) - 1)
            sig_active = int(activity_active[idx])
            csv_rows.append({
                "time_s": f"{t:.6f}",
                "detector": res.name,
                "f0_hz": f"{f0:.4f}",
                "confidence": f"{conf:.4f}",
                "signal_active": sig_active,
            })
    save_csv(csv_rows, out_dir / "f0_curves.csv")

    # --- f0_curves.png ---
    plot_f0_curves(results, wav_path, sr, out_dir / "f0_curves.png")

    # --- audio.wav symlink ---
    audio_link = out_dir / "audio.wav"
    if audio_link.exists() or audio_link.is_symlink():
        audio_link.unlink()
    try:
        audio_link.symlink_to(wav_path.resolve())
    except OSError:
        shutil.copy2(wav_path, audio_link)

    # Build the common np.arange grid (same formula as align_to_grid uses)
    hop_s = _GRID_HOP_MS / 1000.0
    common_grid = np.arange(0.0, duration_s + hop_s * 0.5, hop_s)

    # Map activity onto the common grid via nearest-neighbor lookup in activity_times
    grid_activity_idx = np.searchsorted(activity_times, common_grid, side="left").clip(
        0, len(activity_active) - 1
    )
    signal_active_grid = activity_active[grid_activity_idx]

    # Build f0 grids for all detectors (reused for both pairwise and per-detector)
    grids = {
        res.name: align_to_grid(res, hop_ms=_GRID_HOP_MS, duration_s=duration_s)
        for res in results
    }

    # --- pairwise metrics for summary ---
    summary_rows = []
    if len(results) >= 2:
        det_names = list(grids.keys())
        for i in range(len(det_names)):
            for j in range(i + 1, len(det_names)):
                na, nb = det_names[i], det_names[j]
                metrics = pairwise_disagreement(grids[na], grids[nb])
                summary_rows.append({
                    "file": wav_path.stem,
                    "detector_a": na,
                    "detector_b": nb,
                    **{k: str(v) for k, v in metrics.items()},
                })

    # --- per-detector vs signal activity ---
    per_detector_rows = []
    for res in results:
        metrics = detector_vs_activity(grids[res.name], signal_active_grid)
        per_detector_rows.append({
            "file": wav_path.stem,
            "detector": res.name,
            **{k: str(v) for k, v in metrics.items()},
        })

    return summary_rows, per_detector_rows


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

def write_summary_per_detector(all_rows: list[dict], out_dir: Path) -> None:
    """Write summary_per_detector.csv from per-detector activity metrics."""
    save_csv(all_rows, out_dir / "summary_per_detector.csv")
    log.info("summary_per_detector.csv written to %s", out_dir / "summary_per_detector.csv")


def write_summary(
    all_rows: list[dict],
    out_dir: Path,
    per_detector_rows: list[dict] = None,
) -> None:
    save_csv(all_rows, out_dir / "summary.csv")

    lines = ["# Pitch Eval — Aggregate Summary\n"]

    # ------------------------------------------------------------------ #
    # Section 1: pairwise metrics (yin vs pesto, etc.)
    # ------------------------------------------------------------------ #
    if not all_rows:
        lines.append("\nNo pairwise metrics computed.\n")
    else:
        # Group by (detector_a, detector_b)
        pairs: dict[tuple, list[dict]] = {}
        for row in all_rows:
            key = (row["detector_a"], row["detector_b"])
            pairs.setdefault(key, []).append(row)

        for (da, db), rows in pairs.items():
            lines.append(f"\n## {da} vs {db}\n")

            def _collect(field, _rows=rows):
                vals = []
                for r in _rows:
                    try:
                        v = float(r[field])
                        if not math.isnan(v):
                            vals.append(v)
                    except (ValueError, TypeError):
                        pass
                return vals

            mean_cents = _collect("mean_abs_cents")
            med_cents  = _collect("median_abs_cents")
            pct50      = _collect("pct_disagree_50c")
            frac_both  = _collect("frac_both_voiced")

            def _fmt_stat(vals, decimals=3):
                if not vals:
                    return "n/a"
                fmt = f".{decimals}f"
                return f"{np.mean(vals):{fmt}} (median {np.median(vals):{fmt}})"

            lines.append(f"- Files evaluated: {len(rows)}\n")
            lines.append(f"- Both-voiced fraction: {_fmt_stat(frac_both)}\n")
            lines.append(f"- Mean |deviation|, cents: {_fmt_stat(mean_cents, decimals=1)}\n")
            lines.append(f"- Median |deviation|, cents: {_fmt_stat(med_cents, decimals=1)}\n")
            lines.append(f"- % frames > 50 cents: {_fmt_stat(pct50, decimals=1)}\n")

            # Top-5 worst files by mean_abs_cents
            ranked = sorted(
                rows,
                key=lambda r: float(r["mean_abs_cents"])
                if r["mean_abs_cents"] not in ("nan", "")
                else -1,
                reverse=True,
            )[:5]
            if ranked:
                lines.append("\n### Top-5 files by mean deviation\n\n")
                lines.append("| file | mean_abs_cents | pct_disagree_50c |\n")
                lines.append("|------|---------------|------------------|\n")
                for r in ranked:
                    lines.append(
                        f"| {r['file']} | {r['mean_abs_cents']} | {r['pct_disagree_50c']} |\n"
                    )

    # ------------------------------------------------------------------ #
    # Section 2: per-detector vs signal activity
    # ------------------------------------------------------------------ #
    if per_detector_rows:
        lines.append(
            '\n## Detector vs Signal Activity'
            ' (recall = "ловит ли ноту", precision = "не выдумывает ли")\n\n'
        )

        # Group rows by detector name
        by_detector: dict[str, list[dict]] = {}
        for row in per_detector_rows:
            by_detector.setdefault(row["detector"], []).append(row)

        def _mean_metric(rows_list, field):
            vals = []
            for r in rows_list:
                try:
                    v = float(r[field])
                    if not math.isnan(v):
                        vals.append(v)
                except (ValueError, TypeError):
                    pass
            return float(np.mean(vals)) if vals else float("nan")

        def _fmt_v(v):
            return f"{v:.3f}" if not math.isnan(v) else "n/a"

        lines.append("| detector | mean recall | mean precision | mean F1 |\n")
        lines.append("|----------|-------------|----------------|--------|\n")
        for det_name, det_rows in by_detector.items():
            mr = _mean_metric(det_rows, "recall")
            mp = _mean_metric(det_rows, "precision")
            mf = _mean_metric(det_rows, "f1")
            lines.append(f"| {det_name} | {_fmt_v(mr)} | {_fmt_v(mp)} | {_fmt_v(mf)} |\n")

        # Top-5 worst recall per detector
        for det_name, det_rows in by_detector.items():
            ranked = sorted(
                det_rows,
                key=lambda r: float(r["recall"]) if r["recall"] not in ("nan", "") else 2.0,
            )[:5]
            lines.append(f"\n### Top-5 worst recall — {det_name}\n\n")
            lines.append("| file | recall | precision | f1 |\n")
            lines.append("|------|--------|-----------|----|\n")
            for r in ranked:
                lines.append(
                    f"| {r['file']} | {r['recall']} | {r['precision']} | {r['f1']} |\n"
                )

    (out_dir / "summary.md").write_text("".join(lines))
    log.info("summary.md written to %s", out_dir / "summary.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare pitch detectors on guitar WAV files."
    )
    parser.add_argument(
        "--input", required=True, help="WAV file or directory (recursive)"
    )
    parser.add_argument(
        "--detectors",
        default=",".join(REGISTRY.keys()),
        help=f"Comma-separated detector names. Available: {list(REGISTRY.keys())}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output root directory. Default: runs/pitch_eval/<timestamp>/",
    )
    args = parser.parse_args()

    # Resolve output dir
    if args.output is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = _REPO_ROOT / "runs" / "pitch_eval" / ts
    else:
        out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_root)

    # Instantiate detectors
    requested = [d.strip() for d in args.detectors.split(",") if d.strip()]
    unknown = [d for d in requested if d not in REGISTRY]
    if unknown:
        parser.error(f"Unknown detectors: {unknown}. Available: {list(REGISTRY.keys())}")
    detectors = [REGISTRY[name]() for name in requested]
    log.info("Detectors: %s", requested)

    # Collect files
    wav_files = collect_wav_files(Path(args.input))
    if not wav_files:
        log.error("No WAV files found at %s", args.input)
        sys.exit(1)
    log.info("Found %d WAV file(s)", len(wav_files))

    # Process
    all_summary_rows: list[dict] = []
    all_per_detector_rows: list[dict] = []
    for wav_path in wav_files:
        file_out_dir = out_root / wav_path.stem
        summary_rows, per_detector_rows = process_file(wav_path, detectors, file_out_dir)
        all_summary_rows.extend(summary_rows)
        all_per_detector_rows.extend(per_detector_rows)

    # Aggregate summary
    write_summary_per_detector(all_per_detector_rows, out_root)
    write_summary(all_summary_rows, out_root, per_detector_rows=all_per_detector_rows)
    log.info("Done. Results in %s", out_root)


if __name__ == "__main__":
    main()

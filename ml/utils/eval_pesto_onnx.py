"""Evaluate stateful streaming PESTO ONNX models on note-labelled guitar WAVs.

The expected pitch is read from each filename prefix, for example E2_*.wav or
A#3_*.wav. Metrics are computed on active-signal frames after a short streaming
cache warmup. Each ONNX model is evaluated with its own metadata (chunk/cache).
"""
import argparse
import csv
import gzip
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = (
    ROOT / "models" / "base" / "pesto.onnx",
    ROOT / "models" / "20260706_110636" / "pesto.onnx",
    ROOT / "models" / "20260707_112408" / "pesto.onnx",
)
NOTE_RE = re.compile(r"^([A-G])(#?)(-?\d+)(?:_|$)")
SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
PITCH_JUMP_THRESHOLD_CENTS = 50.0
CONFIDENCE_SWEEP_THRESHOLDS = tuple(index / 10.0 for index in range(1, 10))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("eval_pesto_onnx")


@dataclass(frozen=True)
class MetricSamples:
    cents: np.ndarray
    pitch_deltas: np.ndarray
    settling_times_ms: np.ndarray
    stable_run_durations_ms: np.ndarray


def note_frequency(path: Path) -> tuple[str, float]:
    match = NOTE_RE.match(path.stem)
    if match is None:
        raise ValueError(f"Cannot read note from filename: {path.name}")
    letter, accidental, octave_text = match.groups()
    midi = 12 * (int(octave_text) + 1) + SEMITONES[letter] + (1 if accidental else 0)
    return f"{letter}{accidental}{octave_text}", 440.0 * 2 ** ((midi - 69) / 12)


def activity_mask(audio: np.ndarray, chunk_size: int,
                  margin_db: float) -> tuple[np.ndarray, np.ndarray, float]:
    n_chunks = audio.size // chunk_size
    chunks = audio[:n_chunks * chunk_size].reshape(n_chunks, chunk_size)
    rms = np.sqrt(np.mean(np.square(chunks, dtype=np.float64), axis=1))
    rms_db = 20.0 * np.log10(rms + 1e-10)
    threshold_db = float(np.percentile(rms_db, 10.0) + margin_db)
    return rms_db > threshold_db, rms_db, threshold_db


def run_streaming(session: ort.InferenceSession, audio: np.ndarray,
                  chunk_size: int, cache_size: int) -> tuple[np.ndarray, np.ndarray, float]:
    n_chunks = audio.size // chunk_size
    cache = np.zeros((1, cache_size), dtype=np.float32)
    f0 = np.empty(n_chunks, dtype=np.float32)
    confidence = np.empty(n_chunks, dtype=np.float32)
    started = time.perf_counter()

    for index in range(n_chunks):
        chunk = audio[index * chunk_size:(index + 1) * chunk_size].reshape(1, -1)
        outputs = session.run(None, {"audio": chunk, "cache": cache})
        f0[index] = outputs[0].reshape(-1)[0]
        confidence[index] = outputs[1].reshape(-1)[0]
        cache = outputs[4]

    return f0, confidence, time.perf_counter() - started


def run_lengths(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return ends - starts


def find_onsets(rms_active: np.ndarray, warmup_frames: int,
                minimum_gap_frames: int) -> np.ndarray:
    previous_inactive = np.concatenate(([True], ~rms_active[:-1]))
    candidates = np.flatnonzero(rms_active & previous_inactive)
    onsets = []
    for onset in candidates:
        if onset < warmup_frames:
            continue
        gap_start = max(0, onset - minimum_gap_frames)
        if not rms_active[gap_start:onset].any():
            onsets.append(onset)
    return np.asarray(onsets, dtype=np.int64)


def stable_run_lengths(f0: np.ndarray, measured: np.ndarray) -> np.ndarray:
    lengths = []
    current_length = 0
    previous_index = -2
    for index in np.flatnonzero(measured):
        continues = (
            index == previous_index + 1
            and abs(1200.0 * np.log2(f0[index] / f0[previous_index]))
            <= PITCH_JUMP_THRESHOLD_CENTS
        )
        if continues:
            current_length += 1
        else:
            if current_length:
                lengths.append(current_length)
            current_length = 1
        previous_index = index
    if current_length:
        lengths.append(current_length)
    return np.asarray(lengths, dtype=np.int64)


def compute_metrics(f0: np.ndarray, confidence: np.ndarray, rms_active: np.ndarray,
                    expected_hz: float, confidence_threshold: float,
                    warmup_frames: int, frame_duration_s: float,
                    attack_window_ms: float, settling_tolerance_cents: float,
                    settling_hold_ms: float,
                    minimum_onset_gap_ms: float) -> tuple[dict, MetricSamples]:
    frame_index = np.arange(f0.size)
    eligible = frame_index >= warmup_frames
    active = rms_active & eligible
    inactive = ~rms_active & eligible
    voiced = np.isfinite(f0) & (f0 > 0) & (confidence >= confidence_threshold)
    measured = active & voiced

    frame_cents = np.full(f0.size, np.nan, dtype=np.float64)
    valid_f0 = np.isfinite(f0) & (f0 > 0)
    frame_cents[valid_f0] = 1200.0 * np.log2(f0[valid_f0] / expected_hz)
    cents = frame_cents[measured]
    abs_cents = np.abs(cents)

    pitch_transition_mask = measured[1:] & measured[:-1]
    pitch_delta_cents = 1200.0 * np.log2(
        f0[1:][pitch_transition_mask] / f0[:-1][pitch_transition_mask]
    )
    abs_pitch_delta_cents = np.abs(pitch_delta_cents)
    active_transition_mask = active[1:] & active[:-1]
    voiced_toggle_mask = active_transition_mask & (voiced[1:] != voiced[:-1])

    frame_duration_ms = 1000.0 * frame_duration_s
    minimum_gap_frames = max(1, int(np.ceil(minimum_onset_gap_ms / frame_duration_ms)))
    attack_window_frames = max(1, int(np.ceil(attack_window_ms / frame_duration_ms)))
    settling_hold_frames = max(1, int(np.ceil(settling_hold_ms / frame_duration_ms)))
    onsets = find_onsets(rms_active, warmup_frames, minimum_gap_frames)

    stable_pitch = measured & (np.abs(frame_cents) <= settling_tolerance_cents)
    settling_times = []
    attack_mask = np.zeros(f0.size, dtype=bool)
    for onset in onsets:
        end = min(f0.size, onset + attack_window_frames)
        attack_mask[onset:end] = True
        for candidate in range(onset, end - settling_hold_frames + 1):
            if stable_pitch[candidate:candidate + settling_hold_frames].all():
                settling_times.append((candidate - onset) * frame_duration_ms)
                break
    settling_times_ms = np.asarray(settling_times, dtype=np.float64)

    attack_active = attack_mask & active
    attack_measured = attack_active & voiced
    attack_abs_cents = np.abs(frame_cents[attack_measured])
    dropout_lengths = run_lengths(active & ~voiced)
    stable_run_durations_ms = (
        stable_run_lengths(f0, measured).astype(np.float64) * frame_duration_ms
    )

    n_active = int(active.sum())
    n_inactive = int(inactive.sum())
    n_measured = int(measured.sum())
    n_voiced = int((voiced & eligible).sum())
    n_attack_active = int(attack_active.sum())
    n_attack_measured = int(attack_measured.sum())
    n_attacks = int(onsets.size)

    def percent(mask: np.ndarray) -> float:
        return float(100.0 * mask.mean()) if mask.size else float("nan")

    row = {
        "frames": int(eligible.sum()),
        "active_frames": n_active,
        "inactive_frames": n_inactive,
        "voiced_frames": n_voiced,
        "active_voiced_frames": n_measured,
        "false_voiced_frames": int((voiced & inactive).sum()),
        "voiced_recall_pct": 100.0 * n_measured / n_active if n_active else float("nan"),
        "false_voiced_pct": 100.0 * int((voiced & inactive).sum()) / n_inactive
                            if n_inactive else float("nan"),
        "mean_abs_cents": float(abs_cents.mean()) if abs_cents.size else float("nan"),
        "rmse_cents": float(np.sqrt(np.mean(np.square(cents))))
                      if cents.size else float("nan"),
        "std_cents": float(np.std(cents)) if cents.size else float("nan"),
        "median_abs_cents": float(np.median(abs_cents)) if abs_cents.size else float("nan"),
        "p95_abs_cents": float(np.percentile(abs_cents, 95)) if abs_cents.size else float("nan"),
        "bias_cents": float(cents.mean()) if cents.size else float("nan"),
        "within_20c_pct": percent(abs_cents <= 20.0),
        "within_50c_pct": percent(abs_cents <= 50.0),
        "error_gt_50c_pct": percent(abs_cents > 50.0),
        "octave_error_pct": percent(abs_cents > 600.0),
        "mean_active_confidence": float(confidence[active].mean()) if n_active else float("nan"),
        "pitch_transition_frames": int(pitch_transition_mask.sum()),
        "median_abs_pitch_delta_cents": float(np.median(abs_pitch_delta_cents))
                                        if abs_pitch_delta_cents.size else float("nan"),
        "p95_abs_pitch_delta_cents": float(np.percentile(abs_pitch_delta_cents, 95))
                                     if abs_pitch_delta_cents.size else float("nan"),
        "pitch_jump_gt_50c_frames": int(
            (abs_pitch_delta_cents > PITCH_JUMP_THRESHOLD_CENTS).sum()
        ),
        "pitch_jump_gt_50c_pct": percent(
            abs_pitch_delta_cents > PITCH_JUMP_THRESHOLD_CENTS
        ),
        "active_transition_frames": int(active_transition_mask.sum()),
        "voiced_toggle_frames": int(voiced_toggle_mask.sum()),
        "voiced_toggle_pct": (
            100.0 * voiced_toggle_mask.sum() / active_transition_mask.sum()
            if active_transition_mask.any() else float("nan")
        ),
        "attack_count": n_attacks,
        "settled_attack_count": int(settling_times_ms.size),
        "attack_unsettled_pct": (
            100.0 * (n_attacks - settling_times_ms.size) / n_attacks
            if n_attacks else float("nan")
        ),
        "attack_settling_median_ms": float(np.median(settling_times_ms))
                                     if settling_times_ms.size else float("nan"),
        "attack_settling_p95_ms": float(np.percentile(settling_times_ms, 95))
                                  if settling_times_ms.size else float("nan"),
        "attack_active_frames": n_attack_active,
        "attack_active_voiced_frames": n_attack_measured,
        "attack_voiced_recall_pct": (
            100.0 * n_attack_measured / n_attack_active
            if n_attack_active else float("nan")
        ),
        "attack_octave_error_frames": int((attack_abs_cents > 600.0).sum()),
        "attack_octave_error_pct": percent(attack_abs_cents > 600.0),
        "longest_voiced_dropout_ms": (
            float(dropout_lengths.max() * frame_duration_ms)
            if dropout_lengths.size else 0.0
        ),
        "stable_run_count": int(stable_run_durations_ms.size),
        "mean_stable_run_ms": float(stable_run_durations_ms.mean())
                              if stable_run_durations_ms.size else float("nan"),
        "p95_stable_run_ms": float(np.percentile(stable_run_durations_ms, 95))
                             if stable_run_durations_ms.size else float("nan"),
    }
    return row, MetricSamples(
        cents=cents,
        pitch_deltas=pitch_delta_cents,
        settling_times_ms=settling_times_ms,
        stable_run_durations_ms=stable_run_durations_ms,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_frame_rows(writer: csv.DictWriter, model: str, wav_path: Path, note: str,
                     expected_hz: float, f0: np.ndarray, confidence: np.ndarray,
                     rms_db: np.ndarray, rms_threshold_db: float,
                     rms_active: np.ndarray, confidence_threshold: float,
                     warmup_frames: int, sample_rate: int, chunk_size: int) -> None:
    frame_index = np.arange(f0.size)
    eligible = frame_index >= warmup_frames
    voiced = np.isfinite(f0) & (f0 > 0) & (confidence >= confidence_threshold)
    active_voiced = rms_active & eligible & voiced
    valid_f0 = np.isfinite(f0) & (f0 > 0)
    error_cents = np.full(f0.size, np.nan, dtype=np.float64)
    error_cents[valid_f0] = 1200.0 * np.log2(f0[valid_f0] / expected_hz)
    frame_duration_s = chunk_size / sample_rate

    writer.writerows({
        "model": model,
        "file": wav_path.name,
        "note": note,
        "frame_index": int(index),
        "time_s": index * frame_duration_s,
        "frame_duration_s": frame_duration_s,
        "expected_hz": expected_hz,
        "predicted_hz": float(f0[index]),
        "confidence": float(confidence[index]),
        "rms_db": float(rms_db[index]),
        "rms_threshold_db": rms_threshold_db,
        "rms_active": int(rms_active[index]),
        "voiced": int(voiced[index]),
        "eligible": int(eligible[index]),
        "active_voiced": int(active_voiced[index]),
        "error_cents": float(error_cents[index]),
    } for index in range(f0.size))


def aggregate(model: str, rows: list[dict],
              sample_parts: list[MetricSamples]) -> dict:
    def concatenate(attribute: str) -> np.ndarray:
        parts = [getattr(samples, attribute) for samples in sample_parts]
        return np.concatenate(parts) if parts else np.empty(0)

    cents = concatenate("cents")
    abs_cents = np.abs(cents)
    pitch_deltas = concatenate("pitch_deltas")
    abs_pitch_deltas = np.abs(pitch_deltas)
    settling_times = concatenate("settling_times_ms")
    stable_run_durations = concatenate("stable_run_durations_ms")
    active = sum(int(row["active_frames"]) for row in rows)
    inactive = sum(int(row["inactive_frames"]) for row in rows)
    active_voiced = sum(int(row["active_voiced_frames"]) for row in rows)
    false_voiced = sum(int(row["false_voiced_frames"]) for row in rows)
    pitch_transitions = sum(int(row["pitch_transition_frames"]) for row in rows)
    pitch_jumps = sum(int(row["pitch_jump_gt_50c_frames"]) for row in rows)
    active_transitions = sum(int(row["active_transition_frames"]) for row in rows)
    voiced_toggles = sum(int(row["voiced_toggle_frames"]) for row in rows)
    attacks = sum(int(row["attack_count"]) for row in rows)
    settled_attacks = sum(int(row["settled_attack_count"]) for row in rows)
    attack_active = sum(int(row["attack_active_frames"]) for row in rows)
    attack_measured = sum(int(row["attack_active_voiced_frames"]) for row in rows)
    attack_octave_errors = sum(int(row["attack_octave_error_frames"]) for row in rows)

    def percent(mask: np.ndarray) -> float:
        return float(100.0 * mask.mean()) if mask.size else float("nan")

    return {
        "model": model,
        "files": len(rows),
        "active_frames": active,
        "active_voiced_frames": active_voiced,
        "voiced_recall_pct": 100.0 * active_voiced / active if active else float("nan"),
        "false_voiced_pct": 100.0 * false_voiced / inactive if inactive else float("nan"),
        "mean_abs_cents": float(abs_cents.mean()) if abs_cents.size else float("nan"),
        "rmse_cents": float(np.sqrt(np.mean(np.square(cents))))
                      if cents.size else float("nan"),
        "std_cents": float(np.std(cents)) if cents.size else float("nan"),
        "median_abs_cents": float(np.median(abs_cents)) if abs_cents.size else float("nan"),
        "p95_abs_cents": float(np.percentile(abs_cents, 95)) if abs_cents.size else float("nan"),
        "bias_cents": float(cents.mean()) if cents.size else float("nan"),
        "within_20c_pct": percent(abs_cents <= 20.0),
        "within_50c_pct": percent(abs_cents <= 50.0),
        "error_gt_50c_pct": percent(abs_cents > 50.0),
        "octave_error_pct": percent(abs_cents > 600.0),
        "pitch_transition_frames": pitch_transitions,
        "median_abs_pitch_delta_cents": float(np.median(abs_pitch_deltas))
                                        if abs_pitch_deltas.size else float("nan"),
        "p95_abs_pitch_delta_cents": float(np.percentile(abs_pitch_deltas, 95))
                                     if abs_pitch_deltas.size else float("nan"),
        "pitch_jump_gt_50c_frames": pitch_jumps,
        "pitch_jump_gt_50c_pct": 100.0 * pitch_jumps / pitch_transitions
                                 if pitch_transitions else float("nan"),
        "active_transition_frames": active_transitions,
        "voiced_toggle_frames": voiced_toggles,
        "voiced_toggle_pct": 100.0 * voiced_toggles / active_transitions
                             if active_transitions else float("nan"),
        "attack_count": attacks,
        "settled_attack_count": settled_attacks,
        "attack_unsettled_pct": 100.0 * (attacks - settled_attacks) / attacks
                                if attacks else float("nan"),
        "attack_settling_median_ms": float(np.median(settling_times))
                                     if settling_times.size else float("nan"),
        "attack_settling_p95_ms": float(np.percentile(settling_times, 95))
                                  if settling_times.size else float("nan"),
        "attack_active_frames": attack_active,
        "attack_active_voiced_frames": attack_measured,
        "attack_voiced_recall_pct": 100.0 * attack_measured / attack_active
                                    if attack_active else float("nan"),
        "attack_octave_error_frames": attack_octave_errors,
        "attack_octave_error_pct": 100.0 * attack_octave_errors / attack_measured
                                   if attack_measured else float("nan"),
        "longest_voiced_dropout_ms": max(
            (float(row["longest_voiced_dropout_ms"]) for row in rows), default=0.0
        ),
        "stable_run_count": int(stable_run_durations.size),
        "mean_stable_run_ms": float(stable_run_durations.mean())
                              if stable_run_durations.size else float("nan"),
        "p95_stable_run_ms": float(np.percentile(stable_run_durations, 95))
                             if stable_run_durations.size else float("nan"),
        "inference_seconds": sum(float(row["inference_seconds"]) for row in rows),
    }


def update_confidence_sweep(stats: dict[float, dict[str, int]], f0: np.ndarray,
                            confidence: np.ndarray, rms_active: np.ndarray,
                            warmup_frames: int) -> None:
    eligible = np.arange(f0.size) >= warmup_frames
    active = rms_active & eligible
    inactive = ~rms_active & eligible
    valid_f0 = np.isfinite(f0) & (f0 > 0)
    for threshold, counts in stats.items():
        voiced = valid_f0 & (confidence >= threshold)
        counts["active"] += int(active.sum())
        counts["inactive"] += int(inactive.sum())
        counts["active_voiced"] += int((active & voiced).sum())
        counts["false_voiced"] += int((inactive & voiced).sum())


def confidence_sweep_rows(model: str,
                          stats: dict[float, dict[str, int]]) -> list[dict]:
    rows = []
    for threshold, counts in stats.items():
        true_positive = counts["active_voiced"]
        false_positive = counts["false_voiced"]
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else float("nan")
        )
        recall = true_positive / counts["active"] if counts["active"] else float("nan")
        rows.append({
            "model": model,
            "confidence_threshold": threshold,
            "voiced_recall_pct": 100.0 * recall,
            "false_voiced_pct": 100.0 * false_positive / counts["inactive"]
                                if counts["inactive"] else float("nan"),
            "voiced_precision_pct": 100.0 * precision,
            "voiced_f1_pct": 100.0 * 2.0 * precision * recall / (precision + recall)
                             if precision + recall else float("nan"),
            **counts,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", type=Path, default=list(DEFAULT_MODELS))
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "v0" / "guitar")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--activity-margin-db", type=float, default=10.0)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--attack-window-ms", type=float, default=200.0)
    parser.add_argument("--settling-tolerance-cents", type=float, default=50.0)
    parser.add_argument("--settling-hold-ms", type=float, default=50.0)
    parser.add_argument("--minimum-onset-gap-ms", type=float, default=100.0)
    args = parser.parse_args()

    output = args.output or (
        ROOT / "runs" / "pitch_eval" / f"onnx_models_{datetime.now():%Y%m%d_%H%M%S}"
    )
    wav_paths = sorted(args.input.glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No WAV files found in {args.input}")

    output.mkdir(parents=True, exist_ok=True)
    frame_columns = (
        "model", "file", "note", "frame_index", "time_s", "frame_duration_s",
        "expected_hz", "predicted_hz", "confidence", "rms_db", "rms_threshold_db",
        "rms_active", "voiced", "eligible", "active_voiced", "error_cents",
    )
    per_file_rows: list[dict] = []
    summary_rows: list[dict] = []
    sweep_rows: list[dict] = []
    frame_stream = gzip.open(output / "frames.csv.gz", "wt", newline="")
    frame_writer = csv.DictWriter(frame_stream, fieldnames=frame_columns)
    frame_writer.writeheader()

    for model_path in args.models:
        metadata_path = model_path.with_name("pesto_onnx_meta.json")
        if not model_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Missing model or metadata: {model_path}")
        metadata = json.loads(metadata_path.read_text())
        sample_rate = int(metadata["sample_rate"])
        chunk_size = int(metadata["chunk_size"])
        cache_size = int(metadata["cache_size"])
        model_name = model_path.parent.name
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        model_rows: list[dict] = []
        sample_parts: list[MetricSamples] = []
        sweep_stats = {
            threshold: {"active": 0, "inactive": 0, "active_voiced": 0, "false_voiced": 0}
            for threshold in CONFIDENCE_SWEEP_THRESHOLDS
        }

        log.info("Evaluating %s (%s)", model_name, model_path)
        for wav_path in wav_paths:
            note, expected_hz = note_frequency(wav_path)
            audio, wav_sr = sf.read(wav_path, dtype="float32", always_2d=False)
            if wav_sr != sample_rate:
                raise ValueError(f"{wav_path.name}: sr={wav_sr}, model expects {sample_rate}")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = np.asarray(audio, dtype=np.float32)
            active, rms_db, rms_threshold_db = activity_mask(
                audio, chunk_size, args.activity_margin_db
            )
            f0, confidence, elapsed = run_streaming(
                session, audio, chunk_size=chunk_size, cache_size=cache_size
            )
            write_frame_rows(
                frame_writer, model_name, wav_path, note, expected_hz, f0, confidence,
                rms_db, rms_threshold_db, active, args.confidence_threshold,
                args.warmup_frames, sample_rate, chunk_size,
            )
            metrics, samples = compute_metrics(
                f0, confidence, active, expected_hz,
                confidence_threshold=args.confidence_threshold,
                warmup_frames=args.warmup_frames,
                frame_duration_s=chunk_size / sample_rate,
                attack_window_ms=args.attack_window_ms,
                settling_tolerance_cents=args.settling_tolerance_cents,
                settling_hold_ms=args.settling_hold_ms,
                minimum_onset_gap_ms=args.minimum_onset_gap_ms,
            )
            update_confidence_sweep(
                sweep_stats, f0, confidence, active, args.warmup_frames
            )
            row = {
                "model": model_name,
                "file": wav_path.name,
                "note": note,
                "expected_hz": expected_hz,
                **metrics,
                "inference_seconds": elapsed,
            }
            model_rows.append(row)
            per_file_rows.append(row)
            sample_parts.append(samples)
            log.info("  %-24s recall=%5.1f%% median=%7.2fc octave=%5.2f%%",
                     wav_path.name, metrics["voiced_recall_pct"],
                     metrics["median_abs_cents"], metrics["octave_error_pct"])

        summary = aggregate(model_name, model_rows, sample_parts)
        summary.update({
            "mirror": metadata.get("mirror"),
            "mirror_fn": metadata.get("mirror_fn"),
            "checkpoint": metadata.get("model_name"),
        })
        summary_rows.append(summary)
        sweep_rows.extend(confidence_sweep_rows(model_name, sweep_stats))

    frame_stream.close()
    write_csv(output / "per_file.csv", per_file_rows)
    write_csv(output / "summary.csv", summary_rows)
    write_csv(output / "confidence_sweep.csv", sweep_rows)
    (output / "config.json").write_text(json.dumps({
        "models": [str(path) for path in args.models],
        "input": str(args.input),
        "confidence_threshold": args.confidence_threshold,
        "activity_margin_db": args.activity_margin_db,
        "warmup_frames": args.warmup_frames,
        "pitch_jump_threshold_cents": PITCH_JUMP_THRESHOLD_CENTS,
        "attack_window_ms": args.attack_window_ms,
        "settling_tolerance_cents": args.settling_tolerance_cents,
        "settling_hold_ms": args.settling_hold_ms,
        "minimum_onset_gap_ms": args.minimum_onset_gap_ms,
        "confidence_sweep_thresholds": list(CONFIDENCE_SWEEP_THRESHOLDS),
    }, indent=2))
    log.info("Results written to %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

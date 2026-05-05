#!/usr/bin/env python3
"""
Imports raw Reaper dataset files into data/v0/, renames them, and updates index.csv.

Reaper filename format:
  guitar: "{idx}-{note}_{pickup}-{datetime}.wav"          e.g. 01-E2_h-260427_1446.wav
  bass:   "{idx}-{note}_{pickup}-{datetime} render 001.wav"

Output:
  data/v0/guitar/{note}_{datetime}.wav
  data/v0/bass/{note}_{datetime}.wav
  data/v0/index.csv

Usage:
  python utils/import_dataset.py
"""

import re
import csv
import shutil
from pathlib import Path

import soundfile as sf

# ── Paths ─────────────────────────────────────────────────────────────────────

SRC_DIR    = Path(r"D:\Music\Projects\dataset")
DATA_DIR   = Path(__file__).resolve().parent.parent / "data" / "v0"
INDEX_CSV  = DATA_DIR / "index.csv"
GUITAR_DIR = DATA_DIR / "guitar"
BASS_DIR   = DATA_DIR / "bass"

# ── Note → frequency map (guitar range, standard tuning) ─────────────────────

NOTE_FREQS = {
    "C2":  65.41,  "C#2": 69.30,  "D2":  73.42,  "D#2": 77.78,
    "E2":  82.41,  "F2":  87.31,  "F#2": 92.50,  "G2":  98.00,
    "G#2": 103.83, "A2":  110.00, "A#2": 116.54, "B2":  123.47,
    "C3":  130.81, "C#3": 138.59, "D3":  146.83, "D#3": 155.56,
    "E3":  164.81, "F3":  174.61, "F#3": 185.00, "G3":  196.00,
    "G#3": 207.65, "A3":  220.00, "A#3": 233.08, "B3":  246.94,
    "E4":  329.63, "A4":  440.00,
}

PICKUP_MAP = {"h": "humbucker", "s": "single"}

CSV_FIELDS = [
    "filename_guitar", "filename_bass",
    "note", "note_hz",
    "type", "source",
    "freq_cluster", "bpm_cluster", "tempo_bpm",
    "note_range_min_hz", "note_range_max_hz",
    "duration_sec",
    "guitar_slug", "pickup_type", "pickup_active",
    "tuning", "string_gauge", "interface", "gain_db",
    "has_good_attack", "quality_rating",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

# Matches e.g. "01-E2_h-260427_1446" or "03-F#2_h-260427_1450"
_STEM_RE = re.compile(r"^\d+-([A-G][#b]?\d)_([hs])-(\d{6}_\d{4})$")


def parse_guitar_stem(stem: str):
    """Return (note, pickup_type, datetime_str) or None."""
    m = _STEM_RE.match(stem)
    if not m:
        return None
    note, pickup_code, dt = m.group(1), m.group(2), m.group(3)
    return note, PICKUP_MAP.get(pickup_code, pickup_code), dt


def freq_cluster(hz: float) -> str:
    if hz < 150:
        return "low"
    if hz < 400:
        return "mid"
    return "high"


def load_existing_entries(csv_path: Path) -> set:
    """Return set of already-imported guitar filenames to avoid duplicates."""
    if not csv_path.exists():
        return set()
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["filename_guitar"] for row in reader}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    GUITAR_DIR.mkdir(parents=True, exist_ok=True)
    BASS_DIR.mkdir(parents=True, exist_ok=True)

    already_imported = load_existing_entries(INDEX_CSV)
    write_header = not INDEX_CSV.exists() or INDEX_CSV.stat().st_size == 0

    wav_files = list(SRC_DIR.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files found in {SRC_DIR}")
        return

    # Split into guitar / bass by presence of " render " in stem
    guitar_stems: dict[str, Path] = {}
    bass_stems:   dict[str, Path] = {}

    for f in wav_files:
        if " render " in f.stem:
            base = f.stem.split(" render ")[0].strip()
            bass_stems[base] = f
        else:
            guitar_stems[f.stem] = f

    matched = skipped = duplicates = 0

    with INDEX_CSV.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for base_stem, guitar_path in sorted(guitar_stems.items()):
            parsed = parse_guitar_stem(base_stem)
            if not parsed:
                print(f"  [skip] unrecognised filename: {base_stem}")
                skipped += 1
                continue

            note, pickup_type, dt = parsed
            new_name = f"{note}_{dt}.wav"

            if new_name in already_imported:
                print(f"  [dup]  already in index, skipping: {new_name}")
                duplicates += 1
                continue

            if base_stem not in bass_stems:
                print(f"  [skip] no bass pair for: {guitar_path.name}")
                skipped += 1
                continue

            bass_path = bass_stems[base_stem]

            shutil.copy2(guitar_path, GUITAR_DIR / new_name)
            shutil.copy2(bass_path,   BASS_DIR   / new_name)

            info     = sf.info(str(GUITAR_DIR / new_name))
            duration = round(info.frames / info.samplerate, 3)
            note_hz  = NOTE_FREQS.get(note, 0.0)

            writer.writerow({
                "filename_guitar":   new_name,
                "filename_bass":     new_name,
                "note":              note,
                "note_hz":           note_hz,
                "type":              "single_note",
                "source":            "octaver",
                "freq_cluster":      freq_cluster(note_hz),
                "bpm_cluster":       "",
                "tempo_bpm":         "",
                "note_range_min_hz": note_hz,
                "note_range_max_hz": note_hz,
                "duration_sec":      duration,
                "guitar_slug":       "guitar_1",
                "pickup_type":       pickup_type,
                "pickup_active":     "false",
                "tuning":            "standard",
                "string_gauge":      "",
                "interface":         "",
                "gain_db":           "",
                "has_good_attack":   "",
                "quality_rating":    "",
            })

            print(f"  [ok]   {note:5s} -> {new_name}  ({duration:.1f}s)")
            matched += 1

    print(f"\nDone: {matched} pairs imported, {skipped} skipped, {duplicates} duplicates.")
    print(f"Index: {INDEX_CSV}")


if __name__ == "__main__":
    main()

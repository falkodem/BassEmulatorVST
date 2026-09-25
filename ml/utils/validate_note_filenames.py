"""Validate note-labelled WAV filenames against an expected pitch range."""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTE_RE = re.compile(r"^([A-G])(#?)(-?\d+)$")
FILENAME_RE = re.compile(r"^([A-G])(#?)(-?\d+)_.*\.wav$", re.IGNORECASE)
SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


@dataclass(frozen=True)
class InvalidFilename:
    path: Path
    reason: str


def note_to_midi(note: str) -> int:
    match = NOTE_RE.fullmatch(note)
    if match is None:
        raise ValueError(f"Invalid note: {note!r}; expected notation such as E2 or F#3")
    letter, accidental, octave_text = match.groups()
    return 12 * (int(octave_text) + 1) + SEMITONES[letter] + (1 if accidental else 0)


def validate_filename(path: Path, minimum_midi: int,
                      maximum_midi: int) -> InvalidFilename | None:
    match = FILENAME_RE.fullmatch(path.name)
    if match is None:
        return InvalidFilename(path, "expected filename format NOTE_*.wav, for example E2_take.wav")

    letter, accidental, octave_text = match.groups()
    note = f"{letter.upper()}{accidental}{octave_text}"
    midi = note_to_midi(note)
    if midi < minimum_midi or midi > maximum_midi:
        return InvalidFilename(path, f"note {note} is outside the allowed range")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=ROOT / "data" / "v0" / "guitar")
    parser.add_argument("--min-note", default="E2")
    parser.add_argument("--max-note", default="E4")
    parser.add_argument("--recursive", action="store_true", help="Search for WAV files recursively")
    args = parser.parse_args()

    minimum_midi = note_to_midi(args.min_note)
    maximum_midi = note_to_midi(args.max_note)
    if minimum_midi > maximum_midi:
        parser.error("--min-note must not be above --max-note")

    if args.input.is_file():
        paths = [args.input]
    elif args.input.is_dir():
        paths = sorted(args.input.rglob("*.wav") if args.recursive else args.input.glob("*.wav"))
    else:
        parser.error(f"input does not exist: {args.input}")

    invalid = [
        result
        for path in paths
        if (result := validate_filename(path, minimum_midi, maximum_midi)) is not None
    ]
    for result in invalid:
        print(f"INVALID {result.path}: {result.reason} ({args.min_note}-{args.max_note})")

    valid_count = len(paths) - len(invalid)
    print(f"Checked {len(paths)} WAV files: {valid_count} valid, {len(invalid)} invalid")
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())

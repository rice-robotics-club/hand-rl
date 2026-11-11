"""Convert a MIDI, MP3, or WAV file to a JSON list of notes.

Run from the repository root: python -m scripts.midi_to_notes song.mid -o song.notes.json
"""

import argparse
import json
from pathlib import Path

from src.music.midi import midi_to_notes, named_notes


class Args(argparse.Namespace):
    midi_file: Path = Path()
    output: Path | None = None
    no_octave: bool = False
    named: bool = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("midi_file", type=Path, help="MIDI, MP3, or WAV input file")
    _ = parser.add_argument("-o", "--output", type=Path, help="Save JSON to this file")
    _ = parser.add_argument(
        "--named",
        action="store_true",
        help='Output {"note", "timestamp", "duration"} objects instead of '
        + "[start, duration, MIDI pitch] rows",
    )
    _ = parser.add_argument(
        "--no-octave", action="store_true", help="With --named, write A instead of A4"
    )
    args = parser.parse_args(namespace=Args())
    notes = midi_to_notes(args.midi_file)
    result = json.dumps(
        named_notes(notes, include_octave=not args.no_octave) if args.named else notes,
        indent=2,
    )
    if args.output:
        _ = args.output.write_text(result + "\n", encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()

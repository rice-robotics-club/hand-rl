"""Convert a MIDI, MP3, or WAV file to a JSON list of notes.

Run from the repository root:
    python -m scripts.midi_to_notes song.mid -o song.json            # named notes
    python -m scripts.midi_to_notes song.mid --tuples -o song.json   # note rows

Training reads the --tuples form: [start seconds, duration seconds, MIDI pitch].
"""

import argparse
import json
from pathlib import Path

from src.music.midi import midi_to_notes, named_notes


class Args(argparse.Namespace):
    midi_file: Path = Path()
    output: Path | None = None
    no_octave: bool = False
    tuples: bool = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("midi_file", type=Path, help="MIDI, MP3, or WAV input file")
    _ = parser.add_argument("-o", "--output", type=Path, help="Save JSON to this file")
    _ = parser.add_argument(
        "--no-octave",
        action="store_true",
        help="Use A instead of A4 (named output only)",
    )
    _ = parser.add_argument(
        "--tuples",
        action="store_true",
        help="Output (start, duration, MIDI pitch) triples as JSON arrays",
    )
    args = parser.parse_args(namespace=Args())
    notes = midi_to_notes(args.midi_file)
    result = json.dumps(
        notes if args.tuples else named_notes(notes, include_octave=not args.no_octave),
        indent=2,
    )
    if args.output:
        _ = args.output.write_text(result + "\n", encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()

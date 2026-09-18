"""Convert a MIDI file to a chronological JSON list of notes."""

import argparse
import json
from pathlib import Path

import pretty_midi


def midi_to_notes(path, *, include_octave=True):
    """Return pitched notes with timestamps and key-held durations in seconds.

    Combine all non-drum instruments. Simultaneous notes remain separate entries.
    """
    midi = pretty_midi.PrettyMIDI(str(path))
    notes = sorted(
        (note for instrument in midi.instruments if not instrument.is_drum
         for note in instrument.notes),
        key=lambda note: (note.start, note.pitch, note.end),
    )
    pitch_classes = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return [
        {
            "note": (pretty_midi.note_number_to_name(note.pitch)
                     if include_octave else pitch_classes[note.pitch % 12]),
            "timestamp": note.start,
            "duration": note.end - note.start,
        }
        for note in notes
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("midi_file", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="Save JSON to this file")
    parser.add_argument("--no-octave", action="store_true", help="Use A instead of A4")
    args = parser.parse_args()
    result = json.dumps(
        midi_to_notes(args.midi_file, include_octave=not args.no_octave), indent=2
    )
    if args.output:
        args.output.write_text(result + "\n", encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()

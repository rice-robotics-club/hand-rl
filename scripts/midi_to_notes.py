"""Convert a MIDI, MP3, or WAV file to a chronological JSON list of notes."""

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pretty_midi


def load_midi(path):
    """Transcribe MP3/WAV audio to MIDI in memory; load other files as MIDI."""
    path = Path(path)
    if path.suffix.lower() in {".mp3", ".wav"}:
        # Keep transcription progress out of the JSON output on stdout.
        with redirect_stdout(sys.stderr):
            try:
                from basic_pitch.inference import predict
            except ModuleNotFoundError as exc:
                if exc.name != "basic_pitch":
                    raise
                raise ModuleNotFoundError(
                    "MP3/WAV input requires Basic Pitch. Install it with: "
                    "python -m pip install -r requirements-audio.txt "
                    "(use the Python 3.10 audio environment described in README.md)"
                ) from exc
            _, midi, _ = predict(str(path))
        return midi
    return pretty_midi.PrettyMIDI(str(path))


def midi_to_notes(path, *, include_octave=True):
    """Return pitched notes with timestamps and key-held durations in seconds.

    Combine all non-drum instruments. Simultaneous notes remain separate entries.
    """
    pitch_classes = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return [
        {
            "note": (pretty_midi.note_number_to_name(pitch)
                     if include_octave else pitch_classes[pitch % 12]),
            "timestamp": start,
            "duration": duration,
        }
        for start, duration, pitch in midi_to_note_tuples(path)
    ]


def midi_to_note_tuples(path):
    """Return (start_seconds, duration_seconds, MIDI_pitch) tuples.

    Sort by start time, then pitch and end time. Exclude drums and preserve
    repeated and overlapping notes. MIDI pitch is an integer (60 = C4).
    """
    midi = load_midi(path)
    notes = sorted(
        (note for instrument in midi.instruments if not instrument.is_drum
         for note in instrument.notes),
        key=lambda note: (note.start, note.pitch, note.end),
    )
    return [
        # Basic Pitch can return NumPy scalars, which JSON cannot serialize.
        (float(note.start), float(note.end - note.start), int(note.pitch))
        for note in notes
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("midi_file", type=Path, help="MIDI, MP3, or WAV input file")
    parser.add_argument("-o", "--output", type=Path, help="Save JSON to this file")
    parser.add_argument("--no-octave", action="store_true", help="Use A instead of A4 (named output only)")
    parser.add_argument(
        "--tuples", action="store_true",
        help="Output (start, duration, MIDI pitch) triples as JSON arrays",
    )
    args = parser.parse_args()
    notes = (
        midi_to_note_tuples(args.midi_file) if args.tuples
        else midi_to_notes(args.midi_file, include_octave=not args.no_octave)
    )
    result = json.dumps(notes, indent=2)
    if args.output:
        args.output.write_text(result + "\n", encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()

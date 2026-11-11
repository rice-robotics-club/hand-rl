"""Read notes from MIDI files, or from MP3/WAV audio by transcribing it to MIDI."""

import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import TypedDict

import pretty_midi

from src.music.notes import Note

PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


class NamedNote(TypedDict):
    note: str
    """Note name, e.g. "C4", or "C" without the octave."""
    timestamp: float
    duration: float


def load_midi(path: str | Path) -> pretty_midi.PrettyMIDI:
    """Loads a MIDI file, or transcribes an MP3/WAV file to MIDI in memory.

    Transcription needs Basic Pitch (`pip install basic-pitch`), which is not a project
    dependency.
    """
    path = Path(path)
    if path.suffix.lower() not in {".mp3", ".wav"}:
        return pretty_midi.PrettyMIDI(str(path))
    # Keep transcription progress off stdout, which may be carrying JSON output.
    with redirect_stdout(sys.stderr):
        try:
            # Not a project dependency, so the type checker can't see it.
            import basic_pitch.inference  # pyright: ignore[reportMissingImports]
        except ModuleNotFoundError as exc:
            if exc.name != "basic_pitch":
                raise
            raise ModuleNotFoundError(
                "MP3/WAV input needs Basic Pitch: pip install basic-pitch"
            ) from exc
        _, midi, _ = basic_pitch.inference.predict(str(path))
    return midi


def midi_to_notes(path: str | Path) -> list[Note]:
    """Every pitched note in a MIDI (or MP3/WAV) file, as Note tuples.

    Combines all non-drum instruments, keeping repeated and overlapping notes. Sorted
    by start time, then pitch, then end time.
    """
    midi = load_midi(path)
    notes = sorted(
        (
            note
            for instrument in midi.instruments
            if not instrument.is_drum
            for note in instrument.notes
        ),
        key=lambda note: (note.start, note.pitch, note.end),
    )
    # Basic Pitch can return NumPy scalars, which JSON cannot serialize.
    return [
        (float(note.start), float(note.end - note.start), int(note.pitch))
        for note in notes
    ]


def named_notes(notes: list[Note], *, include_octave: bool = True) -> list[NamedNote]:
    """Notes with names ("C4", or "C" without the octave) instead of MIDI pitches."""
    return [
        {
            "note": pretty_midi.note_number_to_name(pitch)
            if include_octave
            else PITCH_CLASSES[pitch % len(PITCH_CLASSES)],
            "timestamp": start,
            "duration": duration,
        }
        for start, duration, pitch in notes
    ]

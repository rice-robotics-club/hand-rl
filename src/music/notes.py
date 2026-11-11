"""The note format shared by scores, converters, and fingering plans.

A note is (start seconds, duration seconds, MIDI pitch); MIDI pitch 60 is C4. A note
sounds from its start up to, but not including, its end.
"""

import json
import math
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import TypeGuard

from src.music.keyboard import MIDI_PITCHES

type Note = tuple[float, float, int]
"""(start seconds, duration seconds, MIDI pitch)."""


def check_notes(
    notes: Sequence[Sequence[object]], pitches: Collection[int] = MIDI_PITCHES
) -> list[Note]:
    """Validates note rows and returns them as Note tuples.

    Raises ValueError naming the first bad note: each needs three values, a finite
    start >= 0, a finite duration > 0, and an integer pitch in `pitches`.
    """
    checked: list[Note] = []
    for index, row in enumerate(notes):
        if len(row) != 3:
            raise ValueError(f"Note {index}: expected (start, duration, MIDI pitch)")
        start, duration, pitch = row
        if not (_is_finite(start) and _is_finite(duration) and _is_finite(pitch)):
            raise ValueError(f"Note {index}: values must be finite numbers")
        if start < 0 or duration <= 0:
            raise ValueError(f"Note {index}: start must be >= 0 and duration > 0")
        if pitch != int(pitch) or int(pitch) not in pitches:
            raise ValueError(
                f"Note {index}: pitch {pitch} is not an allowed MIDI pitch"
            )
        checked.append((float(start), float(duration), int(pitch)))
    return checked


def load_notes(path: str | Path, pitches: Collection[int] = MIDI_PITCHES) -> list[Note]:
    """Reads and validates a JSON list of [start seconds, duration seconds, MIDI pitch]."""
    rows: object = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not all(isinstance(r, list) for r in rows):
        raise ValueError(f"{path} must contain a JSON list of notes")
    return check_notes(rows, pitches)


def _is_finite(x: object) -> TypeGuard[int | float]:
    """Whether x is a finite int or float (not a bool)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

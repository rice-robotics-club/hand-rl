"""Plan timed finger presses for a stationary five-finger right hand.

Each finger stays over one key (its home key) for the whole piece, so a note is
played by the finger over its key. The plan lists key-action targets, not joint
trajectories, and assumes every finger starts raised above its home key.
"""

from collections.abc import Sequence
from typing import Literal, TypedDict

from src.music.fingering import FINGER_NAMES, HOME_PITCHES
from src.music.notes import check_notes


class PlannedNote(TypedDict):
    start: float
    duration: float
    pitch: int
    finger: int
    """1 (thumb) to 5 (pinky)."""


class FingerAction(TypedDict):
    finger: int
    pitch: int


class PlanEvent(TypedDict):
    """Everything that happens at one time: releases first, then presses."""

    timestamp: float
    release: list[FingerAction]
    press: list[FingerAction]


class StartingFinger(TypedDict):
    finger: int
    name: str
    pitch: int
    state: Literal["raised"]


class FingerPlan(TypedDict):
    mode: Literal["stationary_right_hand"]
    time_unit: Literal["seconds"]
    starting_position: list[StartingFinger]
    notes: list[PlannedNote]
    events: list[PlanEvent]
    """Simultaneous actions are grouped; a repeated note can re-press at its release."""


def notes_to_fingers(
    notes: Sequence[Sequence[object]], *, keys: Sequence[int] = HOME_PITCHES
) -> FingerPlan:
    """Plans which finger presses and releases each note, and when.

    `keys` are the home pitches of the thumb through the pinky. Raises ValueError for
    invalid notes, notes outside the hand position, or overlapping notes on one key.
    """
    keys = tuple(keys)
    if len(keys) != len(FINGER_NAMES) or list(keys) != sorted(set(keys)):
        raise ValueError("keys must contain five distinct ascending MIDI pitches")
    checked = check_notes(notes)
    finger_of = {pitch: number for number, pitch in enumerate(keys, start=1)}

    planned: list[PlannedNote] = []
    for index, (start, duration, pitch) in enumerate(checked):
        if pitch not in finger_of:
            raise ValueError(
                f"Note {index}: pitch {pitch} is outside the fixed hand position {keys}"
            )
        planned.append(
            {
                "start": start,
                "duration": duration,
                "pitch": pitch,
                "finger": finger_of[pitch],
            }
        )
    planned.sort(key=lambda n: (n["start"], n["pitch"], n["duration"]))

    held_until: dict[int, float] = {}
    events: dict[float, PlanEvent] = {}
    for note in planned:
        finger, start = note["finger"], note["start"]
        end = start + note["duration"]
        if start < held_until.get(finger, 0):
            raise ValueError(
                f"Pitch {note['pitch']} starts at {start:g}s while finger {finger} is "
                f"still holding it until {held_until[finger]:g}s. Overlapping presses "
                "of the same key cannot be played independently."
            )
        held_until[finger] = end
        action: FingerAction = {"finger": finger, "pitch": note["pitch"]}
        events.setdefault(start, _empty_event(start))["press"].append(action)
        events.setdefault(end, _empty_event(end))["release"].append(action)

    for event in events.values():
        event["release"].sort(key=lambda a: a["finger"])
        event["press"].sort(key=lambda a: a["finger"])
    return {
        "mode": "stationary_right_hand",
        "time_unit": "seconds",
        "starting_position": [
            {"finger": number, "name": name, "pitch": pitch, "state": "raised"}
            for number, (name, pitch) in enumerate(
                zip(FINGER_NAMES, keys, strict=True), start=1
            )
        ],
        "notes": planned,
        "events": [events[t] for t in sorted(events)],
    }


def _empty_event(timestamp: float) -> PlanEvent:
    return {"timestamp": timestamp, "release": [], "press": []}

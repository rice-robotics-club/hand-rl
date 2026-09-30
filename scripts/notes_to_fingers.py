"""Plan timed finger presses for a stationary five-finger right hand."""

import argparse
import json
import math
from pathlib import Path


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
DEFAULT_KEYS = (60, 62, 64, 65, 67)  # C4, D4, E4, F4, G4


def notes_to_fingers(notes, *, keys=DEFAULT_KEYS):
    """Convert (start_seconds, duration_seconds, MIDI_pitch) rows to a plan.

    Finger numbers 1..5 run from thumb to pinky. Each finger stays over one
    configured key. These are key-action targets, not joint trajectories.
    The controller must start with all fingers raised above their assigned keys.
    """
    keys = tuple(keys)
    if (len(keys) != 5 or any(type(p) is not int or not 0 <= p <= 127 for p in keys)
            or any(a >= b for a, b in zip(keys, keys[1:]))):
        raise ValueError("keys must contain five distinct ascending MIDI pitches (0..127).")
    pitch_to_finger = {pitch: index + 1 for index, pitch in enumerate(keys)}
    assignments = []
    for index, row in enumerate(notes):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(f"Note {index}: expected (start, duration, MIDI pitch).")
        start, duration, pitch = row
        if (isinstance(start, bool) or isinstance(duration, bool)
                or not isinstance(start, (int, float))
                or not isinstance(duration, (int, float))
                or not math.isfinite(start) or not math.isfinite(duration)
                or start < 0 or duration <= 0 or not math.isfinite(start + duration)):
            raise ValueError(f"Note {index}: start must be finite and >= 0; duration finite and > 0.")
        if type(pitch) is not int or not 0 <= pitch <= 127:
            raise ValueError(f"Note {index}: MIDI pitch must be an integer from 0 to 127.")
        if pitch not in pitch_to_finger:
            raise ValueError(
                f"Note {index}: pitch {pitch} is outside the fixed hand position {keys}. "
                "Choose five keys with --keys or use a future hand-repositioning planner."
            )
        assignments.append({"start": float(start), "duration": float(duration),
                            "pitch": pitch, "finger": pitch_to_finger[pitch]})
    assignments.sort(key=lambda event: (event["start"], event["pitch"], event["duration"]))
    held_until = {}
    actions_by_time = {}
    for event in assignments:
        start, duration, pitch, finger = (event[key] for key in ("start", "duration", "pitch", "finger"))
        end = start + duration
        if start < held_until.get(finger, 0):
            raise ValueError(
                f"Pitch {pitch} starts at {start:g}s while finger {finger} is still holding "
                f"it until {held_until[finger]:g}s. Overlapping presses of the same key "
                "cannot be played independently."
            )
        held_until[finger] = end
        for timestamp, action in ((start, "press"), (end, "release")):
            group = actions_by_time.setdefault(timestamp, {"release": [], "press": []})
            group[action].append({"finger": finger, "pitch": pitch})
    return {
        "mode": "stationary_right_hand",
        "time_unit": "seconds",
        "starting_position": [
            {"finger": number, "name": name, "pitch": pitch, "state": "raised"}
            for number, (name, pitch) in enumerate(zip(FINGERS, keys), start=1)
        ],
        "notes": assignments,
        # At each timestamp release first, then press. Entries within either
        # list are simultaneous targets; repeated notes can re-press at release.
        "events": [
            {"timestamp": timestamp, "release": sorted(actions["release"], key=lambda a: a["finger"]),
             "press": sorted(actions["press"], key=lambda a: a["finger"])}
            for timestamp, actions in sorted(actions_by_time.items())
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notes_file", type=Path, help="JSON list of (start, duration, pitch) triples")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--keys", type=int, nargs=5, default=DEFAULT_KEYS,
                        metavar="PITCH", help="Five pitches from thumb to pinky (default: 60 62 64 65 67)")
    args = parser.parse_args()
    try:
        notes = json.loads(args.notes_file.read_text(encoding="utf-8-sig"))
        if not isinstance(notes, list):
            raise ValueError("Input JSON must be a list of note triples.")
        result = json.dumps(notes_to_fingers(notes, keys=args.keys), indent=2) + "\n"
        if args.output:
            args.output.write_text(result, encoding="utf-8")
        else:
            print(result, end="")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

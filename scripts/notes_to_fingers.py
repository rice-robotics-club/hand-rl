"""Plan which finger presses each note, for a hand that stays in one five-key position.

Run from the repository root:
    python -m scripts.notes_to_fingers notes.json -o plan.json
"""

import argparse
import json
from pathlib import Path

from src.music.finger_plan import notes_to_fingers
from src.music.fingering import HOME_PITCHES
from src.music.notes import load_notes


class Args(argparse.Namespace):
    notes_file: Path = Path()
    output: Path | None = None
    keys: list[int]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "notes_file", type=Path, help="JSON list of [start, duration, MIDI pitch]"
    )
    _ = parser.add_argument("-o", "--output", type=Path, help="Save JSON to this file")
    _ = parser.add_argument(
        "--keys",
        type=int,
        nargs=len(HOME_PITCHES),
        default=list(HOME_PITCHES),
        metavar="PITCH",
        help="Home pitch of each finger, thumb to pinky (default: 60 62 64 65 67)",
    )
    args = parser.parse_args(namespace=Args())
    try:
        plan = notes_to_fingers(load_notes(args.notes_file), keys=args.keys)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    result = json.dumps(plan, indent=2) + "\n"
    if args.output:
        _ = args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()

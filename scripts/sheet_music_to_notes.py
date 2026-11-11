"""Read right-hand piano notes from a sheet-music image (or MusicXML) as JSON.

Run from the repository root:
    python -m scripts.sheet_music_to_notes sheet.png -o sheet.notes.json
"""

import argparse
import json
from pathlib import Path

from src.music.sheet_music import musicxml_to_notes, sheet_music_to_notes


class Args(argparse.Namespace):
    input: Path = Path()
    output: Path | None = None
    bpm: float = 120
    audiveris: str = "Audiveris"
    timeout: float = 300
    musicxml: bool = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("input", type=Path, help="Sheet-music JPG or PNG")
    _ = parser.add_argument("-o", "--output", type=Path, help="Save JSON to this file")
    _ = parser.add_argument(
        "--bpm",
        type=float,
        default=120,
        help="Fallback quarter-note BPM (default: 120)",
    )
    _ = parser.add_argument(
        "--audiveris", default="Audiveris", help="Audiveris executable name or path"
    )
    _ = parser.add_argument(
        "--timeout", type=float, default=300, help="Recognition timeout in seconds"
    )
    _ = parser.add_argument(
        "--musicxml", action="store_true", help="Read MusicXML instead of an image"
    )
    args = parser.parse_args(namespace=Args())
    try:
        if args.musicxml:
            notes = musicxml_to_notes(args.input, bpm=args.bpm)
        else:
            notes = sheet_music_to_notes(
                args.input, bpm=args.bpm, audiveris=args.audiveris, timeout=args.timeout
            )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    result = json.dumps(notes, indent=2) + "\n"
    if args.output:
        _ = args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()

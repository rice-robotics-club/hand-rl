"""Read right-hand piano notes from sheet music: MusicXML, or a JPG/PNG image.

Image recognition uses the external Audiveris application. MusicXML parsing needs
music21: `uv sync --extra sheet`.
"""

import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from src.music.midi import midi_to_notes
from src.music.notes import Note


def musicxml_to_notes(path: str | Path, *, bpm: float = 120) -> list[Note]:
    """Read the first (upper/right-hand) staff; use bpm as the fallback tempo.

    Assumes conventional piano staff order. This is staff selection, not hand
    inference for cross-staff notation or scores mixing both hands on one staff.
    """
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("BPM must be a positive finite number.")
    try:
        from music21 import (  # pyright: ignore[reportMissingImports]
            converter,
            midi,
            note,
            tempo,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "music21":
            raise
        raise RuntimeError(
            "Install sheet-music dependencies: uv sync --extra sheet"
        ) from exc

    score = converter.parse(str(path))
    # MusicXML piano staves are parsed as ordered PartStaff objects. Keep the
    # first staff's notes. Retain other staves' timing/tempo/repeat structure.
    for part in list(score.parts)[1:]:
        for event in list(part.recurse().notes):
            event.activeSite.replace(
                event, note.Rest(quarterLength=event.quarterLength)
            )
    opening_tempos = [
        mark
        for mark in score.flatten().getElementsByClass(tempo.MetronomeMark)
        if mark.offset == 0 and mark.getQuarterBPM() is not None
    ]
    if not opening_tempos:
        score.insert(0, tempo.MetronomeMark(number=bpm))

    # music21's MIDI exporter handles ties, chords, repeats and tempo changes.
    midi_file = midi.translate.music21ObjectToMidiFile(score, addStartDelay=False)
    with tempfile.TemporaryDirectory(prefix="sheet-midi-") as directory:
        midi_path = Path(directory) / "recognized.mid"
        midi_path.write_bytes(midi_file.writestr())
        notes = midi_to_notes(midi_path)
    if not notes:
        raise ValueError(
            "No playable pitched notes were found in the right-hand staff."
        )
    return notes


def sheet_music_to_notes(
    path: str | Path,
    *,
    bpm: float = 120,
    audiveris: str = "Audiveris",
    timeout: float = 300,
) -> list[Note]:
    """Recognize one JPG/PNG sheet and return right-hand note tuples.

    Requires the external Audiveris application. OMR results may need correction;
    this function does not guarantee that the recognized score matches the image.
    """
    image_path = Path(path).resolve()
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Sheet input must be a JPG, JPEG, or PNG image.")
    if not image_path.is_file():
        raise FileNotFoundError(f"Sheet image not found: {image_path}")
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("BPM must be a positive finite number.")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be a positive finite number.")
    executable = shutil.which(str(audiveris))
    if executable is None:
        raise FileNotFoundError(
            "Audiveris was not found. Install it and add it to PATH, or pass "
            "--audiveris with the path to its executable. "
            "See https://audiveris.github.io/audiveris/_pages/tutorials/install/binaries/"
        )

    with tempfile.TemporaryDirectory(prefix="sheet-omr-") as directory:
        command = [
            executable,
            "-batch",
            "-transcribe",
            "-export",
            "-output",
            directory,
            "--",
            str(image_path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW
                if sys.platform == "win32"
                else 0,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Sheet recognition exceeded {timeout:g} seconds; try --timeout with a larger value."
            ) from exc
        # Keep JSON stdout clean, but preserve recognition diagnostics.
        for output in (result.stdout, result.stderr):
            if output:
                print(
                    output, file=sys.stderr, end="" if output.endswith("\n") else "\n"
                )
        if result.returncode:
            raise RuntimeError(f"Audiveris failed with exit code {result.returncode}.")
        scores = [
            file
            for file in Path(directory).rglob("*")
            if file.suffix.lower() in {".mxl", ".musicxml", ".xml"} and file.is_file()
        ]
        if len(scores) != 1:
            raise RuntimeError(
                f"Expected one recognized score, found {len(scores)}. "
                "Use an image containing one score, or correct/export it in Audiveris "
                "and use --musicxml."
            )
        return musicxml_to_notes(scores[0], bpm=bpm)

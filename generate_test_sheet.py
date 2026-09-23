"""Generate an original right-hand exercise and independently specified expected notes.

Requires requirements-sheet.txt plus verovio and resvg-py.
"""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from music21 import bar, chord, clef, instrument, metadata, meter, midi, note, stream, tempo
import resvg_py
import verovio

from midi_to_notes import midi_to_note_tuples
from sheet_music_to_notes import musicxml_to_notes


def main():
    directory = Path(__file__).resolve().parent / "samples" / "right_hand"
    directory.mkdir(parents=True, exist_ok=True)
    # Each item is (MIDI pitches, duration in quarter-note beats); [] is a rest.
    measures = [
        [([60], 1), ([62], 1), ([64], 1), ([65], 1)],
        [([67], 2), ([64], 1), ([], 1)],
        [([60, 64], 2), ([62, 65], 2)],
        [([60, 64, 67], 4)],
    ]
    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = "Right-Hand Test"
    score.metadata.composer = "Original test exercise"
    part = stream.Part()
    part.insert(0, instrument.Piano())
    expected = []
    beat = 0.0
    for number, events in enumerate(measures, start=1):
        measure = stream.Measure(number=number)
        if number == 1:
            measure.append(clef.TrebleClef())
            measure.append(meter.TimeSignature("4/4"))
            measure.append(tempo.MetronomeMark(number=120))
        for pitches, length in events:
            if not pitches:
                event = note.Rest(quarterLength=length)
            elif len(pitches) == 1:
                event = note.Note(pitches[0], quarterLength=length)
            else:
                event = chord.Chord(pitches, quarterLength=length)
            if pitches:
                for pitch in event.pitches:
                    if pitch.accidental is not None and pitch.accidental.alter == 0:
                        pitch.accidental = None
            measure.append(event)
            expected.extend((beat * 0.5, length * 0.5, pitch) for pitch in pitches)
            beat += length
        if number == len(measures):
            measure.rightBarline = bar.Barline("final")
        part.append(measure)
    score.insert(0, part)
    xml_path = directory / "right_hand.musicxml"
    score.write("musicxml", fp=xml_path)
    midi_path = directory / "right_hand.mid"
    midi_path.write_bytes(midi.translate.music21ObjectToMidiFile(score).writestr())
    (directory / "expected.notes.json").write_text(
        json.dumps(expected, indent=2) + "\n", encoding="utf-8"
    )

    renderer = verovio.toolkit()
    renderer.setOptions({"pageWidth": 2100, "pageHeight": 1000, "scale": 100,
                         "adjustPageHeight": True, "breaks": "none",
                         "header": "none", "footer": "none"})
    if not renderer.loadFile(str(xml_path)):
        raise RuntimeError("Could not load the generated score for engraving.")
    if renderer.getPageCount() != 1:
        raise RuntimeError("Expected a single-page exercise.")
    svg = renderer.renderToSVG(1)
    # resvg does not load Verovio's embedded webfont for the metronome symbol.
    # Use equivalent plain-text quarter-note BPM, keeping notes as vector paths.
    namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", namespace)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.fromstring(svg)
    for group in root.iter(f"{{{namespace}}}g"):
        if group.get("class") == "tempo":
            text_element = group.find(f"{{{namespace}}}text")
            if text_element is not None:
                position = {key: text_element.get(key, "0") for key in ("x", "y")}
                text_element.clear()
                text_element.attrib.update(position, **{"font-size": "405px"})
                text_element.text = "120 BPM"
    svg = ET.tostring(root, encoding="unicode")
    (directory / "right_hand.svg").write_text(svg, encoding="utf-8")
    (directory / "right_hand.png").write_bytes(
        resvg_py.svg_to_bytes(svg_string=svg, background="white", width=3000)
    )
    assert midi_to_note_tuples(midi_path) == expected, "MIDI differs from known notes"
    assert musicxml_to_notes(xml_path) == expected, "MusicXML differs from known notes"
    print(f"Created {directory}; verified {len(expected)} notes over {beat * 0.5:g} seconds.")


if __name__ == "__main__":
    main()

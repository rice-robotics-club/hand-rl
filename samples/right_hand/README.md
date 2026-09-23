# Right-hand recognition test

`right_hand.png` is an original four-bar piano exercise: single notes, a rest,
two-note chords, and a final C-major chord. It uses only C4 through G4, in 4/4
at 120 BPM. The score lasts 8 seconds and contains 13 individual note events.

- `expected.notes.json`: known `(start_seconds, duration_seconds, MIDI_pitch)` values.
- `right_hand.mid`: matching reference MIDI.
- `right_hand.musicxml`: editable notation and a way to test conversion without OMR.
- `right_hand.svg`: scalable engraved source for the PNG.

Run image recognition after installing Audiveris:

```powershell
.venv-audio/Scripts/python.exe sheet_music_to_notes.py samples/right_hand/right_hand.png -o samples/right_hand/detected.notes.json
```

Compare the result against `expected.notes.json`. The supplied expected values
are calculated from the exercise definition, not from image recognition. The
MIDI and MusicXML conversions are checked against them by the generator; that
does not verify Audiveris recognition of the PNG.

Regenerate from the repository root:

```powershell
uv pip install --python .venv-audio/Scripts/python.exe -r requirements-sheet.txt verovio resvg-py
.venv-audio/Scripts/python.exe generate_test_sheet.py
```

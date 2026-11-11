# Right-hand exercise

An original four-bar piano exercise for the right hand: single notes, a rest,
two-note chords, and a final C-major chord. It uses only C4 through G4, in 4/4
at 120 BPM. The score lasts 8 seconds and contains 13 individual note events.
`config/piano.yaml` trains on it.

- `expected.notes.json`: known `(start_seconds, duration_seconds, MIDI_pitch)`
  values, calculated from the exercise definition. Training reads this
  (`env.notes_path`).
- `robot_finger_movement_plan.json`: which finger plays each note, from
  `python -m scripts.notes_to_fingers`. Training doesn't read it; `finger_nearness`
  uses the same five-finger position from `env.finger_pitches`.
- `right_hand.mid`: matching reference MIDI.
- `right_hand.musicxml`: editable notation, for testing conversion without
  image recognition (OMR).
- `right_hand.png`: the engraved score, for testing image recognition.
- `right_hand.svg`: scalable engraved source for the PNG.

Commands below run from the repository root.

Regenerate the fingering plan after changing the notes:

```sh
python -m scripts.notes_to_fingers data/right_hand/expected.notes.json -o data/right_hand/robot_finger_movement_plan.json
```

Run image recognition (needs [Audiveris](https://audiveris.github.io/audiveris/)
and music21) and compare the result against `expected.notes.json`:

```sh
python -m scripts.sheet_music_to_notes data/right_hand/right_hand.png -o data/right_hand/detected.notes.json
```

The MIDI and MusicXML files are checked against `expected.notes.json` by the
generator; that does not verify Audiveris recognition of the PNG.

Regenerate the score files (needs music21, verovio, and resvg-py, which are not
project dependencies):

```sh
python -m scripts.generate_test_sheet
```

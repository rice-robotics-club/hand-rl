# hand-rl

Convert MIDI files into a chronological sequence of notes, with timestamps and
durations in seconds. No audio processing or Fourier transform is needed.

```sh
python -m pip install -r requirements.txt
python midi_to_notes.py song.mid --no-octave -o notes.json
```

Example output:

```json
[
  {"note": "A", "timestamp": 0.0, "duration": 0.5},
  {"note": "C", "timestamp": 0.5, "duration": 0.25},
  {"note": "B", "timestamp": 0.75, "duration": 0.25},
  {"note": "B", "timestamp": 1.0, "duration": 0.5}
]
```

Omit `--no-octave` to keep octave numbers (`A4`, `C5`, etc.), which distinguish
different keys. Omit `-o notes.json` to print the JSON instead.

You can also use it from Python:

```python
from midi_to_notes import midi_to_notes

notes = midi_to_notes("song.mid", include_octave=False)
names = [entry["note"] for entry in notes]
```

All non-drum instruments are combined. Chord notes share a timestamp and remain
separate entries; repeated notes are preserved. Durations run from note-on to
note-off (key release), without extending for the sustain pedal. Tempo changes
are handled by [pretty_midi](https://github.com/craffel/pretty-midi).

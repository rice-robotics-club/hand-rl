"""Music handling shared by environments: notes, scores, fingering, and audio.

- `keyboard`: the piano's keys and their MIDI pitches.
- `notes`: the (start, duration, pitch) note format, with loading and validation.
- `score`: what each key should be doing at a given time.
- `fingering`: which finger should play each wanted key.
- `finger_plan`: a timed press/release plan for a five-finger hand position.
- `midi`, `sheet_music`: converting MIDI, audio, MusicXML, and sheet images to notes.
- `audio`: playing key presses out loud.
"""

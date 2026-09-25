# hand-rl
what a handy piece of software

hand-rl is the repository for the 2026-2027 Rice Robotics hand reinforcement learning project.

The overall goal of this project is to create a rl-trained hand that can perform both generalized hand movement, and a specialized subnet that is capable of playing a real piano (with a heavier focus on the piano, since that's the cool part)


# general rough outline
## Physics Sim
Our physics sim is written as a layer on top of the Genesis-World library, which provides a nice interface for both running a sim agnostic to and consistent across different dev hardware and getting all the info we need to train the RL model

- We want a base environment class that just like creates an empty 3d space to put things in
  - We can decorate this for fun. last year we made the catbot environment look like minecraft. would love to hear ideas
- we want a generalized "hand" object that is sort of abstracted from the specific hand model we're using (so we can swap out designs as our engineering team iterates on it)
  - the specific hand designs should provide info on like, the sort of 'skeleton' of the hand in terms of where its joints and connecting parts live in a minimal physics sense and then where its 3d model parts live and should be attached to that skeleton in the sim.
    -  this is a pretty common thing in robotics so theres a few differerent common file formats like .xml and .urdf that are used for this that genesis should be able to work with pretty plug-and-play
- we want a piano object that can be placed in the environment and interact with the hand.
  - needs to have like methods to get the positions of the keys boundary boxes similar to what we want the PianoMind neural net to do (so we can easily check its answers and coach it)
  - would be neat if it actually made sound in the sim hahaha
    - does genesis support audio and simulated microphones??? i actually have no idea lol

## Reinforcement Learning
Our RL model consists of one decision-making network that utilizes three specialized subnetworks:
### GeneralPerception
- Camera input --> Object identification. Marks positions of objects in the hand's workspace,
  - formatted as a list of object names and their bounding box in 3d space
### MotorActor
- Planned sequence of "keyframe" positions for the hand to hit --> actual control over the hand that does so in the most energy efficient way possible / the best it can do within safe limits of the real system
-
### PianoMind
- Camera input --> specific key positions when a piano is detected
  - formatted as like a list of key names and their bounding boxes in 3d space
- MIDI song input + external clock + key positions from the prior network --> planned sequence of hand+finger joint movements
  - formatted as a sequence of joint angles for each hand+finger at each timestep
  - This is going to need to be a convolutional neural network of some sort to read the midi input properly.

This set of tools is probably enough for us to directly control it pretty well (i.e. run some script that loads in a MIDI file and generates a plan for the hand, then feeds that to the MotorActor model) ((maybe we want a different structure that allows it to adjust in realtime if the keyboard moves under it or something? right now the plan is "baked" once and sort of set in stone after that. at the very least maybe having the ability to apply a coordinate transform to it after the fact to adjust to real piano position would be nice. worth thinking about))

## audio-tools
- responsible for the loading of a midi file and conversion into a nice array of note events that our neural network can realistically parse
- we also mentioned once that it might be neat generating note events from audio input. could be fun to just like generate a midi track from arbitrary .mp3 input... sing to the robot and it plays along or something?

### MIDI note converter

Convert MIDI files into a chronological sequence of notes, with timestamps and
durations in seconds. No audio processing or Fourier transform is needed.

```sh
python -m pip install -r requirements.txt
python midi_to_notes.py song.mid --no-octave -o notes.json
```

MP3 and WAV inputs are automatically transcribed to MIDI in memory before
extracting notes. On Windows, use a separate Python 3.10 environment for
Basic Pitch's ONNX backend. This avoids conflicts with the main project's
Python environment and any globally installed TensorFlow/NumPy packages:

```sh
uv venv --python 3.10 .venv-audio
uv pip install --python .venv-audio/Scripts/python.exe -r requirements-audio.txt
.venv-audio/Scripts/python.exe midi_to_notes.py song.wav --tuples -o notes.json
```

Audio transcription is approximate. MIDI inputs skip transcription and do not
require Basic Pitch. No intermediate MIDI file is written.

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

For a list of `(start_seconds, duration_seconds, MIDI_pitch)` tuples:

```python
from midi_to_notes import midi_to_note_tuples

notes = midi_to_note_tuples("song.mid")
# Example: [(0.0, 0.5, 69), (0.5, 0.25, 72), (0.75, 0.25, 71)]
```

Pitch is a MIDI note number (60 = C4, 69 = A4). Use `--tuples` on the command
line for the same triples as JSON arrays (JSON has no tuple type).
MP3 and WAV inputs work with this format too. `--no-octave` only affects named
output; tuple pitches always retain their full MIDI note number.

Tuples are sorted by start time and preserve timing for rests and overlapping
notes. Two notes overlap when
`max(start_a, start_b) < min(start_a + duration_a, start_b + duration_b)`.
Notes that start exactly when another ends do not overlap.

All non-drum instruments are combined. Chord notes share a timestamp and remain
separate entries; repeated notes are preserved. Durations run from note-on to
note-off (key release), without extending for the sustain pedal. Tempo changes
are handled by [pretty_midi](https://github.com/craffel/pretty-midi).

### Sheet-music image converter

`sheet_music_to_notes.py` uses [Audiveris](https://audiveris.github.io/audiveris/_pages/tutorials/install/binaries/)
to recognize printed sheet music in JPG/JPEG/PNG images. Install Audiveris
separately and add its executable to PATH, or pass its full path with
`--audiveris`. Then install the Python dependencies and run:

```sh
python -m pip install -r requirements-sheet.txt
python sheet_music_to_notes.py sheet.png --bpm 120 -o sheet.notes.json
```

This converter focuses on right-hand piano music: it takes notes from the first
(upper) staff and excludes notes from the lower staff. Single-staff scores are
kept intact. This assumes conventional piano staff order, not an ensemble score.
It does not infer hands when both hands share one staff or use cross-staff
notation, and it does not check whether a chord is physically reachable by one
hand. No pitch cutoff is applied, so low right-hand notes are preserved.

It returns `(start_seconds, duration_seconds, MIDI_pitch)` tuples in Python
(JSON arrays in the command-line output), matching the MIDI converter:

```python
from sheet_music_to_notes import sheet_music_to_notes

notes = sheet_music_to_notes("sheet.jpg", bpm=120)
# Example: [(0.0, 0.5, 60), (0.0, 0.5, 64), (0.5, 1.0, 67)]
```

Recognized tempos and tempo changes control timing. `--bpm` supplies a fallback
when there is no recognized opening tempo, rather than overriding written tempos.
Chords stay as separate simultaneous notes, ties join held notes, and rests
create gaps in the timeline. Recognized repeats are expanded by the MIDI exporter.

Use a clear, straight image of one printed score. Recognition can miss or
misread notes, accidentals, rhythms, or tempo marks; photos and handwriting are
less reliable. Inspect and correct recognition in Audiveris when needed, then
export MusicXML and run `python sheet_music_to_notes.py corrected.mxl --musicxml`.
The image path uses temporary recognition files and does not save a reviewable
Audiveris project. Increase `--timeout` from its default 300 seconds for slow scans.

### Note sequences to finger actions

`notes_to_fingers.py` assigns note tuples from either converter to a stationary
right hand. Finger numbers are 1=thumb, 2=index, 3=middle, 4=ring, 5=pinky.
The default starting position is raised above C4, D4, E4, F4, G4 respectively.

```powershell
.venv-audio/Scripts/python.exe notes_to_fingers.py samples/right_hand/expected.notes.json -o robot_finger_movement_plan.json
```

The JSON includes `starting_position`, finger assignments in `notes`, and timed
`events`. Each event contains simultaneous `release` and `press` lists. Apply
releases first, then presses at that timestamp. Chords use several fingers at
once; held notes stay down until their own release, and rests produce no presses.
The generated exercise's final chord uses thumb, middle, and pinky from 6–8 seconds.

Use `--keys 72 74 76 77 79` for a C5–G5 starting position, or specify another set
of five ascending MIDI pitches. This only configures the key assignment; it does
not verify the physical reach of the chosen position. Notes outside the five
keys and overlapping presses of the same key are rejected.

These are **finger/key targets, not motor commands**. The robot controller still
needs key coordinates, joint geometry, travel times, and control over pressing
and releasing. Timestamps specify when the key should sound, so physical motion
must begin early enough to meet them. Adjacent repeated notes may require more
release/repress time than this idealized schedule allows. Wrist movement and
automatic fingering across a wider keyboard are not implemented yet.

# Overall Goals / Timeline

# Pure Simulation goals:
Milestone 0: Hand is loaded in a 3d sim environment and can be controlled manually by a simple script

Milestone 1: Hand can be controlled by an RL model doing random actions

Milestone 2: Stationary hand can move fingers and play singular notes with RL

Milestone 3: Stationary hand can play chords with RL

Milestone 4: Hand+Arm can move, play more complex sequences of notes across full keyboard with RL

Milestone 5: Hand+Arm can play through a full song with RL

# Sim2Real goals:
Milestone 0: designated training environment for sim2real transfer learning that loads in a pure-sim trained neural network and its bundle of reward functions.
- Could also do a student-teacher separation here, where maybe the pure-sim is a bigger more complex model with access to unrealistic kinds of data and then we start up a fresh network here in sim2real that has the actual limitations of the real robot and tries to imitate the teacher model's behavior?

Milestone 1: Random sensor noise is simulated for everything (actuators, cameras, mics, etc)

Milestone 2: Motor torque, resistance, physical parameters like that are all noisy as well

Milestone 3:

# Real Robot goals:
Milestone 0: some sort of very basic onboard control loop with ros2 exists on the hand

Milestone 1: Camera, Mic, joint position sensors are all connected as ros2 topics

Milestone 2: reinforcement learning models can be deployed on the robot to control hand movements

Milestone 3:

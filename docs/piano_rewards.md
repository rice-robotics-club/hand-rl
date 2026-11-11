# Score-following rewards

The environment is `HandPianoEnv` in `src/env/hand_piano.py`. It uses:

| File | What it holds |
| --- | --- |
| `src/env/hand_piano.py` | The config, `HandPianoEnvState`, and the env: builds the state, steps, resets. |
| `src/sim/hand_piano.py` | `HandPianoSystem`, the interface the env needs from a system, and `HandPiano`, which implements it in Genesis. |
| `src/music/score.py` | `Score`: the notes, and what each key should be doing at a given time. |
| `src/music/fingering.py` | Which finger should play each wanted key. |
| `src/reward/piano.py` | Note and nearness terms, and `KeyTargets`/`FingerTargets`, the state they read. |
| `src/reward/motion.py` | Smoothness and posture terms, and `MotionState`, the state they read. |
| `src/reward/curriculum.py` | How reward weights change as note recall improves. |

`HandPianoEnvState` inherits from `HandPianoState`, `FingerTargets`, and
`MotionState`, so every term's state type is a base of it. Another environment
can reuse the terms by including the same base classes in its state.

## Score

`env.notes_path` is a JSON list of `[start_seconds, duration_seconds,
MIDI_pitch]`, such as `data/right_hand/expected.notes.json`. Every environment
plays it every episode.

## Design rule: rewards are pure functions of state and action

Each reward term is a module-level function `term(state, action) -> Tensor` with
no access to the environment object. Anything a reward needs must be a field of
the state type the term declares (`KeyTargets`, `FingerTargets`, or
`MotionState`), such as `target_active`, `key_down`, or `closest_finger_distances`.
Each derived field has its own function (`keys_down`, `closest_fingers`,
`finger_distances`, ...), and `HandPianoEnv.state_at` assembles them. Fixed
hyperparameters such as `nearness_scale` are keyword arguments bound once with
`src.reward.bind` when the `RewardFunction` is built.

## Enabled terms

At each control interval, threshold measured key travel at `pressed_threshold`.
Let H be correctly held keys, M missing required keys, E extra keys, and
N=max(number of required keys, 1). The default reward is:

`reward = dt * (H/N - M/N - E)`

Each term has its own nonnegative configuration weight. A correct single note
or complete chord gives +1 per simulated second; missing every requested note
gives -1; each extra key subtracts 1. Correct rests give zero, not a large reward
for doing nothing. Nearness is not counted as a successful press.

Comparing keys every step naturally scores holds, early release, early presses,
and late releases. Simultaneous pitches remain independent targets. MIDI tuples
are `(start_seconds, duration_seconds, MIDI_pitch)`. Targets are start-inclusive
and end-exclusive; lead-in time is added without rounding the original score.

The reward compares the measured state after the action against the target at
the **start** of that action's interval. The next observation contains targets
for the next interval. Thus the last holding action earns credit before a
release becomes due. Resolution is one control interval (configured to 1/60 s).
Notes shorter than that can fall between samples. Adjacent repeated notes of
the same pitch currently need a rest to distinguish releases/restrikes; this
is held-key correctness, not acoustic or note-onset accuracy.

The environment reads dt from the system's Genesis configuration. Rewards are
multiplied by dt so their accumulated scale approximates a time integral;
changing frequency can still change sampled contacts and discounting. Logged
reward terms are weighted rates, before multiplying by dt.

Episode length is derived from the score plus lead-in and a release tail.
An explicit shorter episode length is rejected. Completing the score is
terminal (no timeout bootstrapping). Each parallel environment resets its
score clock independently. The sample has 8 seconds of music plus 1 second of
lead-in/tail, or 540 steps at 60 Hz.

## Observations

Physical state is supplemented with `target_active`, `target_remaining`,
`next_note_valid`, `next_note_time`, `next_note_duration`, and a scalar
`note_correctness`. Correctness is intersection-over-union of requested and
actual keys, with 1 for a correctly silent rest. Unlike the reward, this
diagnostic is always between 0 and 1.

Key channel `i` is MIDI pitch `21 + i` (A0 to C8), the order of the piano's
key joints.

## Optional nearness

The implemented term is the mean `1 - tanh(distance / nearness_scale)` over
currently requested keys. Distance is 3D fingertip-to-press-point distance in
meters, including height. The code default scale is 0.02 m; `config/piano.yaml`
uses 0.1 m. Rests receive zero.

`HandPiano` measures `fingertip_key_distances` as a (envs, 5, 88) matrix: the
distance from each fingertip (thumb to pinky) to each key's press point, in
pressed-channel order. A fingertip is the collision vertex of the distal link
farthest from its last joint. A press point is on the top of the key, centered
across its width, `piano.press_point_inset` (default 0.02 m) behind the key's
front edge. Black keys are measured from their own, shorter front edge.

There are two nearness terms, which differ in which finger they measure:

- **`note_nearness`** (`closest_fingers`, `closest_finger_distances`): each step,
  requested keys get distinct fingers with the least total distance (all
  assignments are checked; at most 120 for five keys). Any finger counts, and
  assignments can change from step to step.
- **`finger_nearness`** (`home_fingers`, `home_finger_distances`): each key is
  measured from the finger it sits under in the hand's starting position,
  `env.finger_pitches` (thumb to pinky; C4 D4 E4 F4 G4 by default). This is the
  same fingering `src/music/finger_plan.py` produces. A wanted key under no
  finger is measured from the nearest fingertip.

A requested key with no finger (more than five keys at once) is measured from
the nearest fingertip. The policy does not observe the assignment.

`config/piano.yaml` enables nearness at weight 1.0 with `nearness_scale: 0.1`
(0.90 at 1 cm, 0.54 at 5 cm, 0.24 at 10 cm, 0.01 at 25 cm), lowers
`extra_notes` to 0.2, and uses a fixed PPO learning rate of 3e-4. These are
untested starting points, not tuned values or a guarantee against hovering.

They respond to runs that plateaued at -7.5, the reward for never pressing a
key (notes are requested for 7.5 s of the episode). In those runs:

- At a 0.02 m scale, nearness was effectively zero beyond a few centimeters
  (0.00009 at 10 cm), so the hand got no signal once it lifted off the keys.
- An extra key cost as much as a whole missed note, so avoiding the keys was
  the safest policy.
- PPO's adaptive schedule held the learning rate at its 1e-5 floor from the
  first update, so the policy barely changed.

## Smoothness and posture

- **`action_rate`**: minus the mean squared change between consecutive actions
  (clipped to [-1, 1]). The previous action is the state field `previous_action`.
- **`finger_posture`**: `exp(-(rms/posture_scale)^2)` of the joints' deviation
  from the system's default posture (`posture_error`). The default posture is
  `HandPiano.initial_joint_positions`, a pianist's hand: curved fingers with
  the five fingertips in a row about 2 cm above the white keys. Zero actions
  hold the hand there, and resets return to it.
- **`hand_orientation`**: `exp(-(angle/orientation_scale)^2)` of the base's
  rotation away from level, palm down (`orientation_error`).

The system also smooths actions (`action_smoothing`, a moving average) and
exposes the smoothed targets as `target_offsets`. `config/piano.yaml` adds
`previous_action` and `target_offsets` to the observations, so checkpoints from
before this change do not load.

## Contact

Fingertips sink into the keys a little because the PD-driven base can push
hard. Measured by lowering the hand onto the keys: 2 substeps gave 2.4 mm for a
4 cm push and 7.4 mm for an 8 cm push; 4 substeps gave 1.2 and 4.0 mm (the same
as 8). Capping the base force at 10 N (`base_force_limit`) brought the hard
push to 2.0 mm while normal presses still played. `config/piano.yaml` uses 4
substeps and that cap.

## Curriculum

The env tracks a running average of note recall (requested key-steps played,
per finished episode; `curriculum_rate` sets how fast it moves). Progress rises
from 0 at `curriculum_start` recall to 1 at `curriculum_end`, never decreases,
and moves each weight linearly from `reward_weights` to `final_reward_weights`.
`config/piano.yaml` uses this to raise `extra_notes` (0.2 to 1.0) and
`wrong_keys` (0 to 0.5), and to move nearness credit from any finger
(`note_nearness`, 1.0 to 0.2) to the planned finger (`finger_nearness`, 0 to
1.0). Progress restarts at 0 with each training run, including resumed ones.
TensorBoard shows `curriculum/note_recall` and `curriculum/progress`, plus
`notes/precision`, `notes/recall`, and `notes/f1` for the keys down at the
last step (recall is 1 during rests); logged
reward terms include their current weight, so a term at weight 0 logs 0.

Intensity and electrical energy are not implemented: they need strike-velocity
targets and actuator measurements/models respectively. Tests use synthetic
key measurements to validate reward logic; they do not establish physical
piano-playing performance.

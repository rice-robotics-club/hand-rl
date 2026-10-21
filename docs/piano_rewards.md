# Score-following rewards

The reward implementation lives in `src/env/hand_piano.py` and uses the existing
`RewardFunction`, `HandPianoAction`, and `VecSystem` interfaces. The default
`config/piano.yaml` now trains score following instead of random base reaching.
Existing reaching checkpoints have different observations and cannot be resumed
with this configuration; start a new run.

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
reward terms are unweighted rates, before multiplying by dt.

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

`env.key_pitches` explicitly maps pressed-state channel indices to MIDI pitches.
The default is ascending A0 through C8 (21..108); verify this against the loaded
Genesis piano DOF ordering before interpreting physical training results. A
permutation can be configured without changing reward code.

## Optional nearness

The implemented term is the mean `1 - tanh(distance / nearness_scale)` over
currently requested keys. Distance is 3D fingertip-to-press-point distance in
meters, including height. The starting scale is 0.02 m. Rests receive zero.

The simulator does not yet supply verified distances, so the nearness weight
defaults to **zero**. A nonzero weight requires `fingertip_key_distances` in the
system state and fails clearly if it is missing. A geometry provider must use
verified fingertip assignments for chords and the same channel order as pressed;
zeros must never stand in for unavailable measurements. Do not substitute base
distance or pitch error. A small weight such as 0.2 is an experiment, not a tuned
value or a guarantee against hovering.

Intensity and electrical energy are not implemented: they need strike-velocity
targets and actuator measurements/models respectively. Tests use synthetic
key measurements to validate reward logic; they do not establish physical
piano-playing performance.

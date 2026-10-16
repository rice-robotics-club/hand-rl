# Piano state contract

This extends the existing typed `RukaState` and `RukaEnv`; `VecSystem`,
`VecAction`, and `RewardFunction` retain their interfaces. This is state and
observation work, not a completed piano simulator or a new action/reward design.

## Timing

`system.control_dt` defaults to 1/60 second. One `VecSystem.step(action)` advances
one control interval. `physics_substeps` defaults to two internal physics steps
per interval; the policy does not run between substeps. Time is simulated time,
not elapsed wall-clock time. Each environment has its own integer step counter;
`time = steps * control_dt` avoids repeated-addition drift. Reset clears only the
selected environments' clocks and immediately refreshes physical measurements.

Before the first step the observation describes time zero. The first returned
step observation describes 1/60 second. A note `(0, 0.5, 60)` is active at
observation times 0 through 29/60, and inactive at 30/60. Musical timestamps are
never rounded to steps: intervals are start-inclusive and end-exclusive.
Sub-step notes may fall between observations; faster control or interval-based
event scoring will be needed when defining rewards for such notes.

## Physical state (`RukaState`)

All fields have a leading environment dimension. Feature widths below exclude it.

| Field | Width | Meaning |
| --- | --- | --- |
| `position`, `orientation`, `velocity` | 3, 4, 3 | Existing Genesis base position (m), quaternion, and velocity (m/s) |
| `joint_positions`, `joint_velocities` | actual DOF count | Genesis DOF order; rotational coordinates rad and rad/s, translational coordinates m and m/s |
| `time`, `control_dt` | 1 each | Float64 simulated seconds since reset and seconds per control step |
| `fingertip_positions` | 15 | Reserved keyboard-frame XYZ in meters, flattened thumb/index/middle/ring/pinky |
| `fingertip_valid` | 5 | 1 only when that finger's position is measured in a verified keyboard frame |
| `key_pitches` | K | Configured MIDI pitches, initially `[60, 62, 64, 65, 67]` |
| `key_displacement` | K | Reserved downward travel in meters from the unpressed position |
| `key_pressed` | K | Reserved 0/1 physical key activation |
| `key_valid` | K | 1 only when key measurements are available |

The current Genesis scene has no keyboard and no verified fingertip mapping.
Therefore fingertip/key measurement arrays AND their validity arrays are zero.
These are explicitly unavailable sensor slots, not simulated key measurements.
Targets must never be copied into measured key state. Wiring these sensors is
future simulator work. DOF state size is discovered from the robot and does not
establish an independent actuator mapping; the existing action remains unchanged.

## Task observations (`RukaEnv`)

Set `env.notes_path` to a JSON array of `[start_seconds, duration_seconds,
MIDI_pitch]` rows, resolved relative to the training process's working directory.
`null` means no target notes. Unsupported pitches are rejected rather than dropped;
expand `system.key_pitches` deliberately if needed. All environments currently
share the same score and keyboard layout, but use their own system clocks.

`get_observations()` returns a TensorDict containing a float32 snapshot of the
physical fields plus these per-key, width-K musical features:

| Field | Meaning |
| --- | --- |
| `target_active` | 1 if this pitch should be held now |
| `target_remaining` | Seconds until its last currently active release, otherwise 0 |
| `next_note_valid` | 1 if this key has a future onset |
| `next_note_time` | Seconds until its next strictly future onset, otherwise 0 |
| `next_note_duration` | Duration of that upcoming note, otherwise 0 |

Multiple pitches can be active together (chords). Overlapping notes of the same
pitch use the latest active release. Repeated pitches still expose their next
onset while held; simultaneous same-pitch future notes use the longest duration.
Original timestamps and comparisons use float64; only policy features use float32.
Returned snapshots are independent of later simulation steps.

The YAML actor/critic observation groups select the new physical and musical
features. Absolute `time` and `control_dt` remain available for inspection but
are not selected by default; relative note times carry the musical timing.

Reward design, episode termination, actuator control, and audio comparison are
outside this change. The existing velocity reward remains a demonstration.

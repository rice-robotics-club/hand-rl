# PROTOTYPE ONLY: five-finger RL experiment

This is an abstract numerical model for developing the training loop **before
CAD and actuators are defined**. It is not a Genesis physics environment, a
model of the real hand, or a deployable controller. It assumes five independent
finger bends above C4–G4. No wrist movement, geometry, collisions, force,
friction, motor dynamics, or audio is simulated.

## Run on Windows from the repository root

```powershell
uv venv --python 3.12 .venv-prototype
uv pip install --python .venv-prototype/Scripts/python.exe -r requirements-prototype.txt --torch-backend cpu
.venv-prototype/Scripts/python.exe -m unittest prototype_rl.test_env -v
.venv-prototype/Scripts/python.exe -m prototype_rl.train --iterations 300
```

Use `--iterations 2 --num-envs 4` for a quick integration check. Use
`--notes path/to/notes.json` for other verified `(start, duration, pitch)` sequences
within C4, D4, E4, F4, G4. Defaults use our eight-second generated exercise.
No rendering window opens. The independent environment avoids changing the
audio dependencies or requiring the unfinished Genesis modules.

## Environment and configuration

`env.py` implements RSL-RL's `VecEnv`. `config.yaml` configures PPO, actor and
critic MLPs, observations, and the simplified environment. Both actor and critic
receive the same `TensorDict["policy"]` observations. Dependencies pin RSL-RL 5.0.0.

- Actions: five values, thumb through pinky, clipped to [-1, 1]. Each sets a
  desired bend: -1 is raised; +1 is fully pressed.
- State: five normalized bends in [0, 1]. Bends move toward actions at a bounded
  speed. A bend >= 0.75 counts as a key press; this is a threshold, not contact physics.
- Observations: bends, normalized velocities, previous actions, six frames of
  target key states (current plus upcoming), and episode progress: 46 numbers.
- Reward: `1 - bend_tracking_MSE - 0.5*missing_keys - 0.5*extra_keys - 0.01*action_change_MSE`.
  Weights are prototype choices, not tuned hardware values.
- Time: 50 ms steps. Note boundaries round up to the next grid point; notes that
  disappear at that resolution are rejected. A 0.5-second lead-in lets the policy
  prepare; a 0.5-second tail rewards release. These extend the episode to 9 seconds.
- Step semantics: apply an action, score against the current timeline frame, then
  advance to the next observation. At score completion the environment returns
  `done` and automatically resets. Completion is terminal, not a timeout.

The reward and evaluation check **held key states**. Adjacent repetitions of the
same note have identical target states, so this prototype does not yet score
re-striking, articulation, velocity, or sound quality. The policy can memorize
the one training score; the evaluation is not a test of new-song generalization.

## Results and next steps

Each run writes a unique directory under ignored `prototype_runs/`, including
TensorBoard logs, configuration, target notes, `evaluation.json`, and
`PROTOTYPE_policy.pt`. Evaluation compares the learned policy to an always-raised
baseline using held-key F1, exact key-state agreement, and mean step reward.
F1 penalizes missing/extra keys so silence alone cannot earn a perfect score.

Validation on the default score (CPU, seed 7, 32 environments, 100 iterations):
five environment tests passed and PPO completed with held-key F1 of 0.945 and
exact key-state agreement of 85.6%, versus 0.0 and 16.7% for always-raised fingers.
These are training-score results in the abstract model, not evidence of physical
playability or performance on unseen music. Checkpoint saving and evaluation
were exercised in both a two-iteration smoke run and this longer run.

Once the hand design exists, replace the abstract bend model with Genesis
joint/contact dynamics, define real actuator actions and observations, and
retrain. Preserve the note targets and time-based reward intent, not these
assumed motor parameters. Add audio validation separately when recordings exist.

API references: [environment](https://leggedrobotics.github.io/rsl_rl/api/env.html),
[configuration](https://leggedrobotics.github.io/rsl_rl/guide/configuration.html).

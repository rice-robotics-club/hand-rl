# Working in hand-rl

A monorepo for RL environments and robots. See README.md for the layout.

## Ask before changing a critical interface

Stop and ask the user, with the options and a recommendation, before adding, removing,
renaming, or changing the type of a field or member on any of these:

- The base classes in `src/sim/vec_system.py` (`VecSystem`, `VecAction`, `GenesisSystem`,
  `ChildSystem`), and `Config` in `src/base_config.py`.
- System state and action types (`PianoState`, `HandPianoState`, `HandPianoAction`,
  `RukaState`, `RukaAction`, ...).
- Interfaces an environment requires from its system, such as `HandPianoSystem` in
  `src/sim/hand_piano.py`.
- Environment state types (`HandPianoEnvState`) and the shared reward state types in
  `src/reward/` (`KeyTargets`, `FingerTargets`, `MotionState`).
- `RewardFunction` and `RewardTerm` in `src/reward/reward.py`.

Config fields, private helpers, and new reward terms that only read existing fields
don't need a prompt.

## No workarounds around interfaces

Don't use `getattr`, `hasattr`, `setattr`, or other duck typing to reach members an
interface doesn't declare, and don't silence the type checker to do it. If code needs
something an interface lacks, that is an interface change: ask (see above). Test fakes
should satisfy the interface they stand in for; where a fake can only cover part of
one, use an explicit `typing.cast` with a comment.

## Where code goes

- `src/sim/` and `src/env/` hold implementations of the `VecSystem` and `VecEnv` base
  classes, one file each. Helpers go elsewhere, so other environments and robots can
  share them: rewards in `src/reward/`, music and notes in `src/music/`, geometry in
  `src/util/`.
- `scripts/` are thin command-line wrappers; logic belongs in `src/`.
- Name numbers that mean something (`NUM_KEYS`, `NUM_JOINTS`, `FREE_JOINT_DOFS`, ...)
  instead of writing literals; reuse the existing constants.

## Rewards

Reward terms are pure functions `term(state, action) -> Tensor`. A value a term needs
must be a field of the state type it declares, computed by its own function when the
environment builds its state. Fixed settings are keyword arguments bound with
`src.reward.bind`.

## Types and checks

Keep types explicit: annotate every parameter, and give tensors jaxtyping shapes. Before
finishing, run (all must pass):

```sh
ruff check src scripts tests
ruff format --check src scripts tests
basedpyright
python -m unittest discover -s tests -t .
```

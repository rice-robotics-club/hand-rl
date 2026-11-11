# hand-rl

what a handy piece of software

hand-rl is the repository for the 2026-2027 Rice Robotics hand reinforcement learning project.

The overall goal of this project is to create a rl-trained hand and arm (named Rozart!) that can perform both generalized hand movement, and a specialized subnet that is capable of playing a real piano (with a heavier focus on the piano, since that's the cool part).

## Layout

This is a monorepo for several environments and robots:

| Directory | What it holds |
| --- | --- |
| `src/sim/` | Systems: one file per `VecSystem` implementation (`hand_piano.py`, `piano.py`, `ruka_sim.py`), plus the base classes in `vec_system.py`. |
| `src/env/` | Environments: one file per `VecEnv` implementation (`hand_piano.py`, `ruka_env.py`, `null.py`). |
| `src/reward/` | Reward terms and helpers shared between environments. |
| `src/music/` | Notes, scores, fingering, MIDI and sheet-music conversion, and audio. |
| `src/config/`, `src/util/` | Config loading, and helpers for the scripts. |
| `scripts/` | Command-line entry points; thin wrappers around library code in `src/`. |
| `config/` | Training configs, and the generated config schema. |
| `data/` | Scores and other inputs. |
| `tests/` | Unit tests, one file per module or environment. |

Ruka (`ruka_sim.py`, `ruka_env.py`, `config/ruka.yaml`) is the first iteration of the
hand; HandPiano is the current piano task.

## Checks

CI runs these on every pull request; run them from the repository root before pushing:

```sh
ruff check src scripts tests
ruff format --check src scripts tests   # `ruff format src scripts tests` fixes it
basedpyright
python -m unittest discover -s tests -t .
```

Ruff and basedpyright settings are in `pyproject.toml`. Every function parameter needs a
type annotation.

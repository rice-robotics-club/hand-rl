"""# Rice Robotics Hand-RL Documentation

This is the repository for the arm + hand robot produced by the R&D team.

It contains a custom framework for structuring a robotics RL environment,
taking advantage of the RSL-RL library produced by the ETH Zürich Robotic
Systems Lab, the Genesis World simulator, and PyTorch.

## Scripts

The process of training, evaluating and testing would be performed by running scripts
specified in [`pyproject.toml`](https://github.com/rice-robotics-club/hand-rl/blob/main/pyproject.toml).

When you set up this project with UV via `uv sync`, these scripts are installed and can be run in a terminal:

- `train`: This is the main script for training a model in an RL environment. Run `train -h` to see all options.
- `evaluate`: This script evaluates a trained model on a given environment. Run `evaluate -h` to see all options.
- `scene`: This script renders a scene of a Genesis simulator based on a given config, without running training.

## Configuration

Rather than having a script for each specific instance or environment you want to train, we will instead use the above scripts with
configuration files specifying the training algorithm, environment and simulation. These will be found in the `config` directory.

## Submodules

- `src.config`: This module contains the configuration classes modeling the training and simulation parameters.
- `src.env`: This module contains environments satisfying the RSL-RL `VecEnv` abstract class, which each provide a situation and reward function for a reinforcement learning scenario.
- `src.sim`: This module contains the `VecSystem` abstract class for a vectorized robotics simulator or dynamic system, allowing the same RL setup to be shared between physical environments (e.g. real robot vs simulation).

"""

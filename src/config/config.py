from pathlib import Path
from typing import Annotated

import yaml
from pydantic import Field, TypeAdapter

from src.base_config import ConfigBase
from src.config.runner import RunnerConfig
from src.env.ruka_env import RukaEnvConfig
from src.sim.ruka_sim import RukaSimConfig

EnvConfig = Annotated[RukaEnvConfig, Field(discriminator="class_name")]

SystemConfig = Annotated[RukaSimConfig, Field(discriminator="class_name")]


class Config[E: EnvConfig, S: SystemConfig](ConfigBase):
    """Root configuration for a training run."""

    runner: RunnerConfig
    """Configuration for a RSL-RL runner."""

    env: E
    """Configuration for the run's environment."""

    system: S
    """Configuration for the run's system."""


def get_config[E: EnvConfig, S: SystemConfig](path: str | Path) -> Config[E, S]:
    """Loads a typed configuration from a file path.

    Type parameters:
        E: The environment configuration type.
        S: The system configuration type.

    Args:
        path: The path to the configuration file.

    Returns:
        The validated configuration.
    """
    raw = yaml.safe_load(Path(path).read_text())
    return TypeAdapter(Config[E, S]).validate_python(raw)

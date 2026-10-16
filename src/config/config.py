from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import Field, TypeAdapter

from src.base_config import Config
from src.config.runner import RunnerConfig
from src.env.hand_piano import HandPianoEnvConfig
from src.env.null import NullEnvConfig
from src.env.ruka_env import RukaEnvConfig
from src.sim.hand_piano import HandPianoConfig
from src.sim.piano import PianoConfig
from src.sim.ruka_sim import RukaSimConfig

EnvConfig = Annotated[
    RukaEnvConfig | NullEnvConfig | HandPianoEnvConfig,
    Field(discriminator="class_name"),
]

SystemConfig = Annotated[
    RukaSimConfig | PianoConfig | HandPianoConfig, Field(discriminator="class_name")
]


class TrainingConfig[E: EnvConfig, S: SystemConfig](Config):
    """Root configuration for a training run.

    Passes the expected runner configuration to the rsl-rl runner, as well as for our user-defined configurations for environments and systems/simulations.
    """

    runner: RunnerConfig
    """Configuration for a RSL-RL runner."""

    env: E
    """Configuration for the run's environment."""

    system: S
    """Configuration for the run's system."""


def get_config[E: EnvConfig, S: SystemConfig](path: str | Path) -> TrainingConfig[E, S]:
    """Loads a typed configuration from a file path.

    Type parameters:
        E: The environment configuration type.
        S: The system configuration type.

    Args:
        path: The path to the configuration file.

    Returns:
        The validated configuration.
    """
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())  # pyright: ignore[reportExplicitAny, reportAny]
    return TypeAdapter(TrainingConfig[E, S]).validate_python(raw)

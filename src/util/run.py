"""Shared helpers for scripts that build systems, environments, and runners from a config.

Import this module before `src.config` or `rsl_rl`, so that the warning filters below are
installed before those imports emit warnings.
"""

import logging
import re
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, override, runtime_checkable

import torch
from tensordict import TensorDict

from src.sim.vec_system import GenesisConfig

warnings.filterwarnings(
    "ignore", message=r".*torch\.jit\.script.*", category=FutureWarning
)

from rsl_rl.env import VecEnv
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from rsl_rl.utils import resolve_class

from src.config import TrainingConfig, get_config
from src.config.config import EnvConfig, SystemConfig
from src.sim import VecSystem

__all__ = [
    "build_env",
    "build_runner",
    "build_system",
    "find_checkpoint",
    "get_config",
    "latest_checkpoint",
    "latest_run",
    "new_run_dir",
    "setup_logging",
]

type Config = TrainingConfig[EnvConfig, SystemConfig]

RUN_TIMESTAMP = re.compile(r"\d{8}-\d{6}$")
"""Matches the `-%Y%m%d-%H%M%S` suffix that `new_run_dir` appends to run directories."""


def setup_logging():
    """Keeps Genesis from duplicating its log output through the root logger."""
    logging.getLogger("genesis").propagate = False


def build_system(cfg: Config) -> VecSystem[TensorDict, torch.Tensor, GenesisConfig]:
    """Instantiates the system (simulation) described by `cfg.system`."""
    system_class, _ = resolve_class(cfg.system.model_dump())
    return system_class(cfg.system)


def build_env(
    cfg: Config, system: VecSystem[TensorDict, torch.Tensor, GenesisConfig]
) -> VecEnv:
    """Instantiates the environment described by `cfg.env` around `system`."""
    env_class, _ = resolve_class(cfg.env.model_dump())
    return env_class(system, cfg.env)


@runtime_checkable
class CheckpointedEnv(Protocol):
    """An environment with training state, beyond the policy, to save in checkpoints.

    For example, a reward curriculum's progress. The runners from build_runner save it
    with every checkpoint and restore it when loading one.
    """

    def checkpoint_state(self) -> dict[str, object]:
        """The state to save; must be picklable."""
        ...

    def load_checkpoint_state(self, state: dict[str, object]) -> None:
        """Restores a state from checkpoint_state."""
        ...


class _SavesEnvState(OnPolicyRunner):
    """Saves a CheckpointedEnv's state in each checkpoint's `infos`, and restores it."""

    @override
    def save(self, path: str, infos: dict[str, object] | None = None) -> None:
        infos = dict(infos or {})
        if isinstance(self.env, CheckpointedEnv):
            infos["env"] = self.env.checkpoint_state()
        super().save(path, infos)

    @override
    def load(
        self,
        path: str,
        load_cfg: dict[str, bool] | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict[str, object]:
        infos: dict[str, object] | None = super().load(
            path, load_cfg, strict, map_location
        )
        env_state = (infos or {}).get("env")
        # Checkpoints from before env state was saved have none; keep the defaults.
        if isinstance(self.env, CheckpointedEnv) and isinstance(env_state, dict):
            self.env.load_checkpoint_state(env_state)
        return infos or {}


class OnPolicyRunnerWithEnvState(_SavesEnvState):
    """OnPolicyRunner that also checkpoints the environment's state."""


class DistillationRunnerWithEnvState(_SavesEnvState, DistillationRunner):
    """DistillationRunner that also checkpoints the environment's state."""


_RUNNERS_WITH_ENV_STATE: dict[Callable[..., object], type[OnPolicyRunner]] = {
    OnPolicyRunner: OnPolicyRunnerWithEnvState,
    DistillationRunner: DistillationRunnerWithEnvState,
}


def build_runner(
    cfg: Config, env: VecEnv, log_dir: Path | str | None
) -> OnPolicyRunner:
    """Instantiates the runner described by `cfg.runner`. Pass `log_dir=None` to skip logging.

    rsl_rl's runners are replaced by versions that also checkpoint the environment's
    state (see CheckpointedEnv).
    """
    runner_class, _ = resolve_class(cfg.runner.model_dump())
    runner_class = _RUNNERS_WITH_ENV_STATE.get(runner_class, runner_class)
    return runner_class(
        env,
        train_cfg=cfg.runner.model_dump(),
        log_dir=None if log_dir is None else str(log_dir),
        device="cuda",
    )


def new_run_dir(cfg: Config, root: Path | str = "logs") -> Path:
    """Returns a new, timestamped run directory such as `logs/piano-20261015-120000`."""
    return Path(root) / f"{cfg.runner.run_name}-{time.strftime('%Y%m%d-%H%M%S')}"


def latest_run(log_dir: Path) -> Path:
    """Resolves `log_dir` to a run directory containing checkpoints.

    `log_dir` may be a run directory itself, a prefix such as `logs/piano` (matching
    `logs/piano-<timestamp>`), or a directory of runs such as `logs`. In the latter two
    cases, the run with the most recent timestamp is returned.
    """
    if any(log_dir.glob("model_*.pt")):
        return log_dir
    candidates = list(log_dir.parent.glob(f"{log_dir.name}-*"))
    if log_dir.is_dir():
        candidates += list(log_dir.iterdir())
    runs = [p for p in candidates if p.is_dir() and RUN_TIMESTAMP.search(p.name)]
    if not runs:
        raise FileNotFoundError(f"No training runs found for {log_dir}")
    return max(runs, key=lambda p: RUN_TIMESTAMP.findall(p.name)[-1])


def find_checkpoint(path: Path) -> Path:
    """A checkpoint file itself, or the latest checkpoint of the latest run at `path`.

    `path` may be a checkpoint, a run directory, a run prefix such as `logs/piano`, or a
    directory of runs; see latest_run.
    """
    return path if path.is_file() else latest_checkpoint(latest_run(path))


def latest_checkpoint(log_dir: Path) -> Path:
    """Returns the checkpoint with the highest iteration number in `log_dir`."""
    checkpoints = list(log_dir.glob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {log_dir}")
    return max(checkpoints, key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))

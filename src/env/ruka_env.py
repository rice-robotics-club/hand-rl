import json
from pathlib import Path
from typing import Literal, override

import torch
from jaxtyping import Float
from pydantic import Field
from rsl_rl.env import VecEnv
from torch import Tensor

from src.base_config import Config
from src.reward import RewardFunction
from src.sim import VecSystem
from src.sim.ruka_sim import RukaAction, RukaState

from .piano_targets import PianoTargets


class PianoObservation(RukaState):
    """Physical state plus per-key musical targets; all policy features are float32."""

    target_active: Float[Tensor, "*batch keys"]
    target_remaining: Float[Tensor, "*batch keys"]
    next_note_valid: Float[Tensor, "*batch keys"]
    next_note_time: Float[Tensor, "*batch keys"]
    next_note_duration: Float[Tensor, "*batch keys"]


def velocity_norm(state: RukaState, _: RukaAction) -> torch.Tensor:
    """Penalizes the magnitude of the hand's linear velocity."""
    return -state.velocity.norm(dim=-1)


class RukaEnvConfig(Config):
    class_name: Literal["src.env.ruka_env:RukaEnv"]

    fingers: Literal[5] = 5
    """The number of fingers to use in the environment."""
    notes_path: str | None = None
    """JSON list of (start seconds, duration seconds, MIDI pitch); None is an empty score."""

    max_episode_length: int = Field(default=600, gt=0)
    """The number of control steps before an environment times out and is reset."""

    reward_weights: dict[str, float] = Field(
        default_factory=lambda: {"velocity_norm": 1.0}
    )
    """The weights for each reward term, keyed by the term's name."""


class RukaEnv(VecEnv):
    def __init__(
        self, sim: VecSystem[RukaState, RukaAction, RukaEnvConfig], cfg: RukaEnvConfig
    ):
        self._sim: VecSystem[RukaState, RukaAction, RukaEnvConfig] = sim
        self.num_envs: int = sim.num_envs
        self.num_actions: int = sim.num_actions
        self.device: torch.device | str = sim.state.position.device
        self.cfg: object = cfg
        self.max_episode_length: int | Tensor = cfg.max_episode_length
        self.episode_length_buf: Tensor = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        notes: list[list[float]] = (
            []
            if cfg.notes_path is None
            else json.loads(Path(cfg.notes_path).read_text())
        )
        self._targets: PianoTargets = PianoTargets(
            notes, sim.state.key_pitches[0].tolist(), self.device
        )
        self._reward: RewardFunction[RukaState, RukaAction] = RewardFunction[
            RukaState, RukaAction
        ](
            [velocity_norm],
            weights=cfg.reward_weights,
            batch_size=self.num_envs,
            device=self.device,
        )

    @override
    def get_observations(self) -> PianoObservation:
        # Physical state belongs to VecSystem; musical targets belong to the task.
        # Clone so subsequent simulator steps cannot mutate a returned observation.
        state = self._sim.state
        observations: dict[str, Tensor] = {
            key: value.clone().float() for key, value in state.items()
        }
        observations.update(self._targets.sample(state.time))
        return PianoObservation(
            **observations,  # pyright: ignore[reportArgumentType]
            batch_size=[self.num_envs],
            device=self.device,
        )

    @override
    def step(self, actions: Tensor):
        action = RukaAction.from_tensor(actions)
        self._sim.step(action)
        rewards = self._reward(self._sim.state, action)
        log = self._reward.log()

        self.episode_length_buf += 1
        time_outs = self.episode_length_buf >= self.max_episode_length
        reset_idx = time_outs.nonzero().flatten()
        if len(reset_idx) > 0:
            # Resetting the system also restarts each environment's score clock.
            self._sim.reset(reset_idx)
            self.episode_length_buf[reset_idx] = 0

        return (
            self.get_observations(),
            rewards,
            time_outs,
            {"time_outs": time_outs, "log": log},
        )

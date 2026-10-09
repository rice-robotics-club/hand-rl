import json
from pathlib import Path
from typing import Literal

import torch
from jaxtyping import Float
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
    """Reward correlates with the magnitude of velocity."""
    return state.velocity.norm(dim=-1)


class RukaEnvConfig(Config):
    class_name: Literal["src.env.ruka_env:RukaEnv"]

    fingers: Literal[5] = 5
    """The number of fingers to use in the environment."""
    notes_path: str | None = None
    """JSON list of (start seconds, duration seconds, MIDI pitch); None is an empty score."""


class RukaEnv(VecEnv):
    def __init__(self, sim: VecSystem[RukaState, RukaAction], cfg: RukaEnvConfig):
        self._sim = sim
        self.num_envs = sim.num_envs
        self.num_actions = 27
        self.device = sim.state.position.device
        self.cfg = cfg
        notes = (
            []
            if cfg.notes_path is None
            else json.loads(Path(cfg.notes_path).read_text())
        )
        self._targets = PianoTargets(
            notes, sim.state.key_pitches[0].tolist(), self.device
        )
        self._reward = RewardFunction[RukaState, RukaAction](
            [velocity_norm], batch_size=sim.num_envs
        )

    def get_observations(self) -> PianoObservation:
        # Physical state belongs to VecSystem; musical targets belong to the task.
        # Clone so subsequent simulator steps cannot mutate a returned observation.
        state = self._sim.state
        observations = {key: value.clone().float() for key, value in state.items()}
        observations.update(self._targets.sample(state.time))
        return PianoObservation(
            **observations, batch_size=[self.num_envs], device=self.device
        )

    def step(self, actions: Tensor):
        actions = RukaAction.from_tensor(actions)
        self._sim.step(actions)
        rewards = self._reward(self._sim.state, actions)
        return (
            self.get_observations(),
            rewards,
            torch.zeros(self.num_envs, device=self.device),
            {"log": self._reward.log()},
        )

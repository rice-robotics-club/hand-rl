from typing import Literal, override

import torch
from rsl_rl.env import VecEnv
from torch import Tensor

from src.base_config import Config
from src.reward import RewardFunction
from src.sim import VecSystem
from src.sim.ruka_sim import RukaAction, RukaState


def velocity_norm(state: RukaState, _: RukaAction) -> torch.Tensor:
    """Reward correlates with the magnitude of velocity."""
    return state.velocity.norm(dim=-1)


class RukaEnvConfig(Config):
    class_name: Literal["src.env.ruka_env:RukaEnv"]

    fingers: int = 5
    """The number of fingers to use in the environment."""


class RukaEnv(VecEnv):
    def __init__(self, sim: VecSystem[RukaState, RukaAction], cfg: RukaEnvConfig):
        self._sim: VecSystem[RukaState, RukaAction] = sim
        self.num_envs: int = sim.num_envs
        self.num_actions: int = 27
        self.device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg: object = cfg
        self._reward: RewardFunction[RukaState, RukaAction] = RewardFunction[
            RukaState, RukaAction
        ]([velocity_norm], batch_size=sim.num_envs)

    @override
    def get_observations(self):
        return self._sim.state

    @override
    def step(self, actions: Tensor):
        actions = RukaAction.from_tensor(actions)
        self._sim.step(actions)
        rewards = self._reward(self._sim.state, actions)
        return (
            self._sim.state,
            rewards,
            torch.zeros(self.num_envs),
            {"log": self._reward.log()},
        )

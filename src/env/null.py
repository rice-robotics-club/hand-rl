from typing import Literal, override

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from src.config import Config
from src.sim import VecSystem
from src.sim.vec_system import AnyVecSystem


class NullEnvConfig(Config):
    class_name: Literal["src.env.null:NullEnv"]

    print_state: bool = False


class NullEnv(VecEnv):
    def __init__(self, system: AnyVecSystem, cfg: NullEnvConfig):
        self._system: AnyVecSystem = system
        self.cfg: object = cfg
        self.num_envs: int = self._system.num_envs
        self.num_actions: int = 1
        self.device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu"
        self._print_state: bool = cfg.print_state

    @override
    def get_observations(self):
        return TensorDict({"dummy": torch.zeros((self.num_envs, 1))})

    @override
    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:  # pyright: ignore[reportMissingTypeArgument]
        self._system.step(actions)
        if self._print_state:
            for field, value in self._system.state.items():
                print(f"{field}: {value}")
        return (
            TensorDict({"dummy": torch.zeros((self.num_envs, 1))}),
            torch.zeros(self.num_envs),
            torch.zeros(self.num_envs),
            {},
        )

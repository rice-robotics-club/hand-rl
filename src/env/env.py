import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from torch import Tensor

from interfaces.vec_sim import VecSim


class Env(VecEnv):
    def __init__(self, sim: VecSim):
        self._sim = sim

    def get_observations(self) -> TensorDict:
        return self._sim.state

    def step(self, actions: Tensor) -> tuple[TensorDict, Tensor, Tensor, dict]:
        self._sim.step(actions)
        return self._sim.state, torch.zeros(), torch.zeros(), {}

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from torch import Tensor

from src.reward.reward import RewardFunction, RewardTerm
from src.sim import VecSystem
from src.sim.ruka_sim import RukaAction, RukaState


class VelocityNorm(RewardTerm):
    """Computes the velocity norm of the state."""

    def __call__(self, state: RukaState) -> torch.Tensor:
        """Returns the velocity norm of the state."""
        return state.velocity.norm(dim=-1)


class RukaEnv(VecEnv):
    def __init__(self, sim: VecSystem[RukaState, RukaAction]):
        self._sim = sim
        self._reward = RewardFunction[RukaState]([VelocityNorm()])
        self.num_envs = sim.num_envs
        self.num_actions = 27
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = {}

    def get_observations(self):
        return self._sim.state

    def step(self, actions: Tensor):
        self._sim.step(RukaAction.from_tensor(actions))
        rewards = self._reward(self._sim.state)
        return (
            self._sim.state,
            rewards,
            torch.zeros(self.num_envs),
            {"log": self._reward.log()},
        )

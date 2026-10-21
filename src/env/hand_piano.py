from typing import Literal, override

import torch
from pydantic import Field
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from torch import Tensor

from src.base_config import Config
from src.reward import RewardFunction
from src.sim import VecSystem
from src.sim.hand_piano import HandPianoAction, HandPianoState


def velocity_norm(state: HandPianoState, _: HandPianoAction) -> torch.Tensor:
    """Penalizes the magnitude of the hand's linear velocity."""
    return -state.velocity.norm(dim=-1)


class HandPianoEnvConfig(Config):
    """Configuration for the HandPianoEnv environment."""

    class_name: Literal["src.env.hand_piano:HandPianoEnv"]

    max_episode_length: int = 250
    """The number of steps before an environment times out and is reset."""

    target_center: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """The center of the region in which reaching targets are sampled."""

    target_range: tuple[float, float, float] = (0.3, 0.3, 0.2)
    """The half-extents of the region in which reaching targets are sampled."""

    reward_weights: dict[str, float] = Field(
        default_factory=lambda: {"target_distance": 1.0, "velocity_norm": 0.1}
    )
    """The weights for each reward term, keyed by the term's name."""


class HandPianoEnv(VecEnv):
    """RL environment for a situation with a hand and a piano.

    As a first task, the hand must move its base to a randomly sampled target position and
    hold still there. We'll use this file to define our reward logic, as well as additional
    processing that does not concern actually setting up our simulation.
    """

    def __init__(
        self,
        system: VecSystem[HandPianoState, HandPianoAction],
        cfg: HandPianoEnvConfig,
    ):
        self._system: VecSystem[HandPianoState, HandPianoAction] = system
        self.cfg: object = cfg
        self.num_envs: int = system.num_envs
        self.num_actions: int = system.num_actions
        self.device: torch.device | str = system.state.device or "cpu"
        self.max_episode_length: int | Tensor = cfg.max_episode_length
        self.episode_length_buf: Tensor = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        self._target_center: Tensor = torch.tensor(
            cfg.target_center, device=self.device
        )
        self._target_range: Tensor = torch.tensor(cfg.target_range, device=self.device)
        self._target: Tensor = torch.zeros((self.num_envs, 3), device=self.device)
        self._sample_targets(torch.arange(self.num_envs, device=self.device))

        self._reward: RewardFunction[HandPianoState, HandPianoAction] = RewardFunction[
            HandPianoState, HandPianoAction
        ](
            [self.target_distance, velocity_norm],
            weights=cfg.reward_weights,
            batch_size=self.num_envs,
            device=self.device,
        )

    def target_distance(self, state: HandPianoState, _: HandPianoAction) -> Tensor:
        """Penalizes the distance between the hand and its target position."""
        return -(state.position - self._target).norm(dim=-1)

    @override
    def get_observations(self) -> TensorDict:
        # The runner keeps a reference to the observations until after the next step, so
        # return a copy rather than the system's state, which is updated in place.
        obs = TensorDict(dict(self._system.state.items()), batch_size=self.num_envs)
        obs["target"] = self._target
        return obs.clone()

    @override
    def step(self, actions: Tensor):
        action = HandPianoAction.from_tensor(actions)
        self._system.step(action)
        rewards = self._reward(self._system.state, action)
        log = self._reward.log()

        self.episode_length_buf += 1
        time_outs = self.episode_length_buf >= self.max_episode_length
        reset_idx = time_outs.nonzero().flatten()
        if len(reset_idx) > 0:
            self._system.reset(reset_idx)
            self._sample_targets(reset_idx)
            self.episode_length_buf[reset_idx] = 0

        return (
            self.get_observations(),
            rewards,
            time_outs,
            {"time_outs": time_outs, "log": log},
        )

    def _sample_targets(self, idx: Tensor):
        noise = torch.rand((len(idx), 3), device=self.device) * 2 - 1
        self._target[idx] = self._target_center + noise * self._target_range

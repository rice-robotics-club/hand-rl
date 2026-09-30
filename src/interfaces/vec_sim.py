from typing import Protocol

from tensordict import TensorDict
from torch import Tensor


class VecSim(Protocol):
    """Interface for a vectorized robotics simulator.

    Allows for abstracting away the simulation setup when defining a rsl_rl VecEnv.
    """

    @property
    def num_envs(self) -> int:
        """Property containing the number of environments in the simulator."""
        raise NotImplementedError

    @property
    def state(self) -> TensorDict:
        """Property containing the current state of the simulator, as a TensorDict."""
        raise NotImplementedError

    def step(self, actions: Tensor) -> None:
        """Steps the simulator forward by one timestep."""
        raise NotImplementedError

    def reset(self, idx: int | Tensor | None = None) -> None:
        """Resets the simulator to its initial state.

        Args:
            idx: Optional index of the environment to reset. If None, resets all environments.
        """
        raise NotImplementedError

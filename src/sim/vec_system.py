"""Defines the interface for a vectorized dynamical system."""

from abc import ABC, abstractmethod
from typing import Protocol, Self

import torch
from tensordict import TensorClass, TensorDict, TypedTensorDict


class VecAction(ABC, TypedTensorDict):
    """Base class for vectorized actions."""

    @classmethod
    @abstractmethod
    def from_tensor(cls, tensor: torch.Tensor) -> Self:
        """Converts a tensor to a VecAction instance.

        Allows defining a mapping from an action tensor returned by a neural network to
        a VecAction instance.
        """
        raise NotImplementedError

    @classmethod
    def from_(cls, action: torch.Tensor | TensorDict | Self) -> Self:
        """Converts an action to a VecAction instance."""
        if isinstance(action, torch.Tensor):
            return cls.from_tensor(action)
        return action


_State = TensorClass | TypedTensorDict | TensorDict
"""Defaults to tensordict, an arbitrary keyed dictionary for use with neural networks."""

_Action = VecAction | TensorClass | TypedTensorDict | torch.Tensor
"""Defaults to torch.Tensor, which is the output of a MLP."""


class VecSystem[
    S: _State = TensorDict,
    A: _Action = torch.Tensor,
](Protocol):
    """Interface for a vectorized dynamical system.

    Possible implementations include simulations, robot controllers, or games. Allows for abstracting away the dynamics when defining a rsl_rl VecEnv, as well as for defining the structure of the system's state and actions/controls.

    Type parameters:
        S: The state type, defaults to TensorDict. Specifying TensorClass or TypedTensorDict also allows defining the state as a structured type.
        A: The action type, defaults to torch.Tensor. Specifying TensorClass or TypedTensorDict also allows defining the action as a structured type.
    """

    @property
    @abstractmethod
    def num_envs(self) -> int:
        """The number of environments in the system."""
        raise NotImplementedError

    @property
    @abstractmethod
    def state(self) -> S:
        """The current state of the system."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: A) -> None:
        """Steps the system forward by one timestep.

        Args:
            action: The action to apply to the system.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, idx: int | torch.Tensor | None = None) -> None:
        """Resets the system to its initial state.

        Args:
            idx: Optional index of the environment to reset. If None, resets all environments.
        """
        raise NotImplementedError

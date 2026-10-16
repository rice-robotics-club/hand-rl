"""Defines the interface for a vectorized dynamical system."""

from abc import ABC, abstractmethod
from typing import Protocol, Self, override

import genesis as gs
import torch
from tensordict import TensorClass, TensorDict, TypedTensorDict

from src.base_config import Config


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


_State = TensorClass | TypedTensorDict | TensorDict | None
"""Defaults to tensordict, an arbitrary keyed dictionary for use with neural networks."""

_Action = VecAction | TensorClass | TypedTensorDict | torch.Tensor | None
"""Defaults to torch.Tensor, which is the output of a MLP."""


class VecSystem[
    S: _State = TensorDict,
    A: _Action = torch.Tensor,
    C: Config = Config,
](Protocol):
    """Interface for a vectorized dynamical system.

    Possible implementations include simulations, robot controllers, or games. Allows for abstracting away the dynamics when defining a rsl_rl VecEnv, as well as for defining the structure of the system's state and actions/controls.

    Type parameters:
        S: The state type, defaults to TensorDict. Specifying TensorClass or TypedTensorDict also allows defining the state as a structured type.
        A: The action type, defaults to torch.Tensor. Specifying TensorClass or TypedTensorDict also allows defining the action as a structured type.
        C: The config type, defaults to Config.
    """

    _cfg: C

    def __init__(self, cfg: C):
        self._cfg = cfg

    @property
    def cfg(self) -> C:
        """The configuration of the system."""
        return self._cfg

    @property
    def num_envs(self) -> int:
        """The number of environments in the system."""
        raise NotImplementedError

    @property
    def num_actions(self) -> int:
        """The size of the flat action tensor accepted by `step`."""
        raise NotImplementedError

    @property
    def state(self) -> S:
        """The current state of the system."""
        raise NotImplementedError

    def step(self, action: A) -> None:
        """Steps the system forward by one timestep.

        Args:
            action: The action to apply to the system.
        """
        raise NotImplementedError

    def reset(self, idx: int | torch.Tensor | None = None) -> None:
        """Resets the system to its initial state.

        Args:
            idx: Optional index of the environment to reset. If None, resets all environments.
        """
        raise NotImplementedError


class _GenesisConfig(Config):
    """Configuration for a Genesis system."""

    sim_options: gs.options.SimOptions = gs.options.SimOptions()
    rigid_options: gs.options.RigidOptions = gs.options.RigidOptions()
    vis_options: gs.options.VisOptions = gs.options.VisOptions()
    viewer_options: gs.options.ViewerOptions = gs.options.ViewerOptions()
    show_viewer: bool = True
    show_FPS: bool = False


class GenesisConfig(Config):
    """Configuration for a Genesis system."""

    genesis: _GenesisConfig = _GenesisConfig()

    num_envs: int = 1
    """Number of environments to run in parallel."""


class GenesisSystem[
    S: _State = TensorDict,
    A: _Action = torch.Tensor,
    C: GenesisConfig = GenesisConfig,
](VecSystem[S, A, C], Protocol):
    """A base class for a dynamic system that is implemented with the Genesis simulator.

    Use this to define a base simulation for an environment.

    The field `_scene` is the scene of the simulation, which you can use to add entities.
    """

    _scene: gs.Scene

    def __init__(self, config: C):
        super().__init__(config)

        gs.init()

        self._scene = (
            gs.Scene(**dict(self.cfg.genesis))  # pyright: ignore[reportAny]
            if self.cfg.genesis
            else gs.Scene()
        )

    @property
    def scene(self) -> gs.Scene:
        """The gs.Scene object associated with this system."""
        return self._scene


class ChildSystem[S: _State, A: _Action, C: Config](VecSystem[S, A, C], Protocol):
    _parent: VecSystem
    _num_envs: int

    def __init__(self, cfg: C, parent: VecSystem):
        super().__init__(cfg)
        self._parent = parent
        self._num_envs = parent.num_envs

    @property
    @override
    def num_envs(self) -> int:
        return self._num_envs

    @override
    def reset(self, idx: int | torch.Tensor | None = None) -> None:
        self._parent.reset(idx)

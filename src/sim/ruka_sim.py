"""Defines the RukaSim simulation, including state, action, and system classes.

Exists as an example implementation of the VecSystem interface.
"""

from typing import TYPE_CHECKING, Literal, override

import genesis as gs
import torch
from jaxtyping import Float
from tensordict import TypedTensorDict

from src.sim.piano import PianoConfig

from .vec_system import GenesisConfig, GenesisSystem, VecAction

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity


class RukaState(TypedTensorDict):
    """Defines an example state to demonstrate a structured State dictionary."""

    position: Float[torch.Tensor, "*batch 3"]
    orientation: Float[torch.Tensor, "*batch 4"]
    velocity: Float[torch.Tensor, "*batch 3"]


class RukaAction(VecAction):
    """Defines an example action to demonstrate the use of VecAction."""

    joint_angles: Float[torch.Tensor, "*batch 27"]

    @classmethod
    @override
    def from_tensor(cls, tensor: Float[torch.Tensor, "*batch 27"]) -> "RukaAction":
        """Converts a tensor to a RukaAction instance."""
        return cls(joint_angles=tensor)


class RukaSimConfig(GenesisConfig):
    class_name: Literal["src.sim.ruka_sim:RukaSim"]

    num_envs: int = 1
    """The number of environments to simulate."""

    piano: PianoConfig
    """The configuration for the piano simulation."""


class RukaSim(GenesisSystem[RukaState, RukaAction, RukaSimConfig]):
    """Defines an example Genesis simulation to demonstrate the use of VecSystem."""

    def __init__(self, config: RukaSimConfig):
        """Initializes the RukaSim with the specified number of environments."""
        super().__init__(config)
        self._num_envs: int = config.num_envs

        gs.init()

        self._scene: gs.Scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                substeps=2,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                # For this locomotion policy, there are usually no more than 20 collision pairs. Setting a low value
                # can save memory. Violating this condition will raise an exception.
                max_collision_pairs=20,
                tolerance=1e-5,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0],
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            show_viewer=True,
        )

        self._plane: RigidEntity = self._scene.add_entity(
            gs.morphs.URDF(
                file="urdf/plane/plane.urdf",
                fixed=True,
            )
        )

        self._robot: RigidEntity = self._scene.add_entity(
            gs.morphs.URDF(
                file="robots/ruka_description/urdf/robot.urdf",
                pos=[0, 0, 0],
                quat=[0, 0, 0, 1],
            ),
        )

        self._scene.build(self._num_envs)

        self._state: RukaState = RukaState(
            position=torch.zeros((self._num_envs, 3)),
            orientation=torch.zeros((self._num_envs, 4)),
            velocity=torch.zeros((self._num_envs, 3)),
            batch_size=self._num_envs,
        )

    @property
    @override
    def scene(self) -> gs.Scene:
        return self._scene

    @property
    @override
    def num_envs(self) -> int:
        """The number of environments in the system."""
        return self._num_envs

    @property
    @override
    def num_actions(self) -> int:
        """One joint angle per robot DOF."""
        return self._robot.n_dofs

    @property
    @override
    def state(self) -> RukaState:
        """The current state of the system."""
        return self._state

    @override
    def step(self, action: RukaAction):
        """Steps the simulation forward by one timestep."""
        self._robot.set_dofs_position(action.joint_angles)
        self._scene.step()
        _ = self._state.position.copy_(torch.as_tensor(self._robot.get_pos()))
        _ = self._state.orientation.copy_(torch.as_tensor(self._robot.get_quat()))
        _ = self._state.velocity.copy_(torch.as_tensor(self._robot.get_vel()))

    @override
    def reset(self, idx: int | torch.Tensor | None = None) -> None:
        """Resets the simulation to the initial state."""
        self._scene.reset(envs_idx=idx)  # type: ignore

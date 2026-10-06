"""Defines the RukaSim simulation, including state, action, and system classes.

Exists as an example implementation of the VecSystem interface.
"""

from typing import Literal

import genesis as gs
import torch
from jaxtyping import Float
from tensordict import TypedTensorDict

from src.base_config import Config

from .vec_system import VecAction, VecSystem


class RukaState(TypedTensorDict):
    """Defines an example state to demonstrate a structured State dictionary."""

    position: Float[torch.Tensor, "*batch 3"]
    orientation: Float[torch.Tensor, "*batch 4"]
    velocity: Float[torch.Tensor, "*batch 3"]


class RukaAction(VecAction):
    """Defines an example action to demonstrate the use of VecAction."""

    joint_angles: Float[torch.Tensor, "*batch 27"]

    @classmethod
    def from_tensor(cls, tensor: Float[torch.Tensor, "*batch 27"]) -> "RukaAction":
        """Converts a tensor to a RukaAction instance."""
        return cls(joint_angles=tensor)


class RukaSimConfig(Config):
    class_name: Literal["src.sim.ruka_sim:RukaSim"]

    num_envs: int = 1
    """The number of environments to simulate."""


class RukaSim(VecSystem[RukaState, RukaAction]):
    """Defines an example Genesis simulation to demonstrate the use of VecSystem."""

    def __init__(self, cfg: RukaSimConfig):
        """Initializes the RukaSim with the specified number of environments."""
        self._num_envs = cfg.num_envs
        self._dt = 1.0 / 60.0

        gs.init()

        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self._dt,
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

        self._scene.add_entity(
            gs.morphs.URDF(
                file="urdf/plane/plane.urdf",
                fixed=True,
            )
        )

        self._robot = self._scene.add_entity(
            gs.morphs.URDF(
                file="robots/ruka_description/urdf/robot.urdf",
                pos=[0, 0, 0],
                quat=[0, 0, 0, 1],
            ),
        )

        self._scene.build(self._num_envs)

        self._state = RukaState(
            position=torch.zeros((self._num_envs, 3)),
            orientation=torch.zeros((self._num_envs, 4)),
            velocity=torch.zeros((self._num_envs, 3)),
            batch_size=self._num_envs,
        )

    @property
    def num_envs(self) -> int:
        """The number of environments in the system."""
        return self._num_envs

    @property
    def state(self) -> RukaState:
        """The current state of the system."""
        return self._state

    def step(self, action: RukaAction):
        """Steps the simulation forward by one timestep."""
        self._robot.set_dofs_position(action.joint_angles)
        self._scene.step()
        self._state.position.copy_(self._robot.get_pos())  # type: ignore
        self._state.orientation.copy_(self._robot.get_quat())
        self._state.velocity.copy_(self._robot.get_vel())

    def reset(self, idx: int | torch.Tensor | None = None):
        """Resets the simulation to the initial state."""
        self._scene.reset(envs_idx=idx)

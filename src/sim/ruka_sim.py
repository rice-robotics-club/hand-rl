"""Defines the RukaSim simulation, including state, action, and system classes.

Exists as an example implementation of the VecSystem interface.
"""

from typing import Literal

import torch
from jaxtyping import Float
from pydantic import Field, field_validator
from tensordict import TypedTensorDict

from src.base_config import Config

from .vec_system import VecAction, VecSystem


class RukaState(TypedTensorDict):
    """Physical snapshot after a control step; see docs/piano_state.md for units.

    Zero-valued sensor slots with validity 0 are unavailable, not measurements.
    Joint arrays follow Genesis DOF order, not the unverified motor mapping.
    """

    position: Float[torch.Tensor, "*batch 3"]
    orientation: Float[torch.Tensor, "*batch 4"]
    velocity: Float[torch.Tensor, "*batch 3"]
    joint_positions: Float[torch.Tensor, "*batch dofs"]
    joint_velocities: Float[torch.Tensor, "*batch dofs"]
    time: Float[torch.Tensor, "*batch 1"]
    control_dt: Float[torch.Tensor, "*batch 1"]
    fingertip_positions: Float[torch.Tensor, "*batch 15"]
    fingertip_valid: Float[torch.Tensor, "*batch 5"]
    key_pitches: Float[torch.Tensor, "*batch keys"]
    key_displacement: Float[torch.Tensor, "*batch keys"]
    key_pressed: Float[torch.Tensor, "*batch keys"]
    key_valid: Float[torch.Tensor, "*batch keys"]


class RukaAction(VecAction):
    """Defines an example action to demonstrate the use of VecAction."""

    joint_angles: Float[torch.Tensor, "*batch 27"]

    @classmethod
    def from_tensor(cls, tensor: Float[torch.Tensor, "*batch 27"]) -> "RukaAction":
        """Converts a tensor to a RukaAction instance."""
        return cls(joint_angles=tensor)


class RukaSimConfig(Config):
    class_name: Literal["src.sim.ruka_sim:RukaSim"]

    num_envs: int = Field(default=1, gt=0)
    """The number of environments to simulate."""
    control_dt: float = Field(default=1.0 / 60.0, gt=0, allow_inf_nan=False)
    """Simulated seconds advanced by each VecSystem.step call."""
    physics_substeps: int = Field(default=2, ge=1)
    """Internal physics steps per control step; not policy decisions."""
    key_pitches: list[int] = Field(
        default_factory=lambda: [60, 62, 64, 65, 67], min_length=1
    )
    """Keyboard observation order, initially C4 through G4 on white keys."""

    @field_validator("key_pitches")
    @classmethod
    def validate_pitches(cls, pitches):
        if len(set(pitches)) != len(pitches) or any(p < 0 or p > 127 for p in pitches):
            raise ValueError("key_pitches must be unique MIDI pitches in 0..127")
        return pitches


class RukaSim(VecSystem[RukaState, RukaAction]):
    """Defines an example Genesis simulation to demonstrate the use of VecSystem."""

    def __init__(self, cfg: RukaSimConfig):
        """Initializes the RukaSim with the specified number of environments."""
        import genesis as gs

        self._num_envs = cfg.num_envs
        self._dt = cfg.control_dt

        gs.init()

        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self._dt,
                substeps=cfg.physics_substeps,
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

        position = self._robot.get_pos()
        device = position.device
        self._steps = torch.zeros((self._num_envs, 1), dtype=torch.int64, device=device)

        def zeros(width):
            return position.new_zeros((self._num_envs, width))

        self._state = RukaState(
            position=position.clone(),
            orientation=self._robot.get_quat().clone(),
            velocity=self._robot.get_vel().clone(),
            joint_positions=self._robot.get_dofs_position().clone(),
            joint_velocities=self._robot.get_dofs_velocity().clone(),
            time=torch.zeros((self._num_envs, 1), dtype=torch.float64, device=device),
            control_dt=torch.full(
                (self._num_envs, 1), self._dt, dtype=torch.float64, device=device
            ),
            fingertip_positions=zeros(15),
            fingertip_valid=zeros(5),
            key_pitches=torch.tensor(
                cfg.key_pitches, device=device, dtype=position.dtype
            )
            .expand(self._num_envs, -1)
            .clone(),
            key_displacement=zeros(len(cfg.key_pitches)),
            key_pressed=zeros(len(cfg.key_pitches)),
            key_valid=zeros(len(cfg.key_pitches)),
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
        self._steps += 1
        self._refresh_state()

    def _refresh_state(self):
        """Refresh real measurements and derive time without cumulative drift."""
        self._state.position.copy_(self._robot.get_pos())  # type: ignore
        self._state.orientation.copy_(self._robot.get_quat())
        self._state.velocity.copy_(self._robot.get_vel())
        self._state.joint_positions.copy_(self._robot.get_dofs_position())
        self._state.joint_velocities.copy_(self._robot.get_dofs_velocity())
        self._state.time.copy_(self._steps.to(torch.float64) * self._dt)

    def reset(self, idx: int | torch.Tensor | None = None):
        """Resets the simulation to the initial state."""
        if isinstance(idx, int):
            idx = torch.tensor([idx], device=self._steps.device)
        self._scene.reset(envs_idx=idx)
        self._steps[slice(None) if idx is None else idx] = 0
        self._refresh_state()

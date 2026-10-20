"""Defines the RukaSim simulation, including state, action, and system classes.

Exists as an example implementation of the VecSystem interface.
"""

from typing import TYPE_CHECKING, Literal, override

import genesis as gs
import torch
from jaxtyping import Float
from pydantic import Field, field_validator
from tensordict import TypedTensorDict

from .vec_system import GenesisConfig, GenesisSystem, VecAction

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

NUM_BASE_DOFS = 6
"""The free joint of the hand base: 3 translational DOFs, then 3 rotational DOFs."""


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
    """Normalized offsets from the robot's default pose, used as PD position targets."""

    joint_angles: Float[torch.Tensor, "*batch 27"]
    """One offset per DOF, in Genesis DOF order (free base first, then joints)."""

    @classmethod
    @override
    def from_tensor(cls, tensor: Float[torch.Tensor, "*batch 27"]) -> "RukaAction":
        """Converts a tensor to a RukaAction instance."""
        return cls(joint_angles=tensor, batch_size=tensor.shape[:-1])


class RukaSimConfig(GenesisConfig):
    class_name: Literal["src.sim.ruka_sim:RukaSim"]

    num_envs: int = Field(default=1, gt=0)
    """The number of environments to simulate."""
    control_dt: float = Field(default=1.0 / 60.0, gt=0, allow_inf_nan=False)
    """Simulated seconds advanced by each VecSystem.step call. Overrides genesis.sim_options.dt."""
    physics_substeps: int = Field(default=2, ge=1)
    """Internal physics steps per control step; not policy decisions. Overrides genesis.sim_options.substeps."""
    key_pitches: list[int] = Field(
        default_factory=lambda: [60, 62, 64, 65, 67], min_length=1
    )
    """Keyboard observation order, initially C4 through G4 on white keys."""

    @field_validator("key_pitches")
    @classmethod
    def validate_pitches(cls, pitches: list[int]) -> list[int]:
        if len(set(pitches)) != len(pitches) or any(p < 0 or p > 127 for p in pitches):
            raise ValueError("key_pitches must be unique MIDI pitches in 0..127")
        return pitches

    initial_position: tuple[float, float, float] = (0.0, 0.0, 0.5)
    """The initial position of the hand base, above the ground plane."""

    action_scale: float = 0.5
    """Scale applied to actions before adding them to the default pose."""

    base_kp: tuple[float, float] = (500.0, 50.0)
    """PD position gains for the base (translation, rotation)."""

    base_kv: tuple[float, float] = (50.0, 5.0)
    """PD velocity gains for the base (translation, rotation)."""


class RukaSim(GenesisSystem[RukaState, RukaAction, RukaSimConfig]):
    """Defines an example Genesis simulation to demonstrate the use of VecSystem."""

    def __init__(self, cfg: RukaSimConfig):
        """Initializes the RukaSim with the specified number of environments."""
        # The control interval is part of the task's timing contract, so it takes
        # precedence over any timing in the scene options.
        sim_options = cfg.genesis.sim_options.model_copy(
            update={"dt": cfg.control_dt, "substeps": cfg.physics_substeps}
        )
        cfg = cfg.model_copy(
            update={"genesis": cfg.genesis.model_copy(update={"sim_options": sim_options})}
        )
        super().__init__(cfg)

        self._num_envs: int = cfg.num_envs
        self._dt: float = cfg.control_dt

        _ = self._scene.add_entity(
            gs.morphs.URDF(
                file="urdf/plane/plane.urdf",
                fixed=True,
            )
        )

        # Gravity compensation lets the floating base hold its pose with PD control,
        # as if the hand were mounted on an arm.
        self._robot: RigidEntity = self._scene.add_entity(
            gs.morphs.URDF(
                file="robots/ruka_description/urdf/robot.urdf",
                pos=cfg.initial_position,
                quat=(0, 0, 0, 1),
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )

        self._scene.build(self._num_envs)

        base_dofs = range(NUM_BASE_DOFS)
        self._robot.set_dofs_kp(
            torch.tensor([cfg.base_kp[0]] * 3 + [cfg.base_kp[1]] * 3), base_dofs
        )
        self._robot.set_dofs_kv(
            torch.tensor([cfg.base_kv[0]] * 3 + [cfg.base_kv[1]] * 3), base_dofs
        )

        self._default_dofs: torch.Tensor = torch.as_tensor(
            self._robot.get_dofs_position()
        )[0].clone()
        lower, upper = self._robot.get_dofs_limit()
        self._dofs_lower: torch.Tensor = torch.as_tensor(lower).clone()
        self._dofs_upper: torch.Tensor = torch.as_tensor(upper).clone()

        position = torch.as_tensor(self._robot.get_pos())
        device = position.device
        self._steps: torch.Tensor = torch.zeros(
            (self._num_envs, 1), dtype=torch.int64, device=device
        )

        def zeros(width: int) -> torch.Tensor:
            return position.new_zeros((self._num_envs, width))

        n_dofs = self._robot.n_dofs
        self._state: RukaState = RukaState(
            position=zeros(3),
            orientation=zeros(4),
            velocity=zeros(3),
            joint_positions=zeros(n_dofs),
            joint_velocities=zeros(n_dofs),
            time=torch.zeros((self._num_envs, 1), dtype=torch.float64, device=device),
            control_dt=torch.full(
                (self._num_envs, 1), self._dt, dtype=torch.float64, device=device
            ),
            fingertip_positions=zeros(15),
            fingertip_valid=zeros(5),
            key_pitches=torch.tensor(cfg.key_pitches, device=device)
            .to(position.dtype)
            .expand(self._num_envs, -1)
            .clone(),
            key_displacement=zeros(len(cfg.key_pitches)),
            key_pressed=zeros(len(cfg.key_pitches)),
            key_valid=zeros(len(cfg.key_pitches)),
            batch_size=self._num_envs,
            device=device,
        )
        self._refresh_state()

    @property
    @override
    def num_envs(self) -> int:
        """The number of environments in the system."""
        return self._num_envs

    @property
    @override
    def num_actions(self) -> int:
        """One PD target per robot DOF."""
        return self._robot.n_dofs

    @property
    @override
    def state(self) -> RukaState:
        """The current state of the system."""
        return self._state

    @override
    def step(self, action: RukaAction):
        """Steps the simulation forward by one timestep."""
        action = RukaAction.from_(action)
        targets = self._default_dofs + self.cfg.action_scale * action.joint_angles
        self._robot.control_dofs_position(
            targets.clamp(self._dofs_lower, self._dofs_upper)
        )
        self._scene.step()
        self._steps += 1
        self._refresh_state()

    def _refresh_state(self):
        """Refresh real measurements and derive time without cumulative drift."""
        _ = self._state.position.copy_(torch.as_tensor(self._robot.get_pos()))
        _ = self._state.orientation.copy_(torch.as_tensor(self._robot.get_quat()))
        _ = self._state.velocity.copy_(torch.as_tensor(self._robot.get_vel()))
        _ = self._state.joint_positions.copy_(
            torch.as_tensor(self._robot.get_dofs_position())
        )
        _ = self._state.joint_velocities.copy_(
            torch.as_tensor(self._robot.get_dofs_velocity())
        )
        _ = self._state.time.copy_(self._steps.to(torch.float64) * self._dt)

    @override
    def reset(self, idx: int | torch.Tensor | None = None):
        """Resets the simulation to the initial state."""
        if isinstance(idx, int):
            idx = torch.tensor([idx], device=self._steps.device)
        self._scene.reset(envs_idx=idx)
        self._steps[slice(None) if idx is None else idx] = 0
        self._refresh_state()

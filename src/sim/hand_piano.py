"""Defines the HandPiano simulation, including state, action, and system classes.

A floating RUKA hand is driven by PD position targets above a piano.
"""

from typing import TYPE_CHECKING, Literal, NotRequired, override

import genesis as gs
import torch
from jaxtyping import Float
from rsl_rl.utils import resolve_callable
from tensordict import TypedTensorDict

from src.sim.piano import PianoConfig, PianoState

from .vec_system import GenesisConfig, GenesisSystem, VecAction, VecSystem

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

NUM_BASE_DOFS = 6
"""The free joint of the hand base: 3 translational DOFs, then 3 rotational DOFs."""


class HandPianoState(PianoState, TypedTensorDict):
    """The state of the hand, along with the state of the piano keys."""

    position: Float[torch.Tensor, "*batch 3"]
    orientation: Float[torch.Tensor, "*batch 4"]
    velocity: Float[torch.Tensor, "*batch 3"]
    joint_positions: Float[torch.Tensor, "*batch 21"]
    fingertip_key_distances: NotRequired[Float[torch.Tensor, "*batch 88"]]
    """Optional measured fingertip-to-press-point distances in meters, in pressed-channel order.

    Not populated until fingertip assignments and piano contact geometry are verified.
    """


type PianoSystem = VecSystem[PianoState, None]


class HandPianoAction(VecAction):
    """Normalized offsets from the hand's default pose, used as PD position targets."""

    base_pose: Float[torch.Tensor, "*batch 6"]
    """Offsets for the base position (x, y, z) and rotation (roll, pitch, yaw)."""

    joint_positions: Float[torch.Tensor, "*batch 21"]
    """Offsets for the wrist and finger joints."""

    @classmethod
    @override
    def from_tensor(cls, tensor: Float[torch.Tensor, "*batch 27"]) -> "HandPianoAction":
        """Converts a tensor to a HandPianoAction instance."""
        return cls(
            base_pose=tensor[..., :NUM_BASE_DOFS],
            joint_positions=tensor[..., NUM_BASE_DOFS:],
            batch_size=tensor.shape[:-1],
        )


class HandPianoConfig(GenesisConfig):
    class_name: Literal["src.sim.hand_piano:HandPiano"]

    num_envs: int = 1
    """The number of environments to simulate."""

    workspace: float = 2.0
    """Base position targets are clamped to [-workspace, workspace] on each axis."""

    initial_position: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """The initial position of the hand base."""

    action_scale: float = 0.5
    """Scale applied to actions before adding them to the default pose."""

    base_kp: tuple[float, float] = (500.0, 50.0)
    """PD position gains for the base (translation, rotation)."""

    base_kv: tuple[float, float] = (50.0, 5.0)
    """PD velocity gains for the base (translation, rotation)."""

    piano: PianoConfig
    """The configuration for the piano simulation."""


class HandPiano(GenesisSystem[HandPianoState, HandPianoAction, HandPianoConfig]):
    """A floating hand above a piano, simulated with Genesis."""

    def __init__(self, cfg: HandPianoConfig):
        """Initializes the HandPiano with the specified number of environments."""
        super().__init__(cfg)

        self._num_envs: int = cfg.num_envs

        # Gravity compensation lets the floating base hold its pose with PD control,
        # as if the hand were mounted on an arm.
        self._hand: RigidEntity = self._scene.add_entity(
            gs.morphs.URDF(
                file="robots/ruka_description/urdf/robot.urdf",
                pos=cfg.initial_position,
                quat=(0, 0, 0, 1),
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )

        self._piano: PianoSystem = resolve_callable(cfg.piano.class_name)(
            cfg.piano, self
        )

        self._scene.build(self._num_envs)

        base_dofs = range(NUM_BASE_DOFS)
        self._hand.set_dofs_kp(
            torch.tensor([cfg.base_kp[0]] * 3 + [cfg.base_kp[1]] * 3), base_dofs
        )
        self._hand.set_dofs_kv(
            torch.tensor([cfg.base_kv[0]] * 3 + [cfg.base_kv[1]] * 3), base_dofs
        )

        self._default_dofs: torch.Tensor = self._hand.get_dofs_position()[0].clone()
        lower, upper = self._hand.get_dofs_limit()
        self._dofs_lower: torch.Tensor = lower.clone()
        self._dofs_upper: torch.Tensor = upper.clone()
        self._dofs_lower[:3] = -cfg.workspace
        self._dofs_upper[:3] = cfg.workspace

        self._state: HandPianoState = HandPianoState(
            position=torch.zeros((self._num_envs, 3), device=gs.device),
            orientation=torch.zeros((self._num_envs, 4), device=gs.device),
            velocity=torch.zeros((self._num_envs, 3), device=gs.device),
            joint_positions=torch.zeros(
                (self._num_envs, self.num_actions - NUM_BASE_DOFS), device=gs.device
            ),
            pressed=torch.zeros((self._num_envs, 88), device=gs.device),
            batch_size=self._num_envs,
            device=gs.device,
        )
        self.reset()

    @property
    @override
    def num_envs(self) -> int:
        """The number of environments in the system."""
        return self._num_envs

    @property
    @override
    def num_actions(self) -> int:
        """One PD target per hand DOF."""
        return self._hand.n_dofs

    @property
    @override
    def state(self) -> HandPianoState:
        """The current state of the system."""
        return self._state

    @override
    def step(self, action: HandPianoAction):
        """Steps the simulation forward by one timestep."""
        action = HandPianoAction.from_(action)
        offsets = torch.cat([action.base_pose, action.joint_positions], dim=-1)
        targets = self._default_dofs + self.cfg.action_scale * offsets
        self._hand.control_dofs_position(
            targets.clamp(self._dofs_lower, self._dofs_upper)
        )
        self._scene.step()
        self._piano.step(None)
        self._update_state()

    @override
    def reset(self, idx: int | torch.Tensor | None = None):
        """Resets the simulation to the initial state."""
        self._scene.reset(envs_idx=idx)
        self._piano.reset(idx)
        self._update_state()

    def _update_state(self):
        _ = self._state.position.copy_(torch.as_tensor(self._hand.get_pos()))
        _ = self._state.orientation.copy_(self._hand.get_quat())
        _ = self._state.velocity.copy_(self._hand.get_vel())
        _ = self._state.joint_positions.copy_(
            torch.as_tensor(self._hand.get_dofs_position())[:, NUM_BASE_DOFS:]
        )
        _ = self._state.pressed.copy_(self._piano.state.pressed)

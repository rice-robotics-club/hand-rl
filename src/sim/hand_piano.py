"""Defines the HandPiano simulation, including state, action, and system classes.

A floating RUKA hand is driven by PD position targets above a piano.
"""

from typing import TYPE_CHECKING, Literal, Protocol, override

import genesis as gs
import genesis.utils.geom as gu
import torch
from jaxtyping import Float
from pydantic import Field
from rsl_rl.utils import resolve_callable
from tensordict import TypedTensorDict

from src.base_config import Config
from src.music.keyboard import NUM_KEYS
from src.sim.piano import Piano, PianoConfig, PianoState
from src.sim.vec_system import GenesisConfig, GenesisSystem, VecAction, VecSystem
from src.util.geometry import (
    FREE_JOINT_DOFS,
    TRANSLATION_DOFS,
    free_joint_values,
    rotvec_to_quat,
)

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

URDF_PATH = "robots/ruka_description/urdf/robot.urdf"

NUM_BASE_DOFS = FREE_JOINT_DOFS
"""The free joint of the hand base: 3 translational DOFs, then 3 rotational DOFs."""

NUM_JOINTS = 21
"""Wrist and finger joints. Checked against the URDF when the scene is built."""

NUM_ACTIONS = NUM_BASE_DOFS + NUM_JOINTS
"""One PD target per DOF: the base, then the joints."""

FINGERTIP_LINKS = (
    "thumb___joint_3",
    "finger___joint_3",
    "finger___joint_3_2",
    "finger___joint_3_3",
    "pinky___joint_3",
)
"""The distal link of each finger, thumb to pinky (finger numbers 1 to 5)."""

NUM_FINGERS = len(FINGERTIP_LINKS)

PIANIST_POSTURE: dict[str, float] = {
    # Fingers curved at every joint. Signs differ per joint because the URDF's joint
    # axes do; each value bends its finger toward the palm.
    "index_mcp": -0.6,
    "index_pip": 0.7,
    "index_dip": -0.35,
    "mid_mcp": 0.6,
    "mid_pip": -0.7,
    "mid_dip": 0.35,
    "ring_mcp": 0.6,
    "ring_pip": 0.7,
    "ring_dip": 0.35,
    "pinky_mcp": 0.55,
    "pinky_pip": 0.6,
    "pinky_dip": 0.35,
    # Thumb tucked in next to the index finger, its tip level with the others.
    "thumb_cmc": -0.45,
    "thumb_mcp": 0.15,
    "thumb_ip": 0.3,
}
"""Joint angles (radians) for a pianist's hand: curved fingers with level fingertips.

With the default base pose, the five fingertips sit in a row 4-5 cm high (about 2 cm
above the white keys), over the front of the keys, about 2 cm apart.
"""


class HandPianoState(PianoState, TypedTensorDict):
    """The state of the hand, along with the state of the piano keys."""

    position: Float[torch.Tensor, "*batch 3"]
    orientation: Float[torch.Tensor, "*batch 4"]
    velocity: Float[torch.Tensor, "*batch 3"]
    joint_positions: Float[torch.Tensor, f"*batch {NUM_JOINTS}"]
    target_offsets: Float[torch.Tensor, f"*batch {NUM_ACTIONS}"]
    """The smoothed, normalized offsets from the default pose that the PD targets use.

    Base DOFs first, then joints. With action_smoothing below 1 these lag the actions,
    so they are part of the state the policy needs.
    """
    fingertip_positions: Float[torch.Tensor, f"*batch {NUM_FINGERS} 3"]
    """World-frame fingertip positions in meters, thumb to pinky."""
    fingertip_key_distances: Float[torch.Tensor, f"*batch {NUM_FINGERS} {NUM_KEYS}"]
    """Distance in meters from each fingertip (thumb to pinky) to each key's press point.

    Keys are in pressed-channel order. Which finger should play which key is decided
    by the environment, which knows the score.
    """


class HandPianoAction(VecAction):
    """Normalized offsets from the hand's default pose, used as PD position targets."""

    base_pose: Float[torch.Tensor, f"*batch {NUM_BASE_DOFS}"]
    """Offsets for the base position (x, y, z), then a rotation vector for the base.

    The rotation vector is expressed in the hand's default (initial) frame, so each
    component turns the hand about one of its own axes from the default orientation.
    """

    joint_positions: Float[torch.Tensor, f"*batch {NUM_JOINTS}"]
    """Offsets for the wrist and finger joints."""

    @classmethod
    @override
    def from_tensor(
        cls, tensor: Float[torch.Tensor, f"*batch {NUM_ACTIONS}"]
    ) -> "HandPianoAction":
        """Converts a tensor to a HandPianoAction instance."""
        return cls(
            base_pose=tensor[..., :NUM_BASE_DOFS],
            joint_positions=tensor[..., NUM_BASE_DOFS:],
            batch_size=tensor.shape[:-1],
        )


class HandPianoSystem(VecSystem[HandPianoState, HandPianoAction, Config], Protocol):
    """What HandPianoEnv needs from a hand-and-piano system, simulated or real.

    HandPiano implements it in Genesis.
    """

    @property
    def default_joint_positions(self) -> Float[torch.Tensor, f"{NUM_JOINTS}"]:
        """The wrist and finger angles the hand starts and resets in."""
        ...

    @property
    def default_orientation(self) -> Float[torch.Tensor, "4"]:
        """The (w, x, y, z) quaternion the hand base starts and resets in."""
        ...

    def show_key_colors(self, colors: Float[torch.Tensor, f"{NUM_KEYS} 4"]) -> None:
        """Colors environment 0's keys with RGBA colors, e.g. in a viewer.

        Zero alpha leaves a key uncolored. Systems without a display ignore this.
        """
        ...


class HandPianoConfig(GenesisConfig):
    class_name: Literal["src.sim.hand_piano:HandPiano"]

    num_envs: int = 1
    """The number of environments to simulate."""

    workspace_lower: tuple[float, float, float] = (0.12, -0.7, 0.0)
    """Lower bounds in meters for the base position targets, per axis."""

    workspace_upper: tuple[float, float, float] = (0.35, 0.7, 0.3)
    """Upper bounds in meters for the base position targets, per axis.

    The default box covers the whole keyboard (y from -0.61 to 0.61) and lets the
    fingertips reach from behind the black keys to just in front of the keys, from
    pressing height to about 20 cm above them.
    """

    initial_position: tuple[float, float, float] = (0.22, -0.015, 0.09)
    """The initial position of the hand base in meters.

    The piano's keys run along y (bass at -y) with the player on the +x side and key
    tops about 2-3.5 cm above z = 0. The default puts the fingertips over the front of
    the keys, about 2 cm above the white keys in the default posture, with the thumb to
    pinky centered on C4 to G4.
    """

    initial_orientation: tuple[float, float, float] = (90.0, 0.0, -90.0)
    """The initial orientation of the hand base as extrinsic x-y-z Euler angles in degrees.

    The default points the fingers away from the player (-x), with the palm facing
    down onto the keys and the thumb toward the bass (-y), as for a right hand.
    """

    initial_joint_positions: dict[str, float] = Field(
        default_factory=lambda: dict(PIANIST_POSTURE)
    )
    """Joint angles in radians, by joint name, for the default pose; others are 0.

    The hand starts and resets here, and zero actions hold it here.
    """

    action_scale: float = 0.5
    """Scale applied to joint actions before adding them to the default pose.

    Actions are clipped to [-1, 1] first, so each target is within this distance (in
    radians) of the default pose.
    """

    base_action_scale: tuple[float, float] | None = None
    """Scales for the base actions: (translation in meters, rotation in radians).

    None uses action_scale for both.
    """

    action_smoothing: float = Field(default=1.0, gt=0.0, le=1.0)
    """Weight of each new action in a moving average of actions; 1 disables smoothing.

    0.25 at 60 Hz settles about 63% of the way to a new action in about 4 steps
    (65 ms), so targets cannot jump from one step to the next.
    """

    base_kp: tuple[float, float] = (500.0, 50.0)
    """PD position gains for the base (translation, rotation)."""

    base_kv: tuple[float, float] = (50.0, 5.0)
    """PD velocity gains for the base (translation, rotation)."""

    base_force_limit: tuple[float, float] | None = None
    """Caps on the base's PD force in newtons and torque in newton-meters, or None.

    The stiff base PD can otherwise drive the hand through the keys; a key needs only
    about 1 N to press. With 4 substeps, 10 N kept a hard push's penetration to 2 mm.
    """

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
                file=URDF_PATH,
                pos=cfg.initial_position,
                euler=cfg.initial_orientation,
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )

        self._piano: Piano = resolve_callable(cfg.piano.class_name)(cfg.piano, self)

        self._scene.build(self._num_envs)
        if self._hand.n_dofs != NUM_ACTIONS:
            raise ValueError(
                f"{URDF_PATH} has {self._hand.n_dofs} DOFs; expected {NUM_ACTIONS}"
            )

        base_dofs = range(NUM_BASE_DOFS)
        self._hand.set_dofs_kp(torch.tensor(free_joint_values(*cfg.base_kp)), base_dofs)
        self._hand.set_dofs_kv(torch.tensor(free_joint_values(*cfg.base_kv)), base_dofs)
        if cfg.base_force_limit is not None:
            limit = torch.tensor(free_joint_values(*cfg.base_force_limit))
            self._hand.set_dofs_force_range(-limit, limit, base_dofs)

        self._fingertip_links: list[int] = [
            self._hand.get_link(name).idx_local for name in FINGERTIP_LINKS
        ]
        self._fingertip_offsets: torch.Tensor = self._measure_fingertip_offsets()
        self._press_points: torch.Tensor = self._piano.press_points()

        self._default_dofs: torch.Tensor = self._hand.get_dofs_position()[0].clone()
        for name, angle in cfg.initial_joint_positions.items():
            self._default_dofs[self._hand.get_joint(name).dofs_idx_local[0]] = angle
        translation, rotation = cfg.base_action_scale or (cfg.action_scale,) * 2
        self._action_scales: torch.Tensor = torch.tensor(
            free_joint_values(translation, rotation) + [cfg.action_scale] * NUM_JOINTS,
            device=gs.device,
        )
        # Euler offsets are degenerate in the default pose (Genesis's free-joint Euler
        # angles are in gimbal lock when the fingers point along -x), so base rotations
        # are composed as quaternions and converted back to Euler targets.
        self._default_quat: torch.Tensor = self._hand.get_quat()[0].clone()
        lower, upper = self._hand.get_dofs_limit()
        self._dofs_lower: torch.Tensor = lower.clone()
        self._dofs_upper: torch.Tensor = upper.clone()
        position = slice(TRANSLATION_DOFS)
        self._dofs_lower[position] = torch.tensor(cfg.workspace_lower, device=gs.device)
        self._dofs_upper[position] = torch.tensor(cfg.workspace_upper, device=gs.device)

        def zeros(*shape: int) -> torch.Tensor:
            return torch.zeros((self._num_envs, *shape), device=gs.device)

        self._state: HandPianoState = HandPianoState(
            position=zeros(3),
            orientation=zeros(4),
            velocity=zeros(3),
            joint_positions=zeros(NUM_JOINTS),
            target_offsets=zeros(NUM_ACTIONS),
            fingertip_positions=zeros(NUM_FINGERS, 3),
            fingertip_key_distances=zeros(NUM_FINGERS, NUM_KEYS),
            pressed=zeros(NUM_KEYS),
            hit_velocity=zeros(NUM_KEYS),
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
        return NUM_ACTIONS

    @property
    @override
    def state(self) -> HandPianoState:
        """The current state of the system."""
        return self._state

    @property
    def default_joint_positions(self) -> Float[torch.Tensor, f"{NUM_JOINTS}"]:
        """The wrist and finger angles of the default pose (the pianist posture)."""
        return self._default_dofs[NUM_BASE_DOFS:]

    @property
    def default_orientation(self) -> Float[torch.Tensor, "4"]:
        """The (w, x, y, z) quaternion of the default base orientation."""
        return self._default_quat

    @override
    def step(self, action: HandPianoAction):
        """Steps the simulation forward by one timestep."""
        action = HandPianoAction.from_(action)
        # Policies sample from an unbounded Gaussian, so clip to the normalized range.
        actions = torch.cat([action.base_pose, action.joint_positions], dim=-1)
        smoothed = self._state.target_offsets
        _ = smoothed.lerp_(actions.clamp(-1.0, 1.0), self.cfg.action_smoothing)
        offsets = self._action_scales * smoothed
        targets = self._default_dofs + offsets
        rotation = gu.transform_quat_by_quat(
            rotvec_to_quat(offsets[..., TRANSLATION_DOFS:NUM_BASE_DOFS]),
            self._default_quat.expand(*offsets.shape[:-1], 4),
        )
        targets[..., TRANSLATION_DOFS:NUM_BASE_DOFS] = torch.as_tensor(
            gu.quat_to_xyz(rotation)
        )
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
        # The scene resets to the built (straight-fingered) pose; move to the default.
        envs = torch.arange(self._num_envs, device=gs.device) if idx is None else idx
        envs = torch.as_tensor(envs, device=gs.device).reshape(-1)
        self._hand.set_dofs_position(
            self._default_dofs.expand(len(envs), -1), envs_idx=envs, zero_velocity=True
        )
        self._state.target_offsets[envs] = 0
        self._piano.reset(idx)
        self._update_state()

    def show_key_colors(self, colors: Float[torch.Tensor, f"{NUM_KEYS} 4"]):
        """Colors environment 0's piano keys in the viewer; see Piano.show_key_colors."""
        self._piano.show_key_colors(colors)

    def _update_state(self):
        _ = self._state.position.copy_(torch.as_tensor(self._hand.get_pos()))
        _ = self._state.orientation.copy_(self._hand.get_quat())
        _ = self._state.velocity.copy_(self._hand.get_vel())
        _ = self._state.joint_positions.copy_(
            torch.as_tensor(self._hand.get_dofs_position())[:, NUM_BASE_DOFS:]
        )
        tips = self._fingertip_positions()
        _ = self._state.fingertip_positions.copy_(tips)
        _ = self._state.fingertip_key_distances.copy_(
            torch.cdist(tips, self._press_points.expand(tips.shape[0], -1, -1))
        )
        _ = self._state.pressed.copy_(self._piano.state.pressed)
        _ = self._state.hit_velocity.copy_(self._piano.state.hit_velocity)

    def _measure_fingertip_offsets(self) -> Float[torch.Tensor, f"{NUM_FINGERS} 3"]:
        """Finds the tip of each distal link, in that link's frame.

        The tip is the collision vertex farthest from the link origin, which sits at the
        finger's last joint. Measured once, from environment 0.
        """
        pos, quat = self._fingertip_link_poses()
        offsets: list[torch.Tensor] = []
        for i, link in enumerate(self._fingertip_links):
            n_verts = self._hand.links[link].n_verts
            verts = torch.as_tensor(self._hand.links[link].get_verts())
            verts = verts.reshape(-1, n_verts, 3)[0]
            inverse = torch.as_tensor(gu.inv_quat(quat[0, i])).expand(n_verts, 4)
            local = torch.as_tensor(gu.transform_by_quat(verts - pos[0, i], inverse))
            offsets.append(local[local.norm(dim=-1).argmax()])
        return torch.stack(offsets)

    def _fingertip_positions(self) -> Float[torch.Tensor, f"envs {NUM_FINGERS} 3"]:
        """World-frame fingertip positions in every environment."""
        pos, quat = self._fingertip_link_poses()
        offsets = self._fingertip_offsets.expand_as(pos)
        return pos + torch.as_tensor(gu.transform_by_quat(offsets, quat))

    def _fingertip_link_poses(
        self,
    ) -> tuple[
        Float[torch.Tensor, f"envs {NUM_FINGERS} 3"],
        Float[torch.Tensor, f"envs {NUM_FINGERS} 4"],
    ]:
        """Position and orientation of each finger's distal link, per environment."""
        pos = torch.as_tensor(self._hand.get_links_pos(self._fingertip_links))
        quat = torch.as_tensor(self._hand.get_links_quat(self._fingertip_links))
        return pos.reshape(-1, NUM_FINGERS, 3), quat.reshape(-1, NUM_FINGERS, 4)

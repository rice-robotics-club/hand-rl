"""Reward terms for how a robot moves and holds itself, for any robot and environment.

Each term is a pure function `term(state, action) -> (envs,) tensor` reading
MotionState. Shapes use named dimensions because they depend on the robot.
"""

from jaxtyping import Float
from tensordict import TensorDictBase, TypedTensorDict
from torch import Tensor

from src.reward.shaping import bump
from src.sim.vec_system import VecAction


class MotionState(TypedTensorDict):
    """How far the robot is from its default pose, and what it last did."""

    previous_action: Float[Tensor, "*batch actions"]
    """The action before the one being judged (rewards) or chosen (observations).

    Clipped to [-1, 1], like the actions the system applies. Zero after a reset.
    """
    posture_error: Float[Tensor, "*batch joints"]
    """Joint angles minus the default posture, in radians."""
    orientation_error: Float[Tensor, "*batch 1"]
    """Radians between the base orientation and its default orientation."""


def flat_action(action: VecAction) -> Tensor:
    """The action as one tensor clipped to [-1, 1], the normalized action range."""
    return action.to_tensor().clamp(-1, 1)


def action_rate(state: MotionState, action: VecAction) -> Tensor:
    """Minus the mean squared change between consecutive actions; discourages jerk."""
    return -(flat_action(action) - state.previous_action).square().mean(-1)


def finger_posture(state: MotionState, _: TensorDictBase, *, scale: float) -> Tensor:
    """1 in the default posture, falling to 1/e at an RMS joint error of `scale` radians."""
    return bump(state.posture_error.square().mean(-1).sqrt(), scale)


def hand_orientation(state: MotionState, _: TensorDictBase, *, scale: float) -> Tensor:
    """1 in the default orientation, falling to 1/e at a rotation of `scale` radians."""
    return bump(state.orientation_error[..., 0], scale)

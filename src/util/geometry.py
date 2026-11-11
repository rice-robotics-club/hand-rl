"""Rotations and free-joint helpers shared by simulated robots.

Quaternions are (w, x, y, z), as in Genesis.
"""

import torch
from jaxtyping import Float
from torch import Tensor

TRANSLATION_DOFS = 3
"""x, y, z."""

ROTATION_DOFS = 3
"""Rotation about x, y, z."""

FREE_JOINT_DOFS = TRANSLATION_DOFS + ROTATION_DOFS
"""A floating base's free joint: translation first, then rotation."""


def free_joint_values(translation: float, rotation: float) -> list[float]:
    """One value per free-joint DOF: `translation` for x, y, z, then `rotation`."""
    return [translation] * TRANSLATION_DOFS + [rotation] * ROTATION_DOFS


def rotvec_to_quat(rotvec: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch 4"]:
    """Converts rotation vectors to quaternions, including zero rotations."""
    angle = rotvec.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    # sin(half) / angle tends to 1/2 as the angle goes to zero.
    scale = torch.where(angle > 1e-8, half.sin() / angle.clamp_min(1e-8), 0.5)
    return torch.cat([half.cos(), scale * rotvec], dim=-1)


def rotation_angle(
    orientation: Float[Tensor, "*batch 4"], reference: Float[Tensor, "*batch 4"]
) -> Float[Tensor, "*batch 1"]:
    """Radians of rotation between two orientations."""
    dot = (orientation * reference).sum(-1, keepdim=True).abs().clamp(max=1.0)
    return 2 * torch.acos(dot)

"""Which finger should play each wanted key.

Fingers are numbered 0 (thumb) to 4 (pinky); NO_FINGER (-1) means none. Distances
are (envs, fingers, keys) tensors of meters from each fingertip to each key.
"""

import functools
import itertools

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
"""A hand's fingers, in finger-number order."""

HOME_PITCHES = (60, 62, 64, 65, 67)
"""C4 D4 E4 F4 G4: the default key under each finger, thumb to pinky."""

NO_FINGER = -1
"""Finger number meaning that no finger plays a key."""


def closest_fingers(
    distances: Float[Tensor, "envs fingers keys"],
    active: Float[Tensor, "envs keys"] | Bool[Tensor, "envs keys"],
) -> Int[Tensor, "envs keys"]:
    """Gives each active key its own finger, with the least total fingertip distance.

    Returns the finger for each key, NO_FINGER for inactive keys. Every way of giving the active keys distinct fingers is checked,
    which is cheap for one hand (at most 5! = 120). If more keys are active than there
    are fingers, the extra keys get no finger.
    """
    envs, fingers, keys = distances.shape
    slots = min(fingers, keys)
    device = distances.device
    active = active.bool()

    # Up to `slots` active keys per environment (topk sorts active keys first).
    slot_keys = active.float().topk(slots, dim=-1).indices  # (envs, slots)
    slot_used = active.gather(-1, slot_keys)
    slot_distances = distances.gather(
        -1, slot_keys.unsqueeze(1).expand(-1, fingers, -1)
    )

    # Every ordering of fingers over the slots, and the total distance of each.
    orderings = _orderings(fingers, slots, device)
    cost = slot_distances[:, orderings, torch.arange(slots, device=device)]
    best = (cost * slot_used.unsqueeze(1)).sum(-1).argmin(-1)

    slot_fingers = torch.where(slot_used, orderings[best], NO_FINGER)
    result = torch.full((envs, keys), NO_FINGER, device=device)
    return result.scatter(-1, slot_keys, slot_fingers)


@functools.cache
def _orderings(
    fingers: int, slots: int, device: torch.device
) -> Int[Tensor, "orderings slots"]:
    """Every way to give `slots` keys distinct fingers, built once per device."""
    return torch.tensor(
        list(itertools.permutations(range(fingers), slots)), device=device
    )


def home_fingers(
    active: Float[Tensor, "envs keys"], finger_keys: Int[Tensor, " fingers"]
) -> Int[Tensor, "envs keys"]:
    """The finger whose home key each active key is; else NO_FINGER.

    `finger_keys` holds the key under each finger, thumb to pinky, in the hand's home
    (starting) position. This is a fixed five-finger position: a key that isn't
    under a finger gets no finger.
    """
    home = torch.full((active.shape[-1],), NO_FINGER, device=active.device)
    home[finger_keys] = torch.arange(len(finger_keys), device=active.device)
    return torch.where(active.bool(), home, NO_FINGER)


def finger_distances(
    distances: Float[Tensor, "envs fingers keys"], fingers: Int[Tensor, "envs keys"]
) -> Float[Tensor, "envs keys"]:
    """Distance from each key to its finger, or to the nearest finger if it has none.

    `fingers` comes from closest_fingers or home_fingers. Every key gets a real
    measurement, in meters.
    """
    chosen = distances.gather(1, fingers.clamp_min(0).unsqueeze(1)).squeeze(1)
    return torch.where(fingers != NO_FINGER, chosen, distances.min(1).values)

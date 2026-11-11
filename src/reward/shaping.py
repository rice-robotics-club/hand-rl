"""Shaping functions that turn an error or distance into a bounded reward."""

import torch
from torch import Tensor


def bump(error: Tensor, scale: float) -> Tensor:
    """exp(-(error / scale)^2): 1 at zero error, 1/e at `scale`."""
    return torch.exp(-((error / scale) ** 2))


def closeness(distance: Tensor, scale: float) -> Tensor:
    """1 - tanh(distance / scale): 1 at zero distance, 0.24 at `scale`.

    Falls off more slowly than `bump`, so it still gives a signal far away. With
    scale 0.1 m: 0.90 at 1 cm, 0.54 at 5 cm, 0.24 at 10 cm, 0.01 at 25 cm.
    """
    return 1 - torch.tanh(distance / scale)

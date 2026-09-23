import genesis as gs
import torch
from genesis.utils.geom import (
    inv_quat,
    quat_to_xyz,
    transform_by_quat,
    transform_quat_by_quat,
)
from tensordict import TensorDict

from src.config import (
    CommandConfig,
    DomainRandConfig,
    EnvConfig,
    ObsConfig,
    RewardConfig,
)
from src.env.genesis import get_or_default

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(
        size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device
    ) + lower

''' 
Base Genesis environment to put stuff in. 
This can and should be extended for specific environments, 
i.e. environments with specific physics models or background props
'''
class BaseEnvironment:
    def __init__(self):
        self.
from typing import TYPE_CHECKING, Literal, override

import genesis as gs
import torch
from jaxtyping import Float
from robopianist.models.piano import piano_mjcf
from tensordict import TypedTensorDict

from src.base_config import Config
from src.sim.vec_system import ChildSystem, GenesisSystem

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity


class PianoState(TypedTensorDict):
    pressed: Float[torch.Tensor, "*batch 88"]
    """The piano keys that are currently pressed, with values in [0.0, 1.0]."""


class PianoAction(TypedTensorDict):
    sustain: Float[torch.Tensor, "*batch 1"]
    """An input corresponding to pressing the sustain pedal on a piano."""


class PianoConfig(Config):
    class_name: Literal["src.sim.piano:Piano"]


class Piano(ChildSystem[PianoState, None, PianoConfig]):
    def __init__(self, cfg: PianoConfig, parent: GenesisSystem):
        super().__init__(cfg, parent)
        self._scene: gs.Scene = parent.scene

        self._piano: RigidEntity = self._scene.add_entity(
            gs.morphs.MJCF(file=piano_mjcf.build().to_xml_string())
        )

        self._state: PianoState = PianoState(
            pressed=torch.zeros((self._num_envs, 88), device=gs.device),
            batch_size=self._num_envs,
            device=gs.device,
        )
        self._key_travel: torch.Tensor | None = None

    @property
    @override
    def num_actions(self) -> int:
        """The piano is driven by contact with the parent system, not by actions."""
        return 0

    @property
    @override
    def state(self) -> PianoState:
        return self._state

    @override
    def step(self, action: None):
        if self._key_travel is None:
            # Joint limits are only available once the parent has built the scene.
            self._key_travel = self._piano.get_dofs_limit()[1]
        positions: torch.Tensor = self._piano.get_dofs_position()
        _ = self._state.pressed.copy_((positions / self._key_travel).clamp(0.0, 1.0))

    @override
    def reset(self, idx: int | torch.Tensor | None = None) -> None:
        # The parent owns and resets the shared scene; only refresh our state here.
        self.step(None)

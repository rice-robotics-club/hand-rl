import re
from typing import TYPE_CHECKING, Literal, override

import genesis as gs
import torch
from genesis.engine.entities.rigid_entity import RigidLink
from jaxtyping import Float
from pydantic import Field
from robopianist.models.piano import piano_mjcf
from tensordict import TypedTensorDict

from src.base_config import Config
from src.sim.vec_system import ChildSystem, GenesisConfig, GenesisSystem

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

    color_pressed: bool = False
    """Highlight pressed keys in the viewer by drawing a colored box over them (environment 0 only)."""

    pressed_color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.7)
    """The RGBA color used to highlight pressed keys."""

    pressed_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """The fraction of a key's travel at which it is highlighted as pressed."""


class Piano(ChildSystem[PianoState, None, PianoConfig]):
    def __init__(
        self,
        cfg: PianoConfig,
        parent: GenesisSystem[PianoState, torch.Tensor, GenesisConfig],
    ):
        super().__init__(cfg, parent)
        self._scene: gs.Scene = parent.scene

        self._piano: RigidEntity = self._scene.add_entity(
            gs.morphs.MJCF(file=piano_mjcf.build().to_xml_string())
        )

        self._keys: list[RigidLink] = list(
            filter(lambda l: re.match(r".*_key_\d+", l.name), self._piano.links)
        )

        self._state: PianoState = PianoState(
            pressed=torch.zeros((self._num_envs, 88), device=gs.device),
            batch_size=self._num_envs,
            device=gs.device,
        )
        self._key_travel: torch.Tensor | None = None
        self._highlights: list[object] = []

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
        if self.cfg.color_pressed and self._scene.viewer is not None:
            self._highlight_pressed()

    def _highlight_pressed(self):
        """Redraws a box over each key that is pressed in environment 0."""
        # Genesis can't recolor a built mesh per key, so draw debug geometry instead.
        for highlight in self._highlights:
            self._scene.clear_debug_object(highlight)
        pressed = self._state.pressed[0] >= self.cfg.pressed_threshold
        # Slightly enlarge each box so it covers the key's surface instead of z-fighting.
        margin = torch.tensor([[-1e-3], [1e-3]], device=gs.device)
        self._highlights = [
            self._scene.draw_debug_box(
                self._keys[i].get_AABB()[0] + margin,
                color=self.cfg.pressed_color,
                wireframe=False,
            )
            for i in pressed.nonzero().flatten().tolist()
        ]

    @override
    def reset(self, idx: int | torch.Tensor | None = None) -> None:
        # The parent owns and resets the shared scene; only refresh our state here.
        self.step(None)

import re
from typing import TYPE_CHECKING, Literal, override

import genesis as gs
import torch
from genesis.engine.entities.rigid_entity import RigidLink
from jaxtyping import Float
from pydantic import Field
from robopianist import SF2_PATH
from robopianist.models.piano import piano_mjcf
from tensordict import TypedTensorDict

from src.base_config import Config
from src.sim.piano_audio import PianoAudio, RealTimePacer
from src.sim.vec_system import ChildSystem, GenesisConfig, GenesisSystem

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity


class PianoState(TypedTensorDict):
    pressed: Float[torch.Tensor, "*batch 88"]
    """The piano keys that are currently pressed, with values in [0.0, 1.0]."""
    hit_velocity: Float[torch.Tensor, "*batch 88"]
    """Key speed (fraction of travel per second) on the step each key last crossed the threshold.

    Held until the next hit. Zero before the first hit and after reset.
    """


class PianoAction(TypedTensorDict):
    sustain: Float[torch.Tensor, "*batch 1"]
    """An input corresponding to pressing the sustain pedal on a piano."""


class PianoConfig(Config):
    class_name: Literal["src.sim.piano:Piano"]

    color_pressed: bool = False
    """Highlight pressed keys in the viewer by drawing a colored box over them (environment 0 only).

    Ignored once something else colors the keys through show_key_colors, such as the
    HandPianoEnv's correctness colors.
    """

    pressed_color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.7)
    """The RGBA color used to highlight pressed keys."""

    pressed_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """The fraction of a key's travel at which it is highlighted as pressed."""

    audio: bool = False
    """Play key presses out loud through the default sound output (one environment).

    Needs FluidSynth and a working audio device. Meant for evaluation and viewing, not
    training: see audio_realtime.
    """

    audio_env: int = Field(default=0, ge=0)
    """The environment whose key presses are played."""

    audio_realtime: bool = True
    """While audio is on, slow the simulation to real time so notes keep their rhythm."""

    soundfont_path: str = str(SF2_PATH)
    """The SoundFont (.sf2) used for the piano sound; defaults to robopianist's."""

    audio_driver: str | None = None
    """FluidSynth audio driver, e.g. "pulseaudio", "pipewire", or "alsa"; None picks one."""

    audio_gain: float = Field(default=0.5, gt=0.0, le=10.0)
    """FluidSynth output gain."""

    full_velocity_speed: float = Field(default=20.0, gt=0.0)
    """Key speed (fractions of travel per second) that plays at full MIDI velocity, 127.

    20 means a key that goes all the way down in 50 ms. This is a starting guess, not a
    calibrated value; slower hits are proportionally quieter, down to min_velocity.
    """

    min_velocity: int = Field(default=30, ge=1, le=127)
    """The quietest MIDI velocity played, so slow presses stay audible."""

    press_point_inset: float = Field(default=0.02, ge=0.0)
    """Distance in meters from a key's front edge to its press point.

    Press points sit on the top surface of each key, centered across its width. Black
    keys end 6 cm short of the white keys' front edge, so their press points are
    farther back by that much.
    """


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
            hit_velocity=torch.zeros((self._num_envs, 88), device=gs.device),
            batch_size=self._num_envs,
            device=gs.device,
        )
        self._key_travel: torch.Tensor | None = None
        self._highlights: list[object] = []
        self._drawn_colors: torch.Tensor | None = None
        self._colored_externally: bool = False

        self._audio: PianoAudio | None = None
        self._pacer: RealTimePacer | None = None
        if cfg.audio:
            if cfg.audio_env >= self._num_envs:
                raise ValueError(
                    f"audio_env {cfg.audio_env} is out of range for {self._num_envs} environments"
                )
            self._audio = PianoAudio(
                cfg.soundfont_path,
                cfg.audio_driver,
                cfg.audio_gain,
                cfg.full_velocity_speed,
                cfg.min_velocity,
            )
            if cfg.audio_realtime:
                self._pacer = RealTimePacer(float(parent.cfg.genesis.sim_options.dt))

    def press_points(self) -> Float[torch.Tensor, "88 3"]:
        """World-frame point on each key, in key (pressed-channel) order, that a finger should strike.

        The piano is fixed and identical in every environment, so the points are
        measured once from environment 0 at rest. Requires a built scene.
        """
        # Keys run along y with the player on the +x side; tops face +z.
        aabbs = torch.stack([key.get_AABB().reshape(-1, 2, 3)[0] for key in self._keys])
        lower, upper = aabbs[:, 0], aabbs[:, 1]
        center = (lower + upper) / 2
        x = (upper[:, 0] - self.cfg.press_point_inset).clamp_min(center[:, 0])
        return torch.stack([x, center[:, 1], upper[:, 2]], dim=-1)

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
        was_down = self._state.pressed >= self.cfg.pressed_threshold
        _ = self._state.pressed.copy_((positions / self._key_travel).clamp(0.0, 1.0))
        is_down = self._state.pressed >= self.cfg.pressed_threshold
        # Hack: we only sample once per control step, so this is the speed at the
        # end of the step where the key crossed, not the exact instant of the crossing.
        speed = (self._piano.get_dofs_velocity() / self._key_travel).abs()
        hit = is_down & ~was_down
        _ = self._state.hit_velocity.copy_(
            torch.where(hit, speed, self._state.hit_velocity)
        )
        if self._audio is not None:
            env = self.cfg.audio_env
            self._audio.update(hit[env], (was_down & ~is_down)[env], speed[env])
        if self._pacer is not None:
            self._pacer.wait()
        if (
            self.cfg.color_pressed
            and not self._colored_externally
            and self._scene.viewer is not None
        ):
            pressed = self._state.pressed[0] >= self.cfg.pressed_threshold
            color = torch.tensor(self.cfg.pressed_color, device=gs.device)
            self._draw_key_colors(pressed.unsqueeze(-1) * color)

    def show_key_colors(self, colors: Float[torch.Tensor, "88 4"]):
        """Colors environment 0's keys in the viewer with RGBA colors, in key order.

        Keys with zero alpha are left uncolored. Calling this turns off color_pressed.
        Does nothing without a viewer.
        """
        self._colored_externally = True
        if self._scene.viewer is not None:
            self._draw_key_colors(colors)

    def _draw_key_colors(self, colors: Float[torch.Tensor, "88 4"]):
        """Redraws a colored box over each key with nonzero alpha in environment 0."""
        if self._drawn_colors is not None and torch.equal(colors, self._drawn_colors):
            return  # Nothing changed; skip redrawing debug geometry.
        self._drawn_colors = colors.clone()
        # Genesis can't recolor a built mesh per key, so draw debug geometry instead.
        for highlight in self._highlights:
            self._scene.clear_debug_object(highlight)
        # Slightly enlarge each box so it covers the key's surface instead of z-fighting.
        margin = torch.tensor([[-1e-3], [1e-3]], device=gs.device)
        self._highlights = [
            self._scene.draw_debug_box(
                self._keys[i].get_AABB()[0] + margin,
                color=tuple(colors[i].tolist()),
                wireframe=False,
            )
            for i in (colors[:, 3] > 0).nonzero().flatten().tolist()
        ]

    @override
    def reset(self, idx: int | torch.Tensor | None = None) -> None:
        # The parent owns and resets the shared scene; only refresh our state here.
        if self._audio is not None and _includes(idx, self.cfg.audio_env):
            # Keys reset up without crossing the threshold, so stop their notes here.
            self._audio.silence()
        if idx is None:
            _ = self._state.pressed.zero_()
            _ = self._state.hit_velocity.zero_()
        else:
            self._state.pressed[idx] = 0
            self._state.hit_velocity[idx] = 0
        self.step(None)


def _includes(idx: int | torch.Tensor | None, env: int) -> bool:
    """Whether a reset of `idx` (None for all) includes environment `env`."""
    if idx is None:
        return True
    return bool((torch.as_tensor(idx) == env).any())

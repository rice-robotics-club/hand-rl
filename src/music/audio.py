"""Plays the simulated piano's key presses out loud with FluidSynth."""

import time

import torch

from src.music.keyboard import LOWEST_PITCH

_CHANNEL = 0
_PROGRAM = 0  # General MIDI Acoustic Grand Piano.
MAX_VELOCITY = 127
"""The loudest MIDI note velocity."""


class PianoAudio:
    """Turns key presses in one environment into live notes.

    A key sounds when it crosses the press threshold and stops when it comes back
    up. Loudness follows how fast the key was moving when it crossed.
    """

    def __init__(
        self,
        soundfont_path: str,
        driver: str | None,
        gain: float,
        full_velocity_speed: float,
        min_velocity: int,
        period_size: int = 1024,
        periods: int = 4,
    ):
        # Imported here so the simulation runs without FluidSynth when audio is off.
        import fluidsynth

        # FluidSynth's default 64-frame buffer underruns on devices that PipeWire runs
        # with large buffers (e.g. 2048 frames for a USB interface), which pulses.
        self._synth = fluidsynth.Synth(
            gain=gain, **{"audio.period-size": period_size, "audio.periods": periods}
        )
        self._synth.start(driver=driver)
        soundfont = self._synth.sfload(soundfont_path)
        if soundfont == -1:
            raise FileNotFoundError(f"Could not load soundfont {soundfont_path}")
        self._synth.program_select(_CHANNEL, soundfont, 0, _PROGRAM)
        self._full_velocity_speed = full_velocity_speed
        self._min_velocity = min_velocity

    def update(self, hit: torch.Tensor, released: torch.Tensor, speed: torch.Tensor):
        """Starts and stops notes for one environment's (keys,) masks and hit speeds."""
        for key in released.nonzero().flatten().tolist():
            self._synth.noteoff(_CHANNEL, LOWEST_PITCH + key)
        hits = hit.nonzero().flatten()
        if len(hits) == 0:
            return
        # Map key speed (fractions of travel per second) onto MIDI velocity.
        velocities = (MAX_VELOCITY * speed[hits] / self._full_velocity_speed).clamp(
            self._min_velocity, MAX_VELOCITY
        )
        for key, velocity in zip(hits.tolist(), velocities.round().int().tolist()):
            self._synth.noteon(_CHANNEL, LOWEST_PITCH + key, velocity)

    def silence(self):
        """Stops every sounding note immediately."""
        self._synth.all_sounds_off(_CHANNEL)

    def close(self):
        self.silence()
        self._synth.delete()


class RealTimePacer:
    """Sleeps so simulated time never runs ahead of wall-clock time."""

    def __init__(self, dt: float):
        self._dt = dt
        self.restart()

    def restart(self):
        self._start = time.perf_counter()
        self._steps = 0

    def wait(self):
        self._steps += 1
        ahead = self._start + self._steps * self._dt - time.perf_counter()
        if ahead > 0:
            time.sleep(ahead)
        elif ahead < -0.25:
            # Running slower than real time; don't try to catch up in a burst.
            self.restart()

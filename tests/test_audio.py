"""PianoAudio note logic, with FluidSynth replaced by a recorder; no audio device needed."""

import sys
import unittest
from types import ModuleType
from typing import ClassVar
from unittest.mock import patch

import torch

from src.music.audio import PianoAudio, RealTimePacer
from src.music.keyboard import NUM_KEYS, key_of


class FakeSynth:
    """Records what PianoAudio asks FluidSynth to play."""

    instances: ClassVar[list["FakeSynth"]] = []

    def __init__(self, gain: float, **settings: int):
        self.events: list[tuple[object, ...]] = []
        FakeSynth.instances.append(self)

    def start(self, driver: str | None = None):
        self.events.append(("start", driver))

    def sfload(self, path: str) -> int:
        return 1

    def program_select(self, channel: int, soundfont: int, bank: int, program: int):
        pass

    def noteon(self, channel: int, pitch: int, velocity: int):
        self.events.append(("on", pitch, velocity))

    def noteoff(self, channel: int, pitch: int):
        self.events.append(("off", pitch))

    def all_sounds_off(self, channel: int):
        self.events.append(("silence",))

    def delete(self):
        pass


class FakeFluidsynth(ModuleType):
    Synth = FakeSynth


class PianoAudioTests(unittest.TestCase):
    def make_audio(self) -> tuple[PianoAudio, list[tuple[object, ...]]]:
        with patch.dict(sys.modules, {"fluidsynth": FakeFluidsynth("fluidsynth")}):
            audio = PianoAudio("piano.sf2", None, 0.5, 20.0, 30)
        return audio, FakeSynth.instances[-1].events

    def test_hits_play_at_speed_scaled_velocity_and_releases_stop(self):
        audio, events = self.make_audio()
        hit = torch.zeros(NUM_KEYS, dtype=torch.bool)
        c4, e4 = key_of(60), key_of(64)
        hit[[c4, e4]] = True
        speed = torch.zeros(NUM_KEYS)
        speed[c4], speed[e4] = 20.0, 1.0  # Full speed, and very slow.
        audio.update(hit, torch.zeros(NUM_KEYS, dtype=torch.bool), speed)
        released = torch.zeros(NUM_KEYS, dtype=torch.bool)
        released[c4] = True
        audio.update(torch.zeros(NUM_KEYS, dtype=torch.bool), released, speed)
        self.assertEqual(events[1:], [("on", 60, 127), ("on", 64, 30), ("off", 60)])

    def test_pacer_never_runs_ahead_of_wall_clock(self):
        clock = iter([0.0, 0.0, 0.01, 0.05])
        sleeps = []
        with (
            patch("src.music.audio.time.perf_counter", lambda: next(clock)),
            patch("src.music.audio.time.sleep", sleeps.append),
        ):
            pacer = RealTimePacer(dt=0.02)
            pacer.wait()  # Step 1 is due at 0.02 s; now is 0.0 s.
            pacer.wait()  # Step 2 is due at 0.04 s; now is 0.01 s.
            pacer.wait()  # Step 3 is due at 0.06 s; now is 0.05 s.
        self.assertEqual([round(s, 3) for s in sleeps], [0.02, 0.03, 0.01])


if __name__ == "__main__":
    unittest.main()

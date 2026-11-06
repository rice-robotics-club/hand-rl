"""PianoAudio note logic, with FluidSynth replaced by a recorder; no audio device needed."""

import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import torch

from src.sim.piano_audio import PianoAudio, RealTimePacer


class FakeSynth:
    def __init__(self, gain):
        self.events = []

    def start(self, driver=None):
        self.events.append(("start", driver))

    def sfload(self, path):
        return 1

    def program_select(self, channel, soundfont, bank, program):
        pass

    def noteon(self, channel, pitch, velocity):
        self.events.append(("on", pitch, velocity))

    def noteoff(self, channel, pitch):
        self.events.append(("off", pitch))

    def all_sounds_off(self, channel):
        self.events.append(("silence",))

    def delete(self):
        pass


class PianoAudioTests(unittest.TestCase):
    def make_audio(self):
        fluidsynth = ModuleType("fluidsynth")
        fluidsynth.Synth = FakeSynth
        with patch.dict(sys.modules, {"fluidsynth": fluidsynth}):
            audio = PianoAudio("piano.sf2", None, 0.5, 20.0, 30)
        return audio, audio._synth.events

    def test_hits_play_at_speed_scaled_velocity_and_releases_stop(self):
        audio, events = self.make_audio()
        hit = torch.zeros(88, dtype=torch.bool)
        hit[[39, 43]] = True  # C4 and E4.
        speed = torch.zeros(88)
        speed[39], speed[43] = 20.0, 1.0  # Full speed, and very slow.
        audio.update(hit, torch.zeros(88, dtype=torch.bool), speed)
        released = torch.zeros(88, dtype=torch.bool)
        released[39] = True
        audio.update(torch.zeros(88, dtype=torch.bool), released, speed)
        self.assertEqual(
            events[1:], [("on", 60, 127), ("on", 64, 30), ("off", 60)]
        )

    def test_pacer_never_runs_ahead_of_wall_clock(self):
        clock = iter([0.0, 0.0, 0.01, 0.05])
        sleeps = []
        with (
            patch("src.sim.piano_audio.time.perf_counter", lambda: next(clock)),
            patch("src.sim.piano_audio.time.sleep", sleeps.append),
        ):
            pacer = RealTimePacer(dt=0.02)
            pacer.wait()  # Step 1 is due at 0.02 s; now is 0.0 s.
            pacer.wait()  # Step 2 is due at 0.04 s; now is 0.01 s.
            pacer.wait()  # Step 3 is due at 0.06 s; now is 0.05 s.
        self.assertEqual([round(s, 3) for s in sleeps], [0.02, 0.03, 0.01])


if __name__ == "__main__":
    unittest.main()

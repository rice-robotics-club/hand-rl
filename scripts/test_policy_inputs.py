"""Deterministic input-loop timing tests without sleeping or hardware."""

import unittest
from threading import Event

from src.deployment.input_loop import InputLoop, training_control_period


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def read(self):
        return self.now

    def wait(self, duration):
        self.now += duration


class Source:
    def __init__(self):
        self.calls = []
        self.sensors = {"joints": [1, 2]}
        self.goal = {"pitch": 60}

    def read_sensors(self):
        self.calls.append("sensors")
        return self.sensors

    def read_goal(self, elapsed_seconds):
        self.calls.append(("goal", elapsed_seconds))
        return self.goal


class InputLoopTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.source = Source()
        self.loop = InputLoop(0.05, clock=self.clock.read, wait=self.clock.wait)

    def test_20_hz_reads_both_inputs_without_processing_drift(self):
        frames = []

        def consume(frame):
            frames.append(frame)
            self.clock.now += 0.01

        count = self.loop.run(self.source, consume, max_frames=4)
        self.assertEqual(count, 4)
        self.assertEqual([f.tick for f in frames], [0, 1, 2, 3])
        for frame, expected in zip(frames, [0, 0.05, 0.1, 0.15]):
            self.assertAlmostEqual(frame.sampled_at, expected)
            self.assertAlmostEqual(frame.elapsed_seconds, expected)
        self.assertEqual(len(self.source.calls), 8)
        self.assertEqual(self.source.calls[::2], ["sensors"] * 4)
        self.assertEqual(
            [c[1] for c in self.source.calls[1::2]], [f.elapsed_seconds for f in frames]
        )

    def test_overrun_skips_ticks_instead_of_bursting(self):
        frames = []

        def consume(frame):
            frames.append(frame)
            if len(frames) == 1:
                self.clock.now += 0.12

        self.loop.run(self.source, consume, max_frames=3)
        self.assertEqual([f.tick for f in frames], [0, 3, 4])
        self.assertAlmostEqual(frames[1].elapsed_seconds, 0.15)
        self.assertAlmostEqual(frames[2].elapsed_seconds, 0.20)

    def test_snapshots_do_not_change_with_source_buffers(self):
        frames = []
        self.loop.run(self.source, frames.append, max_frames=1)
        self.source.sensors["joints"][0] = 99
        self.source.goal["pitch"] = 70
        self.assertEqual(frames[0].sensors["joints"], [1, 2])
        self.assertEqual(frames[0].goal["pitch"], 60)

    def test_stop_before_start_and_during_wait(self):
        stop = Event()
        stop.set()
        self.assertEqual(self.loop.run(self.source, lambda _: None, stop_event=stop), 0)
        self.assertEqual(self.source.calls, [])
        stop.clear()
        loop = InputLoop(0.05, clock=self.clock.read, wait=lambda _: stop.set())
        frames = []
        self.assertEqual(loop.run(self.source, frames.append, stop_event=stop), 1)

    def test_source_errors_stop_acquisition(self):
        self.source.read_sensors = lambda: (_ for _ in ()).throw(
            TimeoutError("sensor timeout")
        )
        with self.assertRaises(TimeoutError):
            self.loop.run(
                self.source, lambda _: self.fail("No frame expected"), max_frames=1
            )

    def test_invalid_periods_and_limits(self):
        for period in [0, -1, float("nan"), float("inf"), True]:
            with self.subTest(period=period), self.assertRaises(ValueError):
                InputLoop(period)
        with self.assertRaises(ValueError):
            self.loop.run(self.source, lambda _: None, max_frames=0)

    def test_training_rate_and_ruka_override(self):
        cfg = {
            "system": {
                "class_name": "src.sim.hand_piano:HandPiano",
                "genesis": {"sim_options": {"dt": 0.05, "substeps": 10}},
            }
        }
        self.assertEqual(training_control_period(cfg), 0.05)
        cfg["system"]["genesis"]["sim_options"]["dt"] = 1 / 60
        self.assertEqual(training_control_period(cfg), 1 / 60)
        cfg["system"]["class_name"] = "src.sim.ruka_sim:RukaSim"
        cfg["system"]["control_dt"] = 0.1
        self.assertEqual(training_control_period(cfg), 0.1)

    def test_missing_timing_is_not_guessed(self):
        with self.assertRaises(ValueError):
            training_control_period(
                {"system": {"class_name": "src.sim.hand_piano:HandPiano"}}
            )


if __name__ == "__main__":
    unittest.main()

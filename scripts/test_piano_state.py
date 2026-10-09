"""State/timing checks using the pulled interfaces without Genesis or robot assets.

Run with Python 3.13 and project dependencies: python -m unittest scripts.test_piano_state
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.env.piano_targets import PianoTargets
from src.env.ruka_env import RukaEnv, RukaEnvConfig
from src.sim.ruka_sim import RukaAction, RukaSim, RukaSimConfig, RukaState


def make_state():
    def zeros(width):
        return torch.zeros(2, width)

    return RukaState(
        position=zeros(3),
        orientation=torch.tensor([[1.0, 0, 0, 0]]).repeat(2, 1),
        velocity=zeros(3),
        joint_positions=zeros(7),
        joint_velocities=zeros(7),
        time=torch.zeros(2, 1, dtype=torch.float64),
        control_dt=torch.full((2, 1), 1 / 60, dtype=torch.float64),
        fingertip_positions=zeros(15),
        fingertip_valid=zeros(5),
        key_pitches=torch.tensor([[60.0, 62, 64, 65, 67]]).repeat(2, 1),
        key_displacement=zeros(5),
        key_pressed=zeros(5),
        key_valid=zeros(5),
        batch_size=[2],
    )


class TargetsTests(unittest.TestCase):
    def test_chords_overlap_repeats_and_release(self):
        targets = PianoTargets(
            [(0, 0.5, 60), (0, 0.25, 64), (0.25, 0.5, 60)], [60, 64], "cpu"
        )
        result = targets.sample(
            torch.tensor([[0.0], [0.25], [0.5], [0.75]], dtype=torch.float64)
        )
        self.assertEqual(
            result["target_active"].tolist(), [[1, 1], [1, 0], [1, 0], [0, 0]]
        )
        self.assertEqual(result["target_remaining"][:, 0].tolist(), [0.5, 0.5, 0.25, 0])
        self.assertEqual(result["next_note_valid"][:, 0].tolist(), [1, 0, 0, 0])
        self.assertEqual(result["next_note_time"][0, 0].item(), 0.25)

    def test_unrounded_onset_and_duration(self):
        targets = PianoTargets([(0.02, 0.1, 60)], [60], "cpu")
        result = targets.sample(
            torch.tensor([[1 / 60], [2 / 60], [0.12]], dtype=torch.float64)
        )
        self.assertEqual(result["target_active"].flatten().tolist()[:2], [0, 1])
        self.assertAlmostEqual(result["next_note_time"][0, 0].item(), 0.02 - 1 / 60)
        self.assertAlmostEqual(result["next_note_duration"][0, 0].item(), 0.1)

    def test_empty_score_and_invalid_input(self):
        result = PianoTargets([], [60], "cpu").sample(
            torch.zeros(2, 1, dtype=torch.float64)
        )
        self.assertTrue(all(not x.any() for x in result.values()))
        for row in [
            [0, 1, 61],
            [0, -1, 60],
            [-1, 1, 60],
            [0, 1, 60.5],
            [0, float("nan"), 60],
            [0, 1],
        ]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                PianoTargets([row], [60], "cpu")


class StateTests(unittest.TestCase):
    def test_environment_uses_system_time_and_returns_independent_snapshots(self):
        state = make_state()
        sim = SimpleNamespace(state=state, num_envs=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notes.json"
            path.write_text(json.dumps([[0, 0.5, 60]]))
            env = RukaEnv(
                sim,
                RukaEnvConfig(
                    class_name="src.env.ruka_env:RukaEnv", notes_path=str(path)
                ),
            )
        old = env.get_observations()
        state.time[1] = 0.5
        state.joint_positions[1] = 1
        current = env.get_observations()
        self.assertEqual(current["target_active"][:, 0].tolist(), [1, 0])
        self.assertEqual(old["joint_positions"].sum().item(), 0)
        self.assertFalse(current["key_valid"].any())
        self.assertFalse(current["fingertip_valid"].any())
        self.assertTrue(all(x.dtype == torch.float32 for x in current.values()))

    def test_sim_step_and_partial_reset_refresh_clock_and_measurements(self):
        # Exercise the real RukaSim step/reset methods with a small fake backend.
        sim = RukaSim.__new__(RukaSim)
        sim._num_envs, sim._dt = 2, 1 / 60
        sim._steps = torch.zeros(2, 1, dtype=torch.int64)
        sim._state = make_state()
        position = torch.ones(2, 3)
        joints = torch.ones(2, 7)

        def reset(envs_idx):
            index = slice(None) if envs_idx is None else envs_idx
            position[index] = 0
            joints[index] = 0

        sim._scene = SimpleNamespace(step=lambda: None, reset=reset)
        sim._robot = SimpleNamespace(
            set_dofs_position=lambda _: None,
            get_pos=lambda: position,
            get_quat=lambda: sim._state.orientation,
            get_vel=lambda: position,
            get_dofs_position=lambda: joints,
            get_dofs_velocity=lambda: joints,
        )
        for _ in range(30):
            sim.step(RukaAction.from_tensor(torch.zeros(2, 27)))
        self.assertEqual(sim.state.time.tolist(), [[0.5], [0.5]])
        sim.reset(0)
        self.assertEqual(sim.state.time.tolist(), [[0], [0.5]])
        self.assertEqual(sim.state.joint_positions[:, 0].tolist(), [0, 1])
        sim.reset()
        self.assertFalse(sim.state.time.any())
        self.assertFalse(sim.state.position.any())

    def test_configuration_rejects_bad_timing_or_keyboard(self):
        for overrides in [
            {"control_dt": 0},
            {"control_dt": float("inf")},
            {"physics_substeps": 0},
            {"key_pitches": [60, 60]},
            {"key_pitches": [128]},
            {"key_pitches": []},
        ]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                RukaSimConfig(class_name="src.sim.ruka_sim:RukaSim", **overrides)


if __name__ == "__main__":
    unittest.main()

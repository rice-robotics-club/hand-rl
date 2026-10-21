"""Synthetic reward and task tests; no Genesis scene or robot assets required."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from tensordict import TypedTensorDict

from src.env.hand_piano import (
    HandPianoEnv,
    HandPianoEnvConfig,
    key_match_terms,
    proximity_reward,
)


class KeyState(TypedTensorDict):
    pressed: torch.Tensor


class FakeSystem:
    def __init__(self, dt=0.25):
        self.num_envs, self.num_actions = 2, 27
        self.cfg = SimpleNamespace(
            genesis=SimpleNamespace(sim_options=SimpleNamespace(dt=dt))
        )
        self.state = KeyState(pressed=torch.zeros(2, 88), batch_size=[2])
        self.reset_indices = []

    def step(self, actions):
        self.state["pressed"].copy_(actions)

    def reset(self, idx):
        self.reset_indices.append(idx.tolist())
        self.state["pressed"][idx] = 0


class RewardTests(unittest.TestCase):
    def test_correct_missing_extra_and_chords(self):
        target = torch.tensor([[1.0, 0, 0], [1.0, 1, 0], [1.0, 0, 0], [0.0, 0, 0]])
        actual = torch.tensor([[1.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0], [0.0, 0, 0]])
        terms = key_match_terms(actual, target, 0.5)
        self.assertEqual(terms["correct_notes"].tolist(), [1, 0.5, 1, 0])
        self.assertEqual(terms["missed_notes"].tolist(), [0, -0.5, 0, 0])
        self.assertEqual(terms["extra_notes"].tolist(), [0, 0, -1, 0])
        self.assertEqual(terms["note_correctness"].tolist(), [1, 0.5, 0.5, 1])

    def test_proximity_is_bounded_monotonic_and_silent_on_rests(self):
        target = torch.tensor([[1.0], [1.0], [1.0], [0.0]])
        result = proximity_reward(
            torch.tensor([[0.0], [0.02], [0.1], [0.0]]), target, 0.02
        )
        self.assertEqual(result[0].item(), 1)
        self.assertGreater(result[0], result[1])
        self.assertGreater(result[1], result[2])
        self.assertEqual(result[3].item(), 0)
        for distances in [torch.tensor([[-1.0]]), torch.tensor([[float("nan")]])]:
            with self.assertRaises(ValueError):
                proximity_reward(distances, torch.ones(1, 1), 0.02)

    def test_correct_press_beats_hovering_and_pressing_every_key(self):
        target = torch.zeros(1, 88)
        target[0, 39] = 1
        values = []
        for actual in [target, torch.zeros_like(target), torch.ones_like(target)]:
            terms = key_match_terms(actual, target, 0.5)
            values.append(
                sum(
                    terms[k] for k in ["correct_notes", "missed_notes", "extra_notes"]
                ).item()
            )
        self.assertEqual(values, [1, -1, -86])
        self.assertGreater(
            values[0], values[1] + 0.2
        )  # Maximum illustrative nearness bonus.

    def test_bad_config_is_rejected(self):
        for kwargs in [
            {"nearness_scale": 0},
            {"max_episode_length": 0},
            {"reward_weights": {"target_distance": 1}},
            {"key_pitches": [60] * 88},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HandPianoEnvConfig(
                    class_name="src.env.hand_piano:HandPianoEnv", **kwargs
                )


class TaskTests(unittest.TestCase):
    def setUp(self):
        # Substitute only physics-facing types. Use the real RewardFunction,
        # environment, score sampler, and torch arithmetic.
        sim = ModuleType("src.sim")
        vec = ModuleType("src.sim.vec_system")
        vec.VecAction = TypedTensorDict
        vec.VecSystem = object
        hand = ModuleType("src.sim.hand_piano")
        hand.HandPianoAction = SimpleNamespace(from_tensor=lambda actions: actions)
        self.modules = patch.dict(
            sys.modules,
            {
                "src.sim": sim,
                "src.sim.vec_system": vec,
                "src.sim.hand_piano": hand,
            },
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "notes.json"
        self.path.write_text(json.dumps([[0, 0.5, 60]]))

    def make_env(self, dt=0.25, **overrides):
        system = FakeSystem(dt)
        cfg = HandPianoEnvConfig(
            class_name="src.env.hand_piano:HandPianoEnv",
            notes_path=str(self.path),
            lead_in_seconds=0,
            release_tail_seconds=0.5,
            **overrides,
        )
        return HandPianoEnv(system, cfg), system

    def test_hold_release_and_terminal_reset(self):
        env, system = self.make_env()
        press = torch.zeros(2, 88)
        press[:, 39] = 1
        first = env.get_observations()
        self.assertEqual(first["target_active"][:, 39].tolist(), [1, 1])
        for _ in range(2):
            obs, reward, done, _ = env.step(press)
            self.assertEqual(reward.tolist(), [0.25, 0.25])
            self.assertFalse(done.any())
        self.assertEqual(obs["target_active"][:, 39].tolist(), [0, 0])
        _, reward, _, _ = env.step(press)  # Late release is an extra press.
        self.assertEqual(reward.tolist(), [-0.25, -0.25])
        obs, reward, done, extras = env.step(torch.zeros_like(press))
        self.assertEqual(reward.tolist(), [0, 0])
        self.assertTrue(done.all())
        self.assertFalse(extras["time_outs"].any())
        self.assertEqual(system.reset_indices, [[0, 1]])
        self.assertEqual(obs["target_active"][:, 39].tolist(), [1, 1])
        self.assertEqual(first["pressed"].sum().item(), 0)

    def test_independent_clocks(self):
        env, system = self.make_env()
        env.episode_length_buf[:] = torch.tensor([3, 1])
        obs, _, done, _ = env.step(torch.zeros(2, 88))
        self.assertEqual(done.tolist(), [True, False])
        self.assertEqual(system.reset_indices, [[0]])
        self.assertEqual(env.episode_length_buf.tolist(), [0, 2])
        self.assertEqual(obs["target_active"][:, 39].tolist(), [1, 0])

    def test_reward_integral_does_not_depend_on_control_frequency(self):
        totals = []
        for dt in [0.25, 0.125]:
            env, _ = self.make_env(dt)
            total = 0
            for _ in range(env.max_episode_length):
                actions = env.get_observations()["target_active"]
                _, reward, _, _ = env.step(actions)
                total += reward[0].item()
            totals.append(total)
        self.assertEqual(totals, [0.5, 0.5])

    def test_incomplete_episode_or_unavailable_geometry_fails(self):
        with self.assertRaisesRegex(ValueError, "at least"):
            self.make_env(max_episode_length=1)
        weights = {
            "correct_notes": 1,
            "missed_notes": 1,
            "extra_notes": 1,
            "note_nearness": 0.2,
        }
        with self.assertRaisesRegex(ValueError, "measured fingertip_key_distances"):
            self.make_env(reward_weights=weights)


if __name__ == "__main__":
    unittest.main()

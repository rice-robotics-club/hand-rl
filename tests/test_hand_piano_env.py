"""HandPianoEnv with a fake system; no Genesis scene or robot assets required."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any, override

import torch
from jaxtyping import Float

from src.base_config import Config
from src.env.hand_piano import REWARD_TERMS, HandPianoEnv, HandPianoEnvConfig
from src.music.keyboard import NUM_KEYS, key_of
from src.reward.motion import action_rate, finger_posture, hand_orientation
from src.reward.piano import finger_nearness, note_nearness
from src.sim.hand_piano import (
    NUM_ACTIONS,
    NUM_FINGERS,
    NUM_JOINTS,
    HandPianoAction,
    HandPianoState,
)

NUM_ENVS = 2
C4 = key_of(60)
CSHARP4 = key_of(61)
MIDDLE_FINGER = 2

ZERO_ACTION = HandPianoAction.from_tensor(torch.zeros(NUM_ENVS, NUM_ACTIONS))

ALL_WEIGHTS = {name: 0.0 for name in REWARD_TERMS} | {
    "correct_notes": 1.0,
    "missed_notes": 1.0,
    "extra_notes": 1.0,
}


class FakeSystem:
    """A HandPianoSystem whose keys go down as the test says, ignoring actions."""

    def __init__(
        self,
        cfg: Config | None = None,
        dt: float = 0.25,
        distances: torch.Tensor | None = None,
    ):
        self._cfg = cfg or Config()
        self.cfg = self._cfg
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.dt = dt
        if distances is None:
            distances = torch.ones(NUM_ENVS, NUM_FINGERS, NUM_KEYS)
        self.state = HandPianoState(
            position=torch.zeros(NUM_ENVS, 3),
            orientation=torch.zeros(NUM_ENVS, 4),
            velocity=torch.zeros(NUM_ENVS, 3),
            joint_positions=torch.zeros(NUM_ENVS, NUM_JOINTS),
            target_offsets=torch.zeros(NUM_ENVS, NUM_ACTIONS),
            fingertip_positions=torch.zeros(NUM_ENVS, NUM_FINGERS, 3),
            fingertip_key_distances=distances,
            pressed=torch.zeros(NUM_ENVS, NUM_KEYS),
            hit_velocity=torch.zeros(NUM_ENVS, NUM_KEYS),
            batch_size=[NUM_ENVS],
        )
        self.state.orientation[:, 0] = 1  # Identity quaternion.
        self.default_joint_positions = torch.zeros(NUM_JOINTS)
        self.default_orientation = torch.tensor([1.0, 0, 0, 0])
        self.next_pressed = torch.zeros(NUM_ENVS, NUM_KEYS)
        self.reset_indices: list[list[int]] = []
        self.shown_colors: list[torch.Tensor] = []

    def show_key_colors(self, colors: Float[torch.Tensor, f"{NUM_KEYS} 4"]) -> None:
        self.shown_colors.append(colors)

    def step(self, action: HandPianoAction) -> None:
        _ = self.state.pressed.copy_(self.next_pressed)

    def reset(self, idx: int | torch.Tensor | None = None) -> None:
        envs = torch.arange(NUM_ENVS) if idx is None else torch.as_tensor(idx)
        self.reset_indices.append(envs.reshape(-1).tolist())
        self.state.pressed[envs] = 0


def step(env: HandPianoEnv, system: FakeSystem, pressed: torch.Tensor):
    """Steps env with a zero action after which exactly `pressed` keys are down."""
    system.next_pressed = pressed
    return env.step(torch.zeros(env.num_envs, env.num_actions))


class TaskTests(unittest.TestCase):
    @override
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "notes.json"
        self.path.write_text(json.dumps([[0, 0.5, 60]]))

    def make_env(
        self,
        dt: float = 0.25,
        distances: torch.Tensor | None = None,
        **overrides: Any,
    ) -> tuple[HandPianoEnv, FakeSystem]:
        system = FakeSystem(dt=dt, distances=distances)
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
        press = torch.zeros(NUM_ENVS, NUM_KEYS)
        press[:, C4] = 1
        first = env.get_observations()
        self.assertEqual(first["target_active"][:, C4].tolist(), [1, 1])
        obs = first
        for _ in range(2):
            obs, reward, done, _ = step(env, system, press)
            self.assertEqual(reward.tolist(), [0.25, 0.25])
            self.assertFalse(done.any())
        self.assertEqual(obs["target_active"][:, C4].tolist(), [0, 0])
        _, reward, _, _ = step(env, system, press)  # Late release is an extra press.
        self.assertEqual(reward.tolist(), [-0.25, -0.25])
        obs, reward, done, extras = step(env, system, torch.zeros_like(press))
        self.assertEqual(reward.tolist(), [0, 0])
        self.assertTrue(done.all())
        self.assertFalse(extras["time_outs"].any())
        self.assertEqual(system.reset_indices, [[0, 1]])
        self.assertEqual(obs["target_active"][:, C4].tolist(), [1, 1])
        self.assertEqual(first["pressed"].sum().item(), 0)

    def test_independent_clocks(self):
        env, system = self.make_env()
        env.episode_length_buf[:] = torch.tensor([3, 1])
        obs, _, done, _ = step(env, system, torch.zeros(NUM_ENVS, NUM_KEYS))
        self.assertEqual(done.tolist(), [True, False])
        self.assertEqual(system.reset_indices, [[0]])
        self.assertEqual(env.episode_length_buf.tolist(), [0, 2])
        self.assertEqual(obs["target_active"][:, C4].tolist(), [1, 0])

    def test_reward_integral_does_not_depend_on_control_frequency(self):
        totals = []
        for dt in [0.25, 0.125]:
            env, system = self.make_env(dt)
            total = 0
            for _ in range(env.max_episode_length):
                actions = env.get_observations()["target_active"]
                _, reward, _, _ = step(env, system, actions)
                total += reward[0].item()
            totals.append(total)
        self.assertEqual(totals, [0.5, 0.5])

    def test_incomplete_episode_fails(self):
        with self.assertRaisesRegex(ValueError, "at least"):
            self.make_env(max_episode_length=1)

    def test_state_has_assigned_distances_and_rewards_are_pure(self):
        distances = torch.ones(NUM_ENVS, NUM_FINGERS, NUM_KEYS)
        distances[:, MIDDLE_FINGER, C4] = (
            0.0  # The middle fingertip is on C4 in both envs.
        )
        env, _ = self.make_env(distances=distances)
        state = env.get_observations()
        self.assertEqual(
            state.closest_fingers[:, C4].tolist(), [MIDDLE_FINGER] * NUM_ENVS
        )
        self.assertEqual(state.closest_finger_distances[:, C4].tolist(), [0, 0])
        # C4 is the thumb's home key, and the thumb is 1 m away; note_nearness still
        # credits the middle finger, finger_nearness does not.
        self.assertEqual(state.home_fingers[:, C4].tolist(), [0, 0])
        self.assertEqual(note_nearness(state, ZERO_ACTION, scale=0.02).tolist(), [1, 1])
        self.assertLess(
            finger_nearness(state, ZERO_ACTION, scale=0.02).max().item(), 1e-6
        )

    def test_env_colors_keys_by_what_the_rewards_scored(self):
        env, system = self.make_env()
        press = torch.zeros(NUM_ENVS, NUM_KEYS)
        press[:, CSHARP4] = 1  # C#4 instead of the requested C4.
        step(env, system, press)
        colors = system.shown_colors[-1]
        self.assertEqual(colors.shape, (NUM_KEYS, 4))
        torch.testing.assert_close(colors[C4], torch.tensor(env.cfg.target_color))
        torch.testing.assert_close(colors[CSHARP4], torch.tensor(env.cfg.wrong_color))
        self.assertEqual((colors[:, 3] > 0).sum().item(), 2)

        env, system = self.make_env(color_correctness=False)
        step(env, system, press)
        self.assertEqual(system.shown_colors, [])

    def test_rewards_use_the_score_at_the_start_of_the_interval(self):
        weights = {**ALL_WEIGHTS, "note_nearness": 1}
        distances = torch.ones(NUM_ENVS, NUM_FINGERS, NUM_KEYS)
        distances[:, MIDDLE_FINGER, C4] = 0.0
        env, system = self.make_env(distances=distances, reward_weights=weights)
        press = torch.zeros(NUM_ENVS, NUM_KEYS)
        press[:, C4] = 1
        _, reward, _, extras = step(env, system, press)
        # Correct note (+1), nearness (+1), over a 0.25 s interval.
        self.assertEqual(reward.tolist(), [0.5, 0.5])
        self.assertEqual(extras["log"]["reward/note_nearness"], 1)

    def test_action_rate_posture_and_orientation_come_from_state(self):
        env, _ = self.make_env()
        state = env.get_observations()
        still = HandPianoAction.from_tensor(torch.zeros(NUM_ENVS, NUM_ACTIONS))
        self.assertEqual(action_rate(state, still).tolist(), [0, 0])
        jump = HandPianoAction.from_tensor(
            torch.full((NUM_ENVS, NUM_ACTIONS), 5.0)
        )  # Clipped to 1.
        self.assertEqual(action_rate(state, jump).tolist(), [-1, -1])
        self.assertEqual(finger_posture(state, still, scale=0.3).tolist(), [1, 1])
        self.assertEqual(hand_orientation(state, still, scale=0.3).tolist(), [1, 1])
        state.posture_error[0] = 0.3
        state.orientation_error[0] = 0.3
        torch.testing.assert_close(
            finger_posture(state, still, scale=0.3), torch.tensor([math.exp(-1), 1.0])
        )
        torch.testing.assert_close(
            hand_orientation(state, still, scale=0.3), torch.tensor([math.exp(-1), 1.0])
        )

    def test_previous_action_is_the_last_clipped_action_and_resets(self):
        env, system = self.make_env()
        system.next_pressed = torch.zeros(NUM_ENVS, NUM_KEYS)
        env.step(torch.full((NUM_ENVS, NUM_ACTIONS), 3.0))
        self.assertTrue((env.get_observations().previous_action == 1).all())
        env.episode_length_buf[:] = torch.tensor([env.max_episode_length - 1, 0])
        obs, *_ = env.step(torch.full((NUM_ENVS, NUM_ACTIONS), 0.5))
        self.assertEqual(obs.previous_action[:, 0].tolist(), [0, 0.5])

    def test_curriculum_raises_penalties_as_recall_improves(self):
        final = ALL_WEIGHTS | {"extra_notes": 2}
        env, system = self.make_env(
            final_reward_weights=final,
            curriculum_start=0.0,
            curriculum_end=1.0,
            curriculum_rate=1.0,
        )
        # Play every requested note for a whole episode in both environments.
        obs = env.get_observations()
        for _ in range(env.max_episode_length - 1):
            obs, *_ = step(env, system, obs["target_active"])
        _, _, done, extras = step(env, system, obs["target_active"])
        self.assertTrue(done.all())
        self.assertEqual(extras["log"]["curriculum/note_recall"], 1.0)
        self.assertEqual(extras["log"]["curriculum/progress"], 1.0)
        self.assertEqual(env.curriculum.weights()["extra_notes"], 2)

    def test_bad_config_is_rejected(self):
        for kwargs in [
            {"nearness_scale": 0},
            {"max_episode_length": 0},
            {"reward_weights": {"target_distance": 1}},
            {"finger_pitches": [60] * 5},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HandPianoEnvConfig(
                    class_name="src.env.hand_piano:HandPianoEnv", **kwargs
                )


if __name__ == "__main__":
    unittest.main()

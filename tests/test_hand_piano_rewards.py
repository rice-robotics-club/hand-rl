"""Synthetic reward and task tests; no Genesis scene or robot assets required."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.env.hand_piano import (
    REWARD_TERMS,
    FingeringPlan,
    HandPianoEnv,
    HandPianoEnvConfig,
    action_rate,
    assign_fingers,
    assigned_distances,
    finger_nearness,
    finger_posture,
    hand_orientation,
    key_colors,
    key_match_terms,
    note_nearness,
    proximity_reward,
)
from src.sim.hand_piano import HandPianoAction, HandPianoState

ALL_WEIGHTS = {name: 0 for name in REWARD_TERMS} | {
    "correct_notes": 1,
    "missed_notes": 1,
    "extra_notes": 1,
}


class FakeSystem:
    """Stands in for HandPiano: keys go down as the test says, ignoring actions."""

    def __init__(self, dt=0.25, distances=None):
        self.num_envs, self.num_actions = 2, 27
        self.cfg = SimpleNamespace(
            genesis=SimpleNamespace(sim_options=SimpleNamespace(dt=dt))
        )
        if distances is None:
            distances = torch.ones(2, 5, 88)
        self.state = HandPianoState(
            position=torch.zeros(2, 3),
            orientation=torch.zeros(2, 4),
            velocity=torch.zeros(2, 3),
            joint_positions=torch.zeros(2, 21),
            target_offsets=torch.zeros(2, 27),
            fingertip_positions=torch.zeros(2, 5, 3),
            fingertip_key_distances=distances,
            pressed=torch.zeros(2, 88),
            hit_velocity=torch.zeros(2, 88),
            batch_size=[2],
        )
        self.state["orientation"][:, 0] = 1  # Identity quaternion.
        self.default_joint_positions = torch.zeros(21)
        self.default_orientation = torch.tensor([1.0, 0, 0, 0])
        self.next_pressed = torch.zeros(2, 88)
        self.reset_indices = []
        self.shown_colors = []

    def show_key_colors(self, colors):
        self.shown_colors.append(colors)

    def step(self, action):
        self.state["pressed"].copy_(self.next_pressed)

    def reset(self, idx):
        self.reset_indices.append(idx.tolist())
        self.state["pressed"][idx] = 0


def step(env, system, pressed):
    """Steps env with a zero action after which exactly `pressed` keys are down."""
    system.next_pressed = pressed
    return env.step(torch.zeros(env.num_envs, env.num_actions))


class RewardTests(unittest.TestCase):
    def test_correct_missing_extra_and_chords(self):
        target = torch.tensor([[1.0, 0, 0], [1.0, 1, 0], [1.0, 0, 0], [0.0, 0, 0]])
        actual = torch.tensor([[1.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0], [0.0, 0, 0]])
        terms = key_match_terms(actual, target)
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
            terms = key_match_terms(actual, target)
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


class KeyColorTests(unittest.TestCase):
    def test_wanted_keys_are_blue_then_green_and_unwanted_presses_are_red(self):
        blue, green, red = (0, 0, 1, 1.0), (0, 1, 0, 1.0), (1, 0, 0, 1.0)
        down = torch.tensor([0.0, 1, 1, 0])
        wanted = torch.tensor([1.0, 1, 0, 0])
        colors = key_colors(down, wanted, blue, green, red)
        self.assertEqual(colors.tolist(), [list(blue), list(green), list(red), [0, 0, 0, 0]])


class FingerAssignmentTests(unittest.TestCase):
    def test_chord_gets_distinct_fingers_with_least_total_distance(self):
        # Finger 0 is nearest both keys, but giving key 1 to finger 1 costs less overall.
        distances = torch.full((1, 5, 3), 1.0)
        distances[0, 0, :2] = torch.tensor([0.1, 0.2])
        distances[0, 1, :2] = torch.tensor([0.9, 0.3])
        fingers = assign_fingers(distances, torch.tensor([[1.0, 1, 0]]))
        self.assertEqual(fingers.tolist(), [[0, 1, -1]])
        torch.testing.assert_close(
            assigned_distances(distances, fingers), torch.tensor([[0.1, 0.3, 1.0]])
        )

    def test_rests_and_extra_keys_are_unassigned(self):
        distances = torch.rand(2, 5, 8)
        active = torch.tensor([[0.0] * 8, [1.0] * 6 + [0.0] * 2])
        fingers = assign_fingers(distances, active)
        self.assertTrue((fingers[0] == -1).all())
        assigned = fingers[1][fingers[1] >= 0]
        self.assertEqual(len(assigned), 5)
        self.assertEqual(len(set(assigned.tolist())), 5)
        self.assertTrue((fingers[1, 6:] == -1).all())
        # Unassigned keys still report a measured distance: the nearest finger.
        nearest = assigned_distances(distances, fingers)
        self.assertTrue(torch.equal(nearest[0], distances[0].min(0).values))

    def test_plan_fingers_follow_score_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps({"notes": [
                {"start": 0, "duration": 1, "pitch": 60, "finger": 1},
                {"start": 0, "duration": 1, "pitch": 64, "finger": 3},
            ]}))
            plan = FingeringPlan(
                path, [[0, 1, 60], [0, 1, 64]], list(range(21, 109)), 0.5, "cpu"
            )
            fingers = plan.sample(torch.tensor([[0.25], [0.75], [1.5]], dtype=torch.float64))
            self.assertEqual(fingers[:, [39, 43]].tolist(), [[-1, -1], [0, 2], [-1, -1]])
            self.assertEqual((fingers >= 0).sum().item(), 2)
            with self.assertRaisesRegex(ValueError, "score's notes"):
                FingeringPlan(path, [[0, 1, 60]], list(range(21, 109)), 0, "cpu")


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "notes.json"
        self.path.write_text(json.dumps([[0, 0.5, 60]]))

    def make_env(self, dt=0.25, distances=None, **overrides):
        system = FakeSystem(dt, distances)
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
            obs, reward, done, _ = step(env, system, press)
            self.assertEqual(reward.tolist(), [0.25, 0.25])
            self.assertFalse(done.any())
        self.assertEqual(obs["target_active"][:, 39].tolist(), [0, 0])
        _, reward, _, _ = step(env, system, press)  # Late release is an extra press.
        self.assertEqual(reward.tolist(), [-0.25, -0.25])
        obs, reward, done, extras = step(env, system, torch.zeros_like(press))
        self.assertEqual(reward.tolist(), [0, 0])
        self.assertTrue(done.all())
        self.assertFalse(extras["time_outs"].any())
        self.assertEqual(system.reset_indices, [[0, 1]])
        self.assertEqual(obs["target_active"][:, 39].tolist(), [1, 1])
        self.assertEqual(first["pressed"].sum().item(), 0)

    def test_independent_clocks(self):
        env, system = self.make_env()
        env.episode_length_buf[:] = torch.tensor([3, 1])
        obs, _, done, _ = step(env, system, torch.zeros(2, 88))
        self.assertEqual(done.tolist(), [True, False])
        self.assertEqual(system.reset_indices, [[0]])
        self.assertEqual(env.episode_length_buf.tolist(), [0, 2])
        self.assertEqual(obs["target_active"][:, 39].tolist(), [1, 0])

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
        distances = torch.full((2, 5, 88), 1.0)
        distances[:, 2, 39] = 0.0  # The middle fingertip is on C4 in both envs.
        env, _ = self.make_env(distances=distances)
        state = env.get_observations()
        self.assertEqual(state.target_fingers[:, 39].tolist(), [2, 2])
        self.assertEqual(state.target_key_distances[:, 39].tolist(), [0, 0])
        self.assertEqual(note_nearness(state, None, scale=0.02).tolist(), [1, 1])

        plan = Path(self.directory.name) / "plan.json"
        plan.write_text(json.dumps({"notes": [
            {"start": 0, "duration": 0.5, "pitch": 60, "finger": 1}
        ]}))
        env, _ = self.make_env(distances=distances, fingering_path=str(plan))
        state = env.get_observations()
        # The plan says the thumb plays C4, and the thumb is 1 m away; nearness still
        # credits the middle finger, finger_nearness does not.
        self.assertEqual(state.planned_fingers[:, 39].tolist(), [0, 0])
        self.assertEqual(state.target_fingers[:, 39].tolist(), [2, 2])
        self.assertEqual(note_nearness(state, None, scale=0.02).tolist(), [1, 1])
        self.assertLess(finger_nearness(state, None, scale=0.02).max().item(), 1e-6)

    def test_env_colors_keys_by_what_the_rewards_scored(self):
        env, system = self.make_env()
        press = torch.zeros(2, 88)
        press[:, 40] = 1  # C#4 instead of the requested C4.
        step(env, system, press)
        colors = system.shown_colors[-1]
        self.assertEqual(colors.shape, (88, 4))
        torch.testing.assert_close(colors[39], torch.tensor(env.cfg.target_color))
        torch.testing.assert_close(colors[40], torch.tensor(env.cfg.wrong_color))
        self.assertEqual((colors[:, 3] > 0).sum().item(), 2)

        env, system = self.make_env(color_correctness=False)
        step(env, system, press)
        self.assertEqual(system.shown_colors, [])

    def test_rewards_use_the_score_at_the_start_of_the_interval(self):
        weights = {**ALL_WEIGHTS, "note_nearness": 1}
        distances = torch.full((2, 5, 88), 1.0)
        distances[:, 2, 39] = 0.0
        env, system = self.make_env(distances=distances, reward_weights=weights)
        press = torch.zeros(2, 88)
        press[:, 39] = 1
        _, reward, _, extras = step(env, system, press)
        # Correct note (+1), nearness (+1), over a 0.25 s interval.
        self.assertEqual(reward.tolist(), [0.5, 0.5])
        self.assertEqual(extras["log"]["reward/note_nearness"], 1)

    def test_action_rate_posture_and_orientation_come_from_state(self):
        env, _ = self.make_env()
        state = env.get_observations()
        still = HandPianoAction.from_tensor(torch.zeros(2, 27))
        self.assertEqual(action_rate(state, still).tolist(), [0, 0])
        jump = HandPianoAction.from_tensor(torch.full((2, 27), 5.0))  # Clipped to 1.
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
        system.next_pressed = torch.zeros(2, 88)
        env.step(torch.full((2, 27), 3.0))
        self.assertTrue((env.get_observations().previous_action == 1).all())
        env.episode_length_buf[:] = torch.tensor([env.max_episode_length - 1, 0])
        obs, *_ = env.step(torch.full((2, 27), 0.5))
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
        for _ in range(env.max_episode_length):
            obs, _, done, extras = step(env, system, obs["target_active"])
        self.assertTrue(done.all())
        self.assertEqual(extras["log"]["curriculum/note_recall"], 1.0)
        self.assertEqual(extras["log"]["curriculum/progress"], 1.0)
        self.assertEqual(env._reward._weights[REWARD_TERMS.index("extra_notes")].item(), 2)

if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from .env import PrototypeFingerEnv


class PrototypeTests(unittest.TestCase):
    def env(self, **kwargs):
        return PrototypeFingerEnv([(0, 0.5, 60), (0, 0.5, 64)], num_envs=2, **kwargs)

    def test_shapes_and_rate_limit(self):
        env = self.env()
        obs, reward, done, _ = env.step(torch.ones(2, 5))
        self.assertEqual(obs["policy"].shape, (2, 46))
        self.assertEqual(reward.shape, (2,))
        self.assertEqual(done.shape, (2,))
        torch.testing.assert_close(env.bend, torch.full((2, 5), 0.2))

    def test_score_chord_and_release(self):
        env = self.env()
        self.assertEqual(env.targets[0].tolist(), [0, 0, 0, 0, 0])
        self.assertEqual(env.targets[10].tolist(), [1, 0, 1, 0, 0])
        self.assertEqual(env.targets[20].tolist(), [0, 0, 0, 0, 0])

    def test_correct_press_beats_wrong_and_missing(self):
        env = self.env(bend_speed=100)
        env.episode_length_buf[:] = 10
        actions = torch.tensor([[1., -1, 1, -1, -1], [-1., 1, -1, 1, -1]])
        _, rewards, _, _ = env.step(actions)
        self.assertGreater(rewards[0].item(), rewards[1].item())

    def test_episode_resets_only_finished_env(self):
        env = self.env()
        env.episode_length_buf[0] = env.max_episode_length - 1
        _, rewards, done, extras = env.step(torch.ones(2, 5))
        self.assertEqual(done.tolist(), [True, False])
        self.assertEqual(env.episode_length_buf.tolist(), [0, 1])
        self.assertEqual(env.bend[0].tolist(), [0] * 5)
        self.assertGreater(env.bend[1, 0].item(), 0)
        self.assertTrue(torch.isfinite(rewards).all())
        self.assertFalse(extras["time_outs"].any())

    def test_invalid_score_or_actions(self):
        with self.assertRaises(ValueError):
            PrototypeFingerEnv([(0, 1, 72)])
        with self.assertRaises(ValueError):
            self.env().step(torch.zeros(2, 6))
        with self.assertRaises(ValueError):
            self.env().step(torch.full((2, 5), float("nan")))


if __name__ == "__main__":
    unittest.main()

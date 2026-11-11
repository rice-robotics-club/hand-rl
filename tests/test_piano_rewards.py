"""Piano reward terms and measurements on synthetic key states."""

import unittest

import torch

from src.music.keyboard import NUM_KEYS, key_of
from src.reward.piano import (
    key_colors,
    key_metrics,
    key_palette,
    nearness,
    score_notes,
)


class RewardTests(unittest.TestCase):
    def test_correct_missing_extra_and_chords(self):
        target = torch.tensor([[1.0, 0, 0], [1.0, 1, 0], [1.0, 0, 0], [0.0, 0, 0]])
        actual = torch.tensor([[1.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0], [0.0, 0, 0]])
        notes = score_notes(actual, target)
        self.assertEqual(notes.correct.tolist(), [1, 0.5, 1, 0])
        self.assertEqual(notes.missed.tolist(), [0, 0.5, 0, 0])
        self.assertEqual(notes.extra.tolist(), [0, 0, 1, 0])
        self.assertEqual(notes.correctness.tolist(), [1, 0.5, 0.5, 1])

    def test_proximity_is_bounded_monotonic_and_silent_on_rests(self):
        target = torch.tensor([[1.0], [1.0], [1.0], [0.0]])
        result = nearness(torch.tensor([[0.0], [0.02], [0.1], [0.0]]), target, 0.02)
        self.assertEqual(result[0].item(), 1)
        self.assertGreater(result[0], result[1])
        self.assertGreater(result[1], result[2])
        self.assertEqual(result[3].item(), 0)
        with self.assertRaises(ValueError):
            nearness(torch.zeros(1, 2), torch.ones(1, 1), 0.02)

    def test_correct_press_beats_hovering_and_pressing_every_key(self):
        target = torch.zeros(1, NUM_KEYS)
        target[0, key_of(60)] = 1
        values = []
        for actual in [target, torch.zeros_like(target), torch.ones_like(target)]:
            notes = score_notes(actual, target)
            values.append((notes.correct - notes.missed - notes.extra).item())
        self.assertEqual(values, [1, -1, -(NUM_KEYS - 2)])
        self.assertGreater(
            values[0], values[1] + 0.2
        )  # Maximum illustrative nearness bonus.


class KeyColorTests(unittest.TestCase):
    def test_wanted_keys_are_blue_then_green_and_unwanted_presses_are_red(self):
        blue, green, red = (0, 0, 1, 1.0), (0, 1, 0, 1.0), (1, 0, 0, 1.0)
        down = torch.tensor([0.0, 1, 1, 0])
        wanted = torch.tensor([1.0, 1, 0, 0])
        colors = key_colors(down, wanted, key_palette(blue, green, red, down.device))
        self.assertEqual(
            colors.tolist(), [list(blue), list(green), list(red), [0, 0, 0, 0]]
        )


class KeyMetricsTests(unittest.TestCase):
    def test_metrics_agree_with_note_counts(self):
        target = torch.tensor([[1.0, 1, 0, 0], [0.0, 0, 0, 0], [0.0, 0, 0, 0]])
        down = torch.tensor([[1.0, 0, 1, 0], [0.0, 0, 0, 0], [1.0, 0, 0, 0]])
        precision, recall, f1 = key_metrics(down, target)
        torch.testing.assert_close(precision, torch.tensor([0.5, 1.0, 0.0]))
        torch.testing.assert_close(recall, torch.tensor([0.5, 1.0, 1.0]))
        torch.testing.assert_close(f1, torch.tensor([0.5, 1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()

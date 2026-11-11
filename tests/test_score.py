"""Score timing: what each key should be doing at a given time."""

import unittest

import torch

from src.music.score import Score


def targets(
    notes: list[tuple[float, float, int]], pitches: list[int], times: list[float]
) -> dict[str, torch.Tensor]:
    """Score.targets for one score, sampled at each of `times` (one env per time)."""
    score = Score.from_notes(notes, pitches, "cpu")
    time = torch.tensor(times, dtype=torch.float64).unsqueeze(-1)
    return score.targets(time, len(pitches))


class TargetsTests(unittest.TestCase):
    def test_chords_overlap_repeats_and_release(self):
        result = targets(
            [(0, 0.5, 60), (0, 0.25, 64), (0.25, 0.5, 60)],
            [60, 64],
            [0, 0.25, 0.5, 0.75],
        )
        self.assertEqual(
            result["target_active"].tolist(), [[1, 1], [1, 0], [1, 0], [0, 0]]
        )
        self.assertEqual(result["target_remaining"][:, 0].tolist(), [0.5, 0.5, 0.25, 0])
        self.assertEqual(result["next_note_valid"][:, 0].tolist(), [1, 0, 0, 0])
        self.assertEqual(result["next_note_time"][0, 0].item(), 0.25)

    def test_unrounded_onset_and_duration(self):
        result = targets([(0.02, 0.1, 60)], [60], [1 / 60, 2 / 60, 0.12])
        self.assertEqual(result["target_active"].flatten().tolist()[:2], [0, 1])
        self.assertAlmostEqual(result["next_note_time"][0, 0].item(), 0.02 - 1 / 60)
        self.assertAlmostEqual(result["next_note_duration"][0, 0].item(), 0.1)

    def test_empty_score_and_invalid_input(self):
        result = targets([], [60], [0, 0])
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
                Score.from_notes([row], [60], "cpu")


if __name__ == "__main__":
    unittest.main()

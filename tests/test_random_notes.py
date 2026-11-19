"""Random notes: the sets that are drawn, and the window that rolls forward in time."""

import math
import unittest
from typing import override

import torch

from src.music.keyboard import NUM_KEYS
from src.music.random_notes import (
    CURRENT,
    NEXT,
    NoteStream,
    RandomNoteSettings,
    draw_next_note_set,
    draw_note_set,
)
from src.music.score import Score

CPU = torch.device("cpu")


def settings(
    min_notes: int = 1,
    max_notes: int = 3,
    closeness: float = 3.0,
    min_hold: float = 0.5,
    max_hold: float = 2.0,
    min_rest: float = 0.0,
    max_rest: float = 0.5,
) -> RandomNoteSettings:
    """Settings that default to the config's defaults."""
    return RandomNoteSettings(
        min_notes, max_notes, closeness, min_hold, max_hold, min_rest, max_rest
    )


def keys_of(start: torch.Tensor, key: torch.Tensor, row: int) -> list[int]:
    """The keys of row's real notes (padding has start = inf)."""
    return key[row][start[row].isfinite()].tolist()


class DrawSetTests(unittest.TestCase):
    @override
    def setUp(self):
        torch.manual_seed(0)

    def test_count_keys_and_holds_stay_in_range(self):
        s = settings(min_notes=2, max_notes=4)
        notes = draw_note_set(torch.full((500,), 1.5), s)
        real = notes.start.isfinite()
        self.assertTrue(((real.sum(-1) >= 2) & (real.sum(-1) <= 4)).all())
        # Every count shows up, so the range is really used.
        self.assertEqual(set(real.sum(-1).tolist()), {2, 3, 4})
        for row in range(500):
            keys = keys_of(notes.start, notes.key, row)
            self.assertEqual(len(keys), len(set(keys)), "keys in a set differ")
        self.assertTrue(((notes.key >= 0) & (notes.key < NUM_KEYS)).all())
        holds = notes.duration[real]
        self.assertTrue(((holds >= 0.5) & (holds < 2.0)).all())
        self.assertTrue((notes.start[real] == 1.5).all())

    def test_padding_is_inert(self):
        notes = draw_note_set(torch.zeros(200), settings(min_notes=1, max_notes=3))
        pad = ~notes.start.isfinite()
        self.assertTrue(pad.any())
        self.assertTrue((notes.duration[pad] == 0).all())

    def test_hold_times_differ_within_a_set(self):
        notes = draw_note_set(torch.zeros(50), settings(min_notes=3, max_notes=3))
        self.assertTrue((notes.duration.std(-1) > 0).all())

    def test_closeness_sets_how_spread_out_keys_are(self):
        def spread(closeness: float) -> float:
            notes = draw_note_set(
                torch.zeros(2000),
                settings(min_notes=3, max_notes=3, closeness=closeness),
            )
            return (notes.key.float().std(-1)).mean().item()

        self.assertLess(spread(1.0), spread(3.0))
        self.assertLess(spread(3.0), spread(15.0))

    def test_tiny_closeness_still_draws_distinct_keys(self):
        # float32 gaussian weights are exactly 0 past ~13 std. The floor keeps
        # multinomial from running out of keys to pick.
        notes = draw_note_set(
            torch.zeros(100), settings(min_notes=5, max_notes=5, closeness=0.01)
        )
        for row in range(100):
            self.assertEqual(len(set(notes.key[row].tolist())), 5)

    def test_next_set_starts_after_every_note_ends_plus_rest(self):
        s = settings(min_notes=3, max_notes=3, min_rest=0.25, max_rest=0.5)
        first = draw_note_set(torch.zeros(200), s)
        second = draw_next_note_set(first, s)
        last_end = (first.start + first.duration).amax(-1)
        gap = second.start[:, 0] - last_end
        self.assertTrue(((gap >= 0.25 - 1e-5) & (gap <= 0.5 + 1e-5)).all())


class BatchedScoreTests(unittest.TestCase):
    def test_per_env_scores_match_one_score_per_env(self):
        torch.manual_seed(1)
        notes = draw_note_set(torch.zeros(4), settings(min_notes=1, max_notes=3))
        batched = Score(notes.start, notes.duration, notes.key)
        for time in [-1.0, 0.0, 0.7, 5.0]:
            got = batched.targets(torch.full((4, 1), time), NUM_KEYS)
            for env in range(4):
                real = notes.start[env].isfinite()
                single = Score(
                    notes.start[env][real],
                    notes.duration[env][real],
                    notes.key[env][real],
                )
                want = single.targets(torch.full((1, 1), time), NUM_KEYS)
                for name in want:
                    torch.testing.assert_close(got[name][env], want[name][0])

    def test_padding_does_not_leak_into_next_note(self):
        # One real note on key 7 and two pads. Pads sit on keys 0 and 7.
        score = Score(
            start=torch.tensor([[2.0, math.inf, math.inf]]),
            duration=torch.tensor([[1.0, 0.0, 0.0]]),
            key=torch.tensor([[7, 0, 7]]),
        )
        got = score.targets(torch.tensor([[0.0]]), NUM_KEYS)
        self.assertEqual(got["next_note_valid"][0].nonzero().flatten().tolist(), [7])
        self.assertEqual(got["next_note_time"][0, 7].item(), 2.0)
        self.assertEqual(got["next_note_duration"][0, 7].item(), 1.0)
        self.assertEqual(got["target_active"].sum().item(), 0)
        self.assertEqual(got["next_note_duration"][0, 0].item(), 0)


class NoteStreamTests(unittest.TestCase):
    @override
    def setUp(self):
        torch.manual_seed(0)

    def now(self, *times: float) -> torch.Tensor:
        return torch.tensor(times).unsqueeze(-1)

    def test_starts_with_a_set_at_zero_and_another_after(self):
        stream = NoteStream(3, settings(), CPU)
        score = stream.score()
        self.assertEqual(score.start.shape, (3, 6))
        starts = score.start.reshape(3, 2, 3)
        self.assertTrue((starts[:, CURRENT, 0] == 0).all())
        self.assertTrue((starts[:, NEXT, 0] > 0.5).all())
        # The lead-in (negative time) shows the first set as upcoming, not sounding.
        targets = score.targets(self.now(-0.5, -0.5, -0.5), NUM_KEYS)
        self.assertEqual(targets["target_active"].sum().item(), 0)
        self.assertTrue((targets["next_note_valid"].sum(-1) >= 1).all())

    def test_advance_promotes_the_next_set_and_keeps_one_ahead(self):
        stream = NoteStream(2, settings(), CPU)
        onset = stream.score().start.reshape(2, 2, 3)[:, NEXT, 0].clone()
        stream.advance(onset.unsqueeze(-1) + 0.01)
        starts = stream.score().start.reshape(2, 2, 3)
        # The old next set is now current and a new next set is waiting.
        torch.testing.assert_close(starts[:, CURRENT, 0], onset)
        self.assertTrue((starts[:, NEXT, 0] > onset + 0.01).all())

    def test_advance_only_moves_envs_that_are_due(self):
        stream = NoteStream(2, settings(), CPU)
        before = stream.score().start.clone()
        onset = before.reshape(2, 2, 3)[0, NEXT, 0].item()
        stream.advance(self.now(onset + 0.01, 0.0))
        after = stream.score().start
        self.assertFalse(torch.equal(after[0], before[0]))
        self.assertTrue(torch.equal(after[1], before[1]))

    def test_advance_catches_up_after_a_long_jump(self):
        # train.py starts envs at a random episode length, so the clock can leap.
        stream = NoteStream(4, settings(), CPU)
        now = self.now(10.0, 25.0, 3.0, 0.0)
        stream.advance(now)
        starts = stream.score().start.reshape(4, 2, 3)
        self.assertTrue((starts[:, CURRENT, 0] <= now[:, 0]).all())
        self.assertTrue((starts[:, NEXT, 0] > now[:, 0]).all())

    def test_advance_never_leaves_a_gap_in_time(self):
        # Walk a clock forward in small steps. Something is always playing or about
        # to play, and the next set never overlaps the current one.
        stream = NoteStream(1, settings(min_rest=0.0, max_rest=0.0), CPU)
        for step in range(400):
            now = self.now(step * 0.05)
            stream.advance(now)
            window = stream.score()
            starts = window.start.reshape(2, 3)
            ends = (window.start + window.duration).reshape(2, 3)
            current_end = ends[CURRENT][starts[CURRENT].isfinite()].max()
            self.assertGreaterEqual(starts[NEXT, 0].item(), current_end.item() - 1e-5)

    def test_reset_redraws_only_those_envs(self):
        stream = NoteStream(3, settings(), CPU)
        stream.advance(self.now(20.0, 20.0, 20.0))
        mid = stream.score().start.clone()
        stream.reset(torch.tensor([1]))
        after = stream.score().start
        self.assertTrue(torch.equal(after[0], mid[0]))
        self.assertTrue(torch.equal(after[2], mid[2]))
        self.assertEqual(after[1].reshape(2, 3)[CURRENT, 0].item(), 0.0)

    def test_same_seed_gives_same_notes(self):
        torch.manual_seed(5)
        a = NoteStream(2, settings(), CPU).score()
        torch.manual_seed(5)
        b = NoteStream(2, settings(), CPU).score()
        self.assertTrue(torch.equal(a.start, b.start))
        self.assertTrue(torch.equal(a.key, b.key))


if __name__ == "__main__":
    unittest.main()

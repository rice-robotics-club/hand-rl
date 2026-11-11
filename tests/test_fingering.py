"""Finger assignment on synthetic fingertip distances."""

import unittest

import torch

from src.music.fingering import closest_fingers, finger_distances, home_fingers


class FingerAssignmentTests(unittest.TestCase):
    def test_chord_gets_distinct_fingers_with_least_total_distance(self):
        # Finger 0 is nearest both keys, but giving key 1 to finger 1 costs less overall.
        distances = torch.full((1, 5, 3), 1.0)
        distances[0, 0, :2] = torch.tensor([0.1, 0.2])
        distances[0, 1, :2] = torch.tensor([0.9, 0.3])
        fingers = closest_fingers(distances, torch.tensor([[1.0, 1, 0]]))
        self.assertEqual(fingers.tolist(), [[0, 1, -1]])
        torch.testing.assert_close(
            finger_distances(distances, fingers), torch.tensor([[0.1, 0.3, 1.0]])
        )

    def test_rests_and_extra_keys_are_unassigned(self):
        distances = torch.rand(2, 5, 8)
        active = torch.tensor([[0.0] * 8, [1.0] * 6 + [0.0] * 2])
        fingers = closest_fingers(distances, active)
        self.assertTrue((fingers[0] == -1).all())
        assigned = fingers[1][fingers[1] >= 0]
        self.assertEqual(len(assigned), 5)
        self.assertEqual(len(set(assigned.tolist())), 5)
        self.assertTrue((fingers[1, 6:] == -1).all())
        # Unassigned keys still report a measured distance: the nearest finger.
        nearest = finger_distances(distances, fingers)
        self.assertTrue(torch.equal(nearest[0], distances[0].min(0).values))

    def test_home_fingers_follow_the_home_position(self):
        active = torch.zeros(1, 88)
        active[0, [39, 43, 50]] = 1  # C4 (thumb), E4 (middle), and B4 (no finger).
        finger_keys = torch.tensor([39, 41, 43, 44, 46])  # C4 D4 E4 F4 G4.
        fingers = home_fingers(active, finger_keys)
        self.assertEqual(fingers[0, [39, 43, 50]].tolist(), [0, 2, -1])
        self.assertEqual((fingers >= 0).sum().item(), 2)


if __name__ == "__main__":
    unittest.main()

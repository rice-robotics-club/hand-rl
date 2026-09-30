import json
from pathlib import Path
import unittest

from notes_to_fingers import notes_to_fingers


class FingerPlannerTests(unittest.TestCase):
    def test_reference_exercise(self):
        path = Path(__file__).parent / "samples/right_hand/expected.notes.json"
        plan = notes_to_fingers(json.loads(path.read_text()))
        self.assertEqual([n["finger"] for n in plan["notes"]],
                         [1, 2, 3, 4, 5, 3, 1, 3, 2, 4, 1, 3, 5])
        final_press = next(e for e in plan["events"] if e["timestamp"] == 6)
        self.assertEqual([a["finger"] for a in final_press["press"]], [1, 3, 5])
        self.assertEqual(plan["events"][-1]["timestamp"], 8)
        # Simulate the key states to ensure every press has a valid release.
        held = set()
        for event in plan["events"]:
            for action in event["release"]:
                self.assertIn(action["finger"], held)
                held.remove(action["finger"])
            for action in event["press"]:
                self.assertNotIn(action["finger"], held)
                held.add(action["finger"])
        self.assertEqual(held, set())

    def test_overlaps_on_different_keys_preserve_holds(self):
        plan = notes_to_fingers([(1, 0.5, 64), (0, 2, 60)])
        self.assertEqual([n["finger"] for n in plan["notes"]], [1, 3])
        self.assertEqual(plan["events"][-1],
                         {"timestamp": 2.0, "release": [{"finger": 1, "pitch": 60}], "press": []})

    def test_repeated_key_releases_before_repress(self):
        plan = notes_to_fingers([(0, 1, 60), (1, 1, 60)])
        event = plan["events"][1]
        self.assertEqual(event["timestamp"], 1)
        self.assertEqual(event["release"], event["press"])

    def test_same_key_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, "still holding"):
            notes_to_fingers([(0, 2, 60), (1, 1, 60)])

    def test_out_of_position_rejected_and_custom_position_supported(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            notes_to_fingers([(0, 1, 72)])
        plan = notes_to_fingers([(0, 1, 72)], keys=(72, 74, 76, 77, 79))
        self.assertEqual(plan["notes"][0]["finger"], 1)

    def test_invalid_inputs(self):
        for row in ((-1, 1, 60), (0, 0, 60), (float("nan"), 1, 60),
                    (0, float("inf"), 60), (0, 1, 60.5), (0, 1), (True, 1, 60)):
            with self.subTest(row=row), self.assertRaises(ValueError):
                notes_to_fingers([row])
        with self.assertRaises(ValueError):
            notes_to_fingers([], keys=(60, 60, 64, 65, 67))
        self.assertEqual(notes_to_fingers([])["events"], [])


if __name__ == "__main__":
    unittest.main()

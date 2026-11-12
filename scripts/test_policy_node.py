"""Checkpoint round-trip tests using the real RSL-RL actor, without Genesis."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from src.deployment.policy_node import PolicyNode


class PolicyNodeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.checkpoint = self.root / "model.pt"
        self.config = self.root / "training.yaml"
        self.layout = self.root / "layout.json"
        self.actor_config = {
            "hidden_dims": [8],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {"class_name": "GaussianDistribution"},
        }
        self.config.write_text(
            yaml.safe_dump(
                {
                    "system": {
                        "class_name": "src.sim.hand_piano:HandPiano",
                        "genesis": {"sim_options": {"dt": 0.05}},
                    },
                    "runner": {
                        "actor": {"class_name": "MLPModel", **self.actor_config},
                        "obs_groups": {"actor": ["position", "target"]},
                    },
                }
            )
        )
        # Deliberately reverse JSON order: training config determines concatenation.
        self.spec = {
            "num_actions": 2,
            "observation_sizes": {"target": 2, "position": 3},
        }
        self.layout.write_text(json.dumps(self.spec))
        self.obs = TensorDict(
            {"position": torch.randn(4, 3) + 5, "target": torch.randn(4, 2)},
            batch_size=[4],
        )
        self.actor = MLPModel(
            self.obs, {"actor": ["position", "target"]}, "actor", 2, **self.actor_config
        )
        self.actor.update_normalization(self.obs)
        self.actor.eval()
        torch.save(
            {"actor_state_dict": self.actor.state_dict(), "iter": 12, "infos": None},
            self.checkpoint,
        )

    def load(self):
        return PolicyNode(self.checkpoint, self.config, self.layout)

    def test_restores_identical_outputs_and_normalization(self):
        node = self.load()
        self.assertEqual(list(node.observation_sizes), ["position", "target"])
        self.assertFalse(node.policy.training)
        self.assertTrue(all(not p.requires_grad for p in node.policy.parameters()))
        with torch.inference_mode():
            expected = self.actor(self.obs)
            actual = node.policy(self.obs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(node.policy(self.obs), actual, rtol=0, atol=0)
        for name, tensor in self.actor.state_dict().items():
            torch.testing.assert_close(
                node.policy.state_dict()[name], tensor, rtol=0, atol=0
            )

    def test_missing_checkpoint(self):
        self.checkpoint.unlink()
        with self.assertRaises(FileNotFoundError):
            self.load()

    def test_node_reads_inputs_at_its_training_rate(self):
        node = self.load()
        self.assertEqual(node.control_rate_hz, 20)
        source = SimpleNamespace(
            read_sensors=lambda: {"position": [0, 0, 0]},
            read_goal=lambda elapsed: {"target": [1, 0]},
        )
        frames = []
        self.assertEqual(node.read_inputs(source, frames.append, max_frames=1), 1)
        self.assertEqual(frames[0].sensors["position"], [0, 0, 0])
        self.assertEqual(frames[0].goal["target"], [1, 0])

    def test_wrong_checkpoint_format(self):
        torch.save({"model_state_dict": {}}, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "actor_state_dict"):
            self.load()

    def test_incompatible_action_count(self):
        self.spec["num_actions"] = 3
        self.layout.write_text(json.dumps(self.spec))
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.load()

    def test_missing_normalizer_is_rejected(self):
        state = {
            k: v
            for k, v in self.actor.state_dict().items()
            if not k.startswith("obs_normalizer.")
        }
        torch.save({"actor_state_dict": state}, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.load()

    def test_bad_observation_names(self):
        self.spec["observation_sizes"] = {"wrong": 5}
        self.layout.write_text(json.dumps(self.spec))
        with self.assertRaisesRegex(ValueError, "exactly match"):
            self.load()


if __name__ == "__main__":
    unittest.main()

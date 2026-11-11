"""RukaSim and RukaEnv timing and state, with fake Genesis objects."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import genesis as gs
import torch

from src.env.ruka_env import RukaEnv, RukaEnvConfig
from src.sim.ruka_sim import (
    NUM_DOFS,
    RukaAction,
    RukaSim,
    RukaSimConfig,
    RukaState,
)
from src.sim.vec_system import VecSystem

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity


def make_state():
    def zeros(width: int) -> torch.Tensor:
        return torch.zeros(2, width)

    return RukaState(
        position=zeros(3),
        orientation=torch.tensor([[1.0, 0, 0, 0]]).repeat(2, 1),
        velocity=zeros(3),
        joint_positions=zeros(7),
        joint_velocities=zeros(7),
        time=torch.zeros(2, 1, dtype=torch.float64),
        control_dt=torch.full((2, 1), 1 / 60, dtype=torch.float64),
        fingertip_positions=zeros(15),
        fingertip_valid=zeros(5),
        key_pitches=torch.tensor([[60.0, 62, 64, 65, 67]]).repeat(2, 1),
        key_displacement=zeros(5),
        key_pressed=zeros(5),
        key_valid=zeros(5),
        batch_size=[2],
    )


class StateTests(unittest.TestCase):
    def test_environment_uses_system_time_and_returns_independent_snapshots(self):
        state = make_state()
        # Only the parts of a RukaSim that RukaEnv reads.
        sim = cast(
            VecSystem[RukaState, RukaAction, RukaEnvConfig],
            SimpleNamespace(state=state, num_envs=2, num_actions=NUM_DOFS),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notes.json"
            path.write_text(json.dumps([[0, 0.5, 60]]))
            env = RukaEnv(
                sim,
                RukaEnvConfig(
                    class_name="src.env.ruka_env:RukaEnv", notes_path=str(path)
                ),
            )
        old = env.get_observations()
        state.time[1] = 0.5
        state.joint_positions[1] = 1
        current = env.get_observations()
        self.assertEqual(current["target_active"][:, 0].tolist(), [1, 0])
        self.assertEqual(old["joint_positions"].sum().item(), 0)
        self.assertFalse(current["key_valid"].any())
        self.assertFalse(current["fingertip_valid"].any())
        # Iterating a TensorDict walks its batch dimension, so ask for the keys.
        keys = current.keys()
        self.assertTrue(all(current[k].dtype == torch.float32 for k in keys))

    def test_sim_step_and_partial_reset_refresh_clock_and_measurements(self):
        # Exercise the real RukaSim step/reset methods with a small fake backend.
        sim = RukaSim.__new__(RukaSim)
        sim._num_envs, sim._dt = 2, 1 / 60
        sim._steps = torch.zeros(2, 1, dtype=torch.int64)
        sim._state = make_state()
        sim._cfg = RukaSimConfig(
            class_name="src.sim.ruka_sim:RukaSim", action_scale=0.5
        )
        sim._default_dofs = torch.zeros(NUM_DOFS)
        sim._dofs_lower = torch.full((NUM_DOFS,), -torch.inf)
        sim._dofs_upper = torch.full((NUM_DOFS,), torch.inf)
        position = torch.ones(2, 3)
        joints = torch.ones(2, 7)

        def reset(envs_idx: int | torch.Tensor | None):
            index = slice(None) if envs_idx is None else envs_idx
            position[index] = 0
            joints[index] = 0

        sim._scene = cast(gs.Scene, SimpleNamespace(step=lambda: None, reset=reset))
        sim._robot = cast(
            "RigidEntity",
            SimpleNamespace(
                control_dofs_position=lambda _: None,
                get_pos=lambda: position,
                get_quat=lambda: sim._state.orientation,
                get_vel=lambda: position,
                get_dofs_position=lambda: joints,
                get_dofs_velocity=lambda: joints,
            ),
        )
        for _ in range(30):
            sim.step(RukaAction.from_tensor(torch.zeros(2, NUM_DOFS)))
        self.assertEqual(sim.state.time.tolist(), [[0.5], [0.5]])
        sim.reset(0)
        self.assertEqual(sim.state.time.tolist(), [[0], [0.5]])
        self.assertEqual(sim.state.joint_positions[:, 0].tolist(), [0, 1])
        sim.reset()
        self.assertFalse(sim.state.time.any())
        self.assertFalse(sim.state.position.any())

    def test_configuration_rejects_bad_timing_or_keyboard(self):
        for overrides in [
            {"control_dt": 0},
            {"control_dt": float("inf")},
            {"physics_substeps": 0},
            {"key_pitches": [60, 60]},
            {"key_pitches": [128]},
            {"key_pitches": []},
        ]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                RukaSimConfig(class_name="src.sim.ruka_sim:RukaSim", **overrides)


if __name__ == "__main__":
    unittest.main()

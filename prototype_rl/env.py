"""PROTOTYPE ONLY: vectorized key-state tracking with five abstract finger bends.

There are no meshes, contacts, torques, collisions, audio, or hardware commands.
"""

import math

import torch
from tensordict import TensorDict
from rsl_rl.env import VecEnv

from notes_to_fingers import DEFAULT_KEYS, notes_to_fingers


class PrototypeFingerEnv(VecEnv):
    num_actions = 5

    def __init__(self, notes, *, num_envs=32, device="cpu", dt=0.05,
                 bend_speed=4.0, press_threshold=0.75, lookahead_steps=6,
                 lead_in_seconds=0.5, release_tail_seconds=0.5):
        for name, value in (("dt", dt), ("bend_speed", bend_speed),
                            ("lead_in_seconds", lead_in_seconds),
                            ("release_tail_seconds", release_tail_seconds)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if type(lookahead_steps) is not int or lookahead_steps < 1:
            raise ValueError("lookahead_steps must be a positive integer")
        if not 0 < press_threshold < 1:
            raise ValueError("press_threshold must lie between 0 and 1")
        plan = notes_to_fingers(notes)
        if not plan["notes"]:
            raise ValueError("Training needs at least one target note")
        self.num_envs, self.device = num_envs, torch.device(device)
        self.dt, self.bend_speed = dt, bend_speed
        self.press_threshold, self.lookahead_steps = press_threshold, lookahead_steps
        self.cfg = dict(prototype=True, model="abstract_bends_without_physics",
                        keys=list(DEFAULT_KEYS), num_envs=num_envs, dt=dt,
                        bend_speed=bend_speed, press_threshold=press_threshold,
                        lookahead_steps=lookahead_steps, lead_in_seconds=lead_in_seconds,
                        release_tail_seconds=release_tail_seconds)
        end = max(n["start"] + n["duration"] for n in plan["notes"])
        self.max_episode_length = math.ceil((lead_in_seconds + end + release_tail_seconds) / dt)
        self.targets = torch.zeros(self.max_episode_length + lookahead_steps, 5, device=self.device)
        for n in plan["notes"]:
            start = math.ceil((n["start"] + lead_in_seconds) / dt - 1e-9)
            stop = math.ceil((n["start"] + n["duration"] + lead_in_seconds) / dt - 1e-9)
            if stop <= start:
                raise ValueError("A note is shorter than the time grid; reduce dt")
            self.targets[start:stop, n["finger"] - 1] = 1
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.bend = torch.zeros(num_envs, 5, device=self.device)
        self.velocity = torch.zeros_like(self.bend)
        self.last_action = torch.full_like(self.bend, -1)

    @torch.inference_mode()
    def reset(self, ids=None):
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        self.episode_length_buf[ids] = 0
        self.bend[ids] = 0
        self.velocity[ids] = 0
        self.last_action[ids] = -1
        return self.get_observations()

    def get_observations(self):
        indices = self.episode_length_buf[:, None] + torch.arange(self.lookahead_steps, device=self.device)
        future = self.targets[indices].flatten(1)
        phase = self.episode_length_buf[:, None] / self.max_episode_length
        policy = torch.cat((self.bend, self.velocity / self.bend_speed,
                            self.last_action, future, phase), dim=1)
        return TensorDict({"policy": policy}, batch_size=[self.num_envs], device=self.device)

    def step(self, actions):
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"Expected actions shaped ({self.num_envs}, 5)")
        actions = actions.to(self.device)
        if not torch.isfinite(actions).all():
            raise ValueError("Actions must be finite")
        # An action is a desired bend: -1 means raised, +1 means fully pressed.
        actions = actions.clamp(-1, 1)
        target = self.targets[self.episode_length_buf]
        desired = (actions + 1) / 2
        delta = (desired - self.bend).clamp(-self.bend_speed * self.dt, self.bend_speed * self.dt)
        self.bend = (self.bend + delta).clamp(0, 1)
        self.velocity = delta / self.dt
        pressed = self.bend >= self.press_threshold
        expected = target.bool()
        missing = (expected & ~pressed).sum(dim=1).float()
        extra = (~expected & pressed).sum(dim=1).float()
        tracking = (self.bend - target).square().mean(dim=1)
        effort = (actions - self.last_action).square().mean(dim=1)
        reward = 1 - tracking - 0.5 * missing - 0.5 * extra - 0.01 * effort
        self.last_action = actions.clone()
        self.episode_length_buf += 1
        done = self.episode_length_buf >= self.max_episode_length
        extras = {
            # End of the score is a true finite-task terminal, not a truncation.
            "time_outs": torch.zeros_like(done),
            "log": {"/prototype/tracking_mse": tracking.mean(),
                    "/prototype/missing_keys": missing.mean(),
                    "/prototype/extra_keys": extra.mean()},
            "prototype_pressed": pressed.clone(),
            "prototype_target": expected.clone(),
        }
        if done.any():
            self.reset(done.nonzero(as_tuple=False).flatten())
        return self.get_observations(), reward, done, extras

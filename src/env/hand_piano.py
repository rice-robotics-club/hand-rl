"""Score-following rewards for the existing hand/piano system."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal, override

import torch
from pydantic import Field, field_validator
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from torch import Tensor

from src.base_config import Config
from src.env.piano_targets import PianoTargets

if TYPE_CHECKING:
    from src.sim import VecSystem
    from src.sim.hand_piano import HandPianoAction, HandPianoState


def key_match_terms(
    pressed: Tensor, target: Tensor, threshold: float
) -> dict[str, Tensor]:
    """Compare activated keys with the score, without rewarding 87 idle keys.

    Hits and misses are normalized by chord size; each extra key costs one.
    Rests produce zero hits/misses and still penalize extra keys.
    """
    actual, required = pressed >= threshold, target.bool()
    count = required.sum(-1).clamp_min(1)
    hits = (actual & required).sum(-1)
    misses = (~actual & required).sum(-1)
    extras = (actual & ~required).sum(-1)
    return {
        "correct_notes": hits / count,
        "missed_notes": -misses / count,
        "extra_notes": -extras.float(),
        "note_correctness": torch.where(
            (required | actual).any(-1),
            hits / (hits + misses + extras).clamp_min(1),
            torch.ones_like(hits, dtype=torch.float32),
        ),
    }


def proximity_reward(distances: Tensor, target: Tensor, scale: float) -> Tensor:
    """Mean 1-tanh(d/scale) over active keys; zero during rests.

    Distances must be measured 3D distances in meters for assigned fingertip/key
    pairs, in the same order as pressed. Not pitch or hand-base errors.
    """
    if distances.shape != target.shape:
        raise ValueError("Fingertip/key distances must match the target key shape")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Nearness scale must be finite and positive")
    if not torch.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("Fingertip/key distances must be finite and nonnegative")
    return ((1 - torch.tanh(distances / scale)) * target).sum(-1) / target.sum(
        -1
    ).clamp_min(1)


class HandPianoEnvConfig(Config):
    """Configuration for the HandPianoEnv environment."""

    class_name: Literal["src.env.hand_piano:HandPianoEnv"]

    notes_path: str = "samples/right_hand/expected.notes.json"
    """JSON (start seconds, duration seconds, MIDI pitch) rows; relative to working directory."""
    key_pitches: list[int] = Field(
        default_factory=lambda: list(range(21, 109)), min_length=88, max_length=88
    )
    """MIDI pitch of each pressed-state channel; must match piano DOF order."""
    max_episode_length: int | None = Field(default=None, gt=0)
    """Control steps per episode; None covers score, lead-in, and release tail."""
    lead_in_seconds: float = Field(default=0.5, ge=0, allow_inf_nan=False)
    """Preparation time before the first score note."""
    release_tail_seconds: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    """Time after the score to reward releasing held keys."""
    pressed_threshold: float = Field(default=0.5, gt=0, le=1)
    """Fraction of key travel counted as an actual press."""
    nearness_scale: float = Field(default=0.02, gt=0, allow_inf_nan=False)
    """Distance scale in meters for 1-tanh(distance/scale)."""
    reward_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "correct_notes": 1.0,
            "missed_notes": 1.0,
            "extra_notes": 1.0,
            "note_nearness": 0.0,
        }
    )
    """Nonnegative weights; nearness requires measured fingertip/key distances."""

    @field_validator("key_pitches")
    @classmethod
    def validate_pitches(cls, pitches):
        if set(pitches) != set(range(21, 109)):
            raise ValueError(
                "key_pitches must contain each piano MIDI pitch 21..108 once"
            )
        return pitches

    @field_validator("reward_weights")
    @classmethod
    def validate_weights(cls, weights):
        names = {"correct_notes", "missed_notes", "extra_notes", "note_nearness"}
        if set(weights) != names:
            raise ValueError(f"Specify all reward weights: {sorted(names)}")
        if any(not math.isfinite(w) or w < 0 for w in weights.values()):
            raise ValueError("Reward weights must be finite and nonnegative")
        return weights


class HandPianoEnv(VecEnv):
    """Follow a timed score using physical key activation, with optional nearness."""

    def __init__(
        self,
        system: VecSystem[HandPianoState, HandPianoAction],
        cfg: HandPianoEnvConfig,
    ):
        from src.reward import RewardFunction

        self._system = system
        self.cfg = cfg
        self.num_envs = system.num_envs
        self.num_actions = system.num_actions
        self.device = system.state.pressed.device
        # Use the scene's interval rather than an independent task timestep.
        self._dt = float(system.cfg.genesis.sim_options.dt)
        if not math.isfinite(self._dt) or self._dt <= 0:
            raise ValueError("The simulator timestep must be finite and positive")
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        notes = json.loads(Path(cfg.notes_path).read_text(encoding="utf-8-sig"))
        if not isinstance(notes, list) or not notes:
            raise ValueError("Piano training requires a nonempty JSON note list")
        PianoTargets(notes, cfg.key_pitches, self.device)
        shifted = [(s + cfg.lead_in_seconds, d, p) for s, d, p in notes]
        self._score = PianoTargets(shifted, cfg.key_pitches, self.device)
        end = max(s + d for s, d, _ in shifted) + cfg.release_tail_seconds
        needed = math.ceil(end / self._dt)
        self.max_episode_length = cfg.max_episode_length or needed
        if self.max_episode_length < needed:
            raise ValueError(
                f"Episode needs at least {needed} steps to cover this score and release tail"
            )
        if system.state.pressed.shape != (self.num_envs, len(cfg.key_pitches)):
            raise ValueError("Pressed-state channels must match configured key_pitches")
        if cfg.reward_weights["note_nearness"] > 0:
            distances = system.state.get("fingertip_key_distances", None)
            if distances is None:
                raise ValueError(
                    "Enable note_nearness only after the system supplies measured fingertip_key_distances"
                )
            proximity_reward(
                distances, torch.zeros_like(system.state.pressed), cfg.nearness_scale
            )
        self._reward_target = self._targets()["target_active"]
        self._reward = RewardFunction(
            [
                self.correct_notes,
                self.missed_notes,
                self.extra_notes,
                self.note_nearness,
            ],
            weights=cfg.reward_weights,
            batch_size=self.num_envs,
            device=self.device,
        )

    def _targets(self):
        time = self.episode_length_buf.to(torch.float64).unsqueeze(-1) * self._dt
        return self._score.sample(time)

    def _matches(self, state):
        return key_match_terms(
            state.pressed, self._reward_target, self.cfg.pressed_threshold
        )

    def correct_notes(self, state, _):
        """Fraction of requested notes pressed during this control interval."""
        return self._matches(state)["correct_notes"]

    def missed_notes(self, state, _):
        """Negative fraction of requested notes missing during this interval."""
        return self._matches(state)["missed_notes"]

    def extra_notes(self, state, _):
        """Negative count of unrequested keys, including early presses and late releases."""
        return self._matches(state)["extra_notes"]

    def note_nearness(self, state, _):
        """Optional spatial guidance; never infer distances from the base pose."""
        if self.cfg.reward_weights["note_nearness"] == 0:
            return torch.zeros(self.num_envs, device=self.device)
        return proximity_reward(
            state["fingertip_key_distances"],
            self._reward_target,
            self.cfg.nearness_scale,
        )

    @override
    def get_observations(self) -> TensorDict:
        obs = TensorDict(dict(self._system.state.items()), batch_size=self.num_envs)
        targets = self._targets()
        obs.update(targets)
        obs["note_correctness"] = key_match_terms(
            self._system.state.pressed,
            targets["target_active"],
            self.cfg.pressed_threshold,
        )["note_correctness"].unsqueeze(-1)
        return obs.clone()

    @override
    def step(self, actions: Tensor):
        from src.sim.hand_piano import HandPianoAction

        # Score the result against the interval the policy just acted on.
        self._reward_target = self._targets()["target_active"]
        action = HandPianoAction.from_tensor(actions)
        self._system.step(action)
        rewards = self._reward(self._system.state, action) * self._dt
        log = self._reward.log()  # Unweighted rates before multiplication by dt.
        self.episode_length_buf += 1
        time_outs = self.episode_length_buf >= self.max_episode_length
        reset_idx = time_outs.nonzero().flatten()
        if len(reset_idx) > 0:
            self._system.reset(reset_idx)
            self.episode_length_buf[reset_idx] = 0
        # Completing this finite score is a terminal task, not a truncated
        # continuing episode. Do not bootstrap a value from the next score.
        return (
            self.get_observations(),
            rewards,
            time_outs,
            {"time_outs": torch.zeros_like(time_outs), "log": log},
        )

    def key_metrics (self, active, goal, threshold): 
        num_envs = active.shape[0]
        precision= []
        recall = []
        f1_scores = []
        for env in range(num_envs):
            tp = 0
            fp = 0
            fn = 0
            for piano_key in range(88):
                pressed = active[env,piano_key]
                expected = goal[env, piano_key] 
                if pressed >= threshold and  expected >= threshold: 
                    tp += 1
                elif pressed >= threshold and  expected < threshold: 
                    fp += 1
                elif pressed < threshold and  expected >= threshold: 
                    fn += 1 
            if (tp + fp) == 0 and (tp + fn) == 0: 
                precision_value = 1.0 
                recall_value = 1.0
                f1_score = 1.0
            else:      
                precision_value = (tp/(tp+fp))
                recall_value = (tp/(tp+fn))
                if precision_value + recall_value > 0:
                    f1_score = (2*precision_value*recall_value)/(precision_value + recall_value)
                else: 
                    f1_score = 0.0
            precision.append(precision_value)
            recall.append(recall_value)
            f1_scores.append(f1_score)
        precision = torch.tensor(precision)
        recall = torch.tensor(recall)
        f1_scores = torch.tensor(f1_scores)
        return precision, recall, f1_scores 


"""Score-following rewards for the hand/piano system.

Every reward term is a pure function of an environment state and an action. Values a
reward needs are fields of HandPianoEnvState, each derived by its own function from
the system state and the score.
"""


import functools
import itertools
import json
import math
from pathlib import Path
from typing import Literal, override

import torch
from jaxtyping import Float, Int
from pydantic import Field, field_validator, model_validator
from rsl_rl.env import VecEnv
from torch import Tensor

from src.base_config import Config
from src.env.piano_targets import PianoTargets
from src.sim.hand_piano import HandPianoAction, HandPianoConfig, HandPianoState
from src.sim.vec_system import VecSystem


class HandPianoEnvState(HandPianoState):
    """The system state plus the score-derived fields that rewards and policies use.

    Score fields describe the score at the state's time. Rewards see the state after
    an action together with the score at the start of that action's interval.
    """

    target_active: Float[Tensor, "*batch 88"]
    """1 for keys the score wants held now."""
    target_remaining: Float[Tensor, "*batch 88"]
    """Seconds left in each active note; 0 for inactive keys."""
    next_note_valid: Float[Tensor, "*batch 88"]
    """1 for keys with an upcoming onset."""
    next_note_time: Float[Tensor, "*batch 88"]
    """Seconds until each key's next onset; 0 if none."""
    next_note_duration: Float[Tensor, "*batch 88"]
    """Duration of each key's next note; 0 if none."""
    key_down: Float[Tensor, "*batch 88"]
    """1 for keys pressed at least pressed_threshold of their travel; see keys_down."""
    note_correctness: Float[Tensor, "*batch 1"]
    """Intersection over union of down and wanted keys; see note_correctness."""
    target_fingers: Int[Tensor, "*batch 88"]
    """Finger (0 = thumb to 4 = pinky) assigned to each active key, else -1."""
    target_key_distances: Float[Tensor, "*batch 88"]
    """Meters from each key to its assigned fingertip, or the nearest one if none."""
    target_hit_velocity: Float[Tensor, "*batch 88"]
    """Wanted key speed at the hit for active keys; 0 for inactive keys."""
    planned_fingers: Int[Tensor, "*batch 88"]
    """Finger the fingering plan assigns each active key, else -1.

    Equal to target_fingers when no plan is configured.
    """
    planned_key_distances: Float[Tensor, "*batch 88"]
    """Meters from each key to its planned fingertip, or the nearest one if none."""
    previous_action: Float[Tensor, "*batch 27"]
    """The action before the one being judged (rewards) or chosen (observations).

    Clipped to [-1, 1], like the actions the system applies. Zero after a reset.
    """
    posture_error: Float[Tensor, "*batch 21"]
    """Wrist and finger angles minus the system's default (pianist) posture, radians."""
    orientation_error: Float[Tensor, "*batch 1"]
    """Angle in radians between the base orientation and its default orientation."""


# ---------------------------------------------------------------------------
# Derived state fields
# ---------------------------------------------------------------------------


def keys_down(pressed: Tensor, threshold: float) -> Tensor:
    """1.0 where a key has traveled at least `threshold` of the way down."""
    return (pressed >= threshold).float()


def key_match_terms(key_down: Tensor, target: Tensor) -> dict[str, Tensor]:
    """Compare down keys with the score, without rewarding 87 idle keys.

    Hits and misses are normalized by chord size; each extra key costs one.
    Rests produce zero hits/misses and still penalize extra keys.
    """
    actual, required = key_down.bool(), target.bool()
    count = required.sum(-1).clamp_min(1)
    hits = (actual & required).sum(-1)
    misses = (~actual & required).sum(-1)
    extra_keys = actual & ~required
    extras = extra_keys.sum(-1)
    return {
        "extra_keys": extra_keys,
        "correct_notes": hits / count,
        "missed_notes": -misses / count,
        "extra_notes": -extras.float(),
        "note_correctness": torch.where(
            (required | actual).any(-1),
            hits / (hits + misses + extras).clamp_min(1),
            torch.ones_like(hits, dtype=torch.float32),
        ),
    }


def assign_fingers(distances: Tensor, active: Tensor) -> Tensor:
    """Assign each active key its own finger, minimizing the total fingertip distance.

    distances is (envs, fingers, keys); active is (envs, keys). Returns (envs, keys)
    finger indices, -1 for keys left unassigned. Every assignment of up to
    `fingers` active keys to distinct fingers is checked, which is cheap for one
    hand (at most 5! = 120). If more keys are active than there are fingers, the
    extra keys are left unassigned.
    """
    envs, fingers, keys = distances.shape
    slots = min(fingers, keys)
    active = active.bool()
    # Pick up to `slots` active keys per environment (active keys sort first).
    key_idx = active.float().topk(slots, dim=-1).indices  # (envs, slots)
    valid = active.gather(-1, key_idx)
    slot_dist = distances.gather(-1, key_idx.unsqueeze(1).expand(-1, fingers, -1))
    perms = torch.tensor(
        list(itertools.permutations(range(fingers), slots)), device=distances.device
    )  # (perms, slots): the finger for each slot.
    cost = slot_dist[:, perms, torch.arange(slots, device=distances.device)]
    best = (cost * valid.unsqueeze(1)).sum(-1).argmin(-1)
    slot_fingers = torch.where(valid, perms[best], -1)
    result = torch.full((envs, keys), -1, device=distances.device)
    return result.scatter(-1, key_idx, slot_fingers)


def assigned_distances(distances: Tensor, fingers: Tensor) -> Tensor:
    """Distance from each key to its assigned finger, or to the nearest finger if none.

    distances is (envs, fingers, keys); fingers is (envs, keys) from assign_fingers or
    a fingering plan. Returns measured (envs, keys) distances, with no placeholders.
    """
    chosen = distances.gather(1, fingers.clamp_min(0).unsqueeze(1)).squeeze(1)
    return torch.where(fingers >= 0, chosen, distances.min(1).values)


class FingeringPlan:
    """Which finger should play each key over time, from a planned fingering.

    Reads the "notes" of a plan written by scripts/notes_to_fingers.py: entries with
    start, duration, pitch, and finger (1 = thumb to 5 = pinky). Active intervals are
    [start, end), like the score.
    """

    def __init__(self, path, score_notes, pitches, lead_in, device):
        plan = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        notes = plan["notes"] if isinstance(plan, dict) else plan
        rows = [(n["start"], n["duration"], n["pitch"], n["finger"]) for n in notes]
        if sorted(r[:3] for r in rows) != sorted(tuple(n) for n in score_notes):
            raise ValueError(
                f"Fingering plan {path} must contain exactly the score's notes"
            )
        if any(f not in range(1, 6) for *_, f in rows):
            raise ValueError("Fingering plan fingers must be 1 (thumb) to 5 (pinky)")
        key = {pitch: i for i, pitch in enumerate(pitches)}
        self._start = torch.tensor(
            [r[0] + lead_in for r in rows], dtype=torch.float64, device=device
        )
        self._end = self._start + torch.tensor(
            [r[1] for r in rows], dtype=torch.float64, device=device
        )
        self._finger = torch.tensor([r[3] - 1 for r in rows], device=device)
        self._key = torch.tensor([key[r[2]] for r in rows], device=device)
        self._num_keys = len(pitches)

    def sample(self, time: Tensor) -> Tensor:
        """Planned finger per key at each environment's time (envs, 1); -1 where none."""
        playing = (self._start <= time) & (time < self._end)  # (envs, notes)
        fingers = torch.where(playing, self._finger, -1)
        result = torch.full(
            (time.shape[0], self._num_keys), -1, device=time.device
        )
        # Notes not playing scatter -1 and leave the key unassigned.
        return result.scatter_reduce(
            -1, self._key.expand_as(fingers), fingers, reduce="amax"
        )


def orientation_error(orientation: Tensor, default: Tensor) -> Tensor:
    """Rotation angle in radians from `default` to each (w, x, y, z) quaternion, as (..., 1)."""
    dot = (orientation * default).sum(-1, keepdim=True).abs().clamp(max=1.0)
    return 2 * torch.acos(dot)


def flat_action(action: HandPianoAction) -> Tensor:
    """The action as one (..., 27) tensor clipped to [-1, 1], as the system applies it."""
    return torch.cat([action.base_pose, action.joint_positions], dim=-1).clamp(-1, 1)


def key_colors(
    key_down: Tensor,
    target: Tensor,
    target_color: tuple[float, float, float, float],
    correct_color: tuple[float, float, float, float],
    wrong_color: tuple[float, float, float, float],
) -> Tensor:
    """RGBA color per key showing correctness: (..., keys) inputs to (..., keys, 4).

    Wanted keys that are up get target_color, wanted keys that are down get
    correct_color, and unwanted keys that are down get wrong_color. Other keys are
    transparent (zero alpha).
    """
    down, wanted = key_down.bool().unsqueeze(-1), target.bool().unsqueeze(-1)
    def color(rgba):
        return torch.tensor(rgba, device=key_down.device)

    none = torch.zeros(4, device=key_down.device)
    return torch.where(
        wanted,
        torch.where(down, color(correct_color), color(target_color)),
        torch.where(down, color(wrong_color), none),
    )


# ---------------------------------------------------------------------------
# Reward helpers on plain tensors
# ---------------------------------------------------------------------------


def proximity_reward(distances: Tensor, target: Tensor, scale: float) -> Tensor:
    """Mean 1-tanh(d/scale) over active keys; zero during rests.

    Distances are measured 3D distances in meters from each key's press point to the
    fingertip assigned to it, in pressed order.
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


def key_intensity_reward(
    target_intensities: Tensor, key_velocities: Tensor, scale: float = 0.2
) -> Tensor:
    """Mean exp(-(err/scale)^2) over active keys, where err = intensity - target.

    Both inputs are (envs, keys). A key is active when its target intensity is
    above zero. Rests give zero. Peaks at 1 on the target, falls off smoothly.
    """
    # Pass state.hit_velocity: the latched speed from the step each key crossed
    # the press threshold. Active keys not yet hit read zero and score low.
    intensity = key_velocities.abs()
    active = (target_intensities > 0).float()
    err = (intensity - target_intensities) / scale
    score = torch.exp(-err.square()) * active
    return score.sum(-1) / active.sum(-1).clamp_min(1)

def wrong_key_reward(
        extra_keys: Tensor
) -> Tensor: 
    """
    flat scalar value. if there's any wrong notes, it's basically just 
    a flat alarm that activates. 
    
    intentionally choosing to avoid scaling per-wrong-note per isabella's advice,
    one of the research papers found that a flat discouragement did a better job of 
    making the robot still bold enough to explore playing
    """
    # return 1 for the ones that have any extra keys
    return -extra_keys.any(-1).float()


# ---------------------------------------------------------------------------
# Reward terms: pure functions of (state, action)
# ---------------------------------------------------------------------------


def correct_notes(state: HandPianoEnvState, _: HandPianoAction) -> Tensor:
    """Fraction of requested notes down during this control interval."""
    return key_match_terms(state.key_down, state.target_active)["correct_notes"]


def missed_notes(state: HandPianoEnvState, _: HandPianoAction) -> Tensor:
    """Negative fraction of requested notes not down during this interval."""
    return key_match_terms(state.key_down, state.target_active)["missed_notes"]


def extra_notes(state: HandPianoEnvState, _: HandPianoAction) -> Tensor:
    """Negative count of unrequested keys, including early presses and late releases."""
    return key_match_terms(state.key_down, state.target_active)["extra_notes"]


def wrong_keys(state: HandPianoEnvState, _: HandPianoAction) -> Tensor:
    """Flat -1 per env while any unrequested key is down."""
    return wrong_key_reward(
        key_match_terms(state.key_down, state.target_active)["extra_keys"]
    )


def note_nearness(
    state: HandPianoEnvState, _: HandPianoAction, *, scale: float
) -> Tensor:
    """How close each requested key is to the fingertip assigned to play it."""
    return proximity_reward(state.target_key_distances, state.target_active, scale)


def key_intensity(
    state: HandPianoEnvState, _: HandPianoAction, *, scale: float
) -> Tensor:
    """How close the hit-time key speed is to the target, on active keys."""
    return key_intensity_reward(state.target_hit_velocity, state.hit_velocity, scale)


def finger_nearness(
    state: HandPianoEnvState, _: HandPianoAction, *, scale: float
) -> Tensor:
    """How close each requested key is to the fingertip the fingering plan assigns."""
    return proximity_reward(state.planned_key_distances, state.target_active, scale)


def action_rate(state: HandPianoEnvState, action: HandPianoAction) -> Tensor:
    """Negative mean squared change between consecutive actions; discourages jerk."""
    return -(flat_action(action) - state.previous_action).square().mean(-1)


def finger_posture(
    state: HandPianoEnvState, _: HandPianoAction, *, scale: float
) -> Tensor:
    """exp(-(rms/scale)^2) of the joints' deviation from the pianist posture; 1 at rest."""
    rms = state.posture_error.square().mean(-1).sqrt()
    return torch.exp(-((rms / scale) ** 2))


def hand_orientation(
    state: HandPianoEnvState, _: HandPianoAction, *, scale: float
) -> Tensor:
    """exp(-(angle/scale)^2) of the base's rotation away from level, palm down."""
    return torch.exp(-((state.orientation_error[..., 0] / scale) ** 2))


def _bind(term, **params):
    """Fixes a term's hyperparameters, keeping its name for weights and logging."""
    return functools.update_wrapper(functools.partial(term, **params), term)


REWARD_TERMS = (
    "correct_notes",
    "missed_notes",
    "extra_notes",
    "note_nearness",
    "finger_nearness",
    "key_intensity",
    "wrong_keys",
    "action_rate",
    "finger_posture",
    "hand_orientation",
)
"""Every reward term's name; each needs a weight in the config."""


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
    fingering_path: str | None = None
    """Optional fingering plan (from scripts/notes_to_fingers.py) for finger_nearness.

    note_nearness always measures each active key from the closest distinct finger.
    finger_nearness measures it from the plan's finger, or matches note_nearness if
    this is None.
    """
    reward_weights: dict[str, float] = Field(
        default_factory=lambda: {name: 0.0 for name in REWARD_TERMS}
        | {"correct_notes": 1.0, "missed_notes": 1.0, "extra_notes": 1.0}
    )
    """Nonnegative weight per reward term; at curriculum progress 0 if one is set."""
    final_reward_weights: dict[str, float] | None = None
    """Weights at curriculum progress 1; weights move linearly between the two sets.

    None keeps reward_weights throughout. Progress rises with note recall; see
    curriculum_start and curriculum_end.
    """
    curriculum_start: float = Field(default=0.1, ge=0, le=1)
    """Average note recall at which curriculum progress starts rising from 0."""
    curriculum_end: float = Field(default=0.7, ge=0, le=1)
    """Average note recall at which curriculum progress reaches 1."""
    curriculum_rate: float = Field(default=0.2, gt=0, le=1)
    """How far one full batch of finished episodes moves the average note recall.

    Smaller is smoother; a reward that changes quickly destabilizes PPO's value
    estimates. Progress never decreases.
    """
    posture_scale: float = Field(default=0.3, gt=0, allow_inf_nan=False)
    """RMS joint deviation in radians at which finger_posture falls to 1/e."""
    orientation_scale: float = Field(default=0.3, gt=0, allow_inf_nan=False)
    """Base rotation in radians at which hand_orientation falls to 1/e."""
    target_hit_velocity: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    """Wanted key speed at the hit, in fractions of key travel per second.

    The score has no dynamics yet, so every note uses this one value. Print real
    hit_velocity values and tune this before turning key_intensity on.
    """
    color_correctness: bool = True
    """Color environment 0's keys in the viewer by correctness, as scored by the rewards.

    Wanted keys are target_color until pressed, then correct_color; unwanted pressed
    keys are wrong_color. Replaces the piano's color_pressed. Does nothing without a
    viewer.
    """
    target_color: tuple[float, float, float, float] = (0.2, 0.4, 1.0, 0.7)
    """RGBA color for keys the score wants that are not down (blue)."""
    correct_color: tuple[float, float, float, float] = (0.2, 0.9, 0.3, 0.7)
    """RGBA color for wanted keys that are down (green)."""
    wrong_color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.7)
    """RGBA color for unwanted keys that are down (red)."""
    intensity_scale: float = Field(default=0.2, gt=0, allow_inf_nan=False)
    """Width of the exp(-(err/scale)^2) bump around target_hit_velocity."""

    @field_validator("key_pitches")
    @classmethod
    def validate_pitches(cls, pitches):
        if set(pitches) != set(range(21, 109)):
            raise ValueError(
                "key_pitches must contain each piano MIDI pitch 21..108 once"
            )
        return pitches

    @field_validator("reward_weights", "final_reward_weights")
    @classmethod
    def validate_weights(cls, weights):
        if weights is None:
            return weights
        if set(weights) != set(REWARD_TERMS):
            raise ValueError(f"Specify all reward weights: {sorted(REWARD_TERMS)}")
        if any(not math.isfinite(w) or w < 0 for w in weights.values()):
            raise ValueError("Reward weights must be finite and nonnegative")
        return weights

    @model_validator(mode="after")
    def validate_curriculum(self):
        if self.curriculum_end <= self.curriculum_start:
            raise ValueError("curriculum_end must be above curriculum_start")
        return self


class HandPianoEnv(VecEnv):
    """Follow a timed score using physical key activation, with optional nearness."""

    def __init__(
        self,
        system: VecSystem[HandPianoState, HandPianoAction, HandPianoConfig],
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
        self._fingering = (
            FingeringPlan(
                cfg.fingering_path,
                notes,
                cfg.key_pitches,
                cfg.lead_in_seconds,
                self.device,
            )
            if cfg.fingering_path is not None
            else None
        )
        self._reward = RewardFunction[HandPianoEnvState, HandPianoAction](
            [
                correct_notes,
                missed_notes,
                extra_notes,
                _bind(note_nearness, scale=cfg.nearness_scale),
                _bind(finger_nearness, scale=cfg.nearness_scale),
                _bind(key_intensity, scale=cfg.intensity_scale),
                wrong_keys,
                action_rate,
                _bind(finger_posture, scale=cfg.posture_scale),
                _bind(hand_orientation, scale=cfg.orientation_scale),
            ],
            weights=cfg.reward_weights,
            batch_size=self.num_envs,
            device=self.device,
        )
        self._previous_action = torch.zeros(
            self.num_envs, self.num_actions, device=self.device
        )
        # Curriculum: per-episode note recall, and its running average across episodes.
        self._episode_hits = torch.zeros(self.num_envs, device=self.device)
        self._episode_required = torch.zeros(self.num_envs, device=self.device)
        self.note_recall = 0.0
        self.curriculum_progress = 0.0

    def _time(self) -> Tensor:
        """Each environment's score time in seconds, as (envs, 1)."""
        return self.episode_length_buf.to(torch.float64).unsqueeze(-1) * self._dt

    def state_at(self, time: Tensor, previous_action: Tensor) -> HandPianoEnvState:
        """The current system state with the score fields for `time`, as (envs, 1)."""
        system = self._system.state
        targets = self._score.sample(time)
        target = targets["target_active"]
        key_down = keys_down(system.pressed, self.cfg.pressed_threshold)
        distances = system.fingertip_key_distances
        fingers = assign_fingers(distances, target)
        planned = fingers if self._fingering is None else self._fingering.sample(time)
        return HandPianoEnvState(
            **dict(system.items()),
            **targets,
            key_down=key_down,
            note_correctness=key_match_terms(key_down, target)[
                "note_correctness"
            ].unsqueeze(-1),
            target_fingers=fingers,
            target_key_distances=assigned_distances(distances, fingers),
            target_hit_velocity=target * self.cfg.target_hit_velocity,
            planned_fingers=planned,
            planned_key_distances=assigned_distances(distances, planned),
            previous_action=previous_action,
            posture_error=system.joint_positions - self._system.default_joint_positions,
            orientation_error=orientation_error(
                system.orientation, self._system.default_orientation
            ),
            batch_size=[self.num_envs],
            device=self.device,
        )

    def _update_curriculum(self, finished: Tensor):
        """Folds finished episodes' note recall into the average, then sets weights."""
        required = self._episode_required[finished]
        played = required > 0
        if played.any():
            recall = (self._episode_hits[finished][played] / required[played]).mean()
            share = played.sum().item() / self.num_envs
            self.note_recall += (
                self.cfg.curriculum_rate * share * (recall.item() - self.note_recall)
            )
        self._episode_hits[finished] = 0
        self._episode_required[finished] = 0
        span = self.cfg.curriculum_end - self.cfg.curriculum_start
        progress = min(max((self.note_recall - self.cfg.curriculum_start) / span, 0), 1)
        # Never lower the bar: a dip in recall should not loosen the penalties again.
        self.curriculum_progress = max(self.curriculum_progress, progress)
        final = self.cfg.final_reward_weights
        if final is not None:
            start, p = self.cfg.reward_weights, self.curriculum_progress
            self._reward.set_weights(
                {name: start[name] + p * (final[name] - start[name]) for name in start}
            )

    def _show_correctness(self, state: HandPianoEnvState):
        """Colors environment 0's keys in the viewer by what the rewards just scored."""
        show = getattr(self._system, "show_key_colors", None)
        if not self.cfg.color_correctness or show is None:
            return
        show(
            key_colors(
                state.key_down[0],
                state.target_active[0],
                self.cfg.target_color,
                self.cfg.correct_color,
                self.cfg.wrong_color,
            )
        )

    @override
    def get_observations(self) -> HandPianoEnvState:
        # Clone so later simulator steps cannot mutate a returned observation.
        return self.state_at(self._time(), self._previous_action).clone()

    @override
    def step(self, actions: Tensor):
        action = HandPianoAction.from_tensor(actions)
        # Score the result against the interval the policy just acted on.
        interval_start = self._time()
        self._system.step(action)
        state = self.state_at(interval_start, self._previous_action)
        rewards = self._reward(state, action) * self._dt
        self._show_correctness(state)
        log = self._reward.log()  # Weighted rates, before multiplication by dt.
        self._previous_action = flat_action(action)
        wanted, down = state.target_active.bool(), state.key_down.bool()
        self._episode_hits += (wanted & down).sum(-1)
        self._episode_required += wanted.sum(-1)
        self.episode_length_buf += 1
        time_outs = self.episode_length_buf >= self.max_episode_length
        reset_idx = time_outs.nonzero().flatten()
        if len(reset_idx) > 0:
            self._update_curriculum(reset_idx)
            self._system.reset(reset_idx)
            self.episode_length_buf[reset_idx] = 0
            self._previous_action[reset_idx] = 0
        log["curriculum/note_recall"] = self.note_recall
        log["curriculum/progress"] = self.curriculum_progress
        # Completing this finite score is a terminal task, not a truncated
        # continuing episode. Do not bootstrap a value from the next score.
        return (
            self.get_observations(),
            rewards,
            time_outs,
            {"time_outs": torch.zeros_like(time_outs), "log": log},
        )


def key_metrics(active, goal, threshold=0.5):
    # Convert values into True/False predictions
    active_pressed = active >= threshold
    goal_pressed = goal >= threshold

    # Count true positives, false positives, and false negatives
    tp = (active_pressed & goal_pressed).sum(dim=-1).float()
    fp = (active_pressed & ~goal_pressed).sum(dim=-1).float()
    fn = (~active_pressed & goal_pressed).sum(dim=-1).float()

    # Calculate precision and recall
    precision = torch.where(
        tp + fp > 0,
        tp / (tp + fp).clamp(min=1),
        torch.where(tp + fn == 0, 1.0, 0.0)
    )

    recall = torch.where(
        tp + fn > 0,
        tp / (tp + fn).clamp(min=1),
        1.0
    )

    # Calculate F1 score
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall).clamp(min=1e-8),
        0.0
    )

    return precision, recall, f1

"""An environment for playing a score on the piano with the floating hand.

Every environment plays the same score, read from a JSON file. The pieces it uses:
    src/music/score.py       what the score asks of each key over time
    src/music/fingering.py   which finger should play each wanted key
    src/reward/piano.py      reward terms for playing notes
    src/reward/motion.py     reward terms for smooth motion and posture
    src/reward/curriculum.py how reward weights change as playing improves
"""

import math
from typing import Literal, override

import torch
from pydantic import Field, field_validator, model_validator
from rsl_rl.env import VecEnv
from torch import Tensor

from src.base_config import Config
from src.music.fingering import (
    HOME_PITCHES,
    closest_fingers,
    finger_distances,
    home_fingers,
)
from src.music.keyboard import PIANO_PITCHES, key_of
from src.music.notes import load_notes
from src.music.score import Score
from src.reward import RewardFunction, RewardTerm, bind, motion, piano
from src.reward.curriculum import Curriculum, RecallTracker
from src.reward.motion import MotionState
from src.reward.piano import RGBA, FingerTargets
from src.sim.hand_piano import HandPianoAction, HandPianoState, HandPianoSystem
from src.util.geometry import rotation_angle


class HandPianoEnvState(HandPianoState, FingerTargets, MotionState):
    """The system state plus the score-derived fields that rewards and policies use.

    Rewards see the state after an action, with the score as it was at the start of
    that action's interval. Each base class lists the fields one group of reward
    terms reads; see src/reward/piano.py and src/reward/motion.py.
    """


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

    # The score.
    notes_path: str = "data/right_hand/expected.notes.json"
    """JSON list of [start seconds, duration seconds, MIDI pitch] to play; relative to
    the working directory."""
    lead_in_seconds: float = Field(default=0.5, ge=0, allow_inf_nan=False)
    """Time to get ready before the score starts."""
    release_tail_seconds: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    """Time after the score to release held keys."""
    max_episode_length: int | None = Field(default=None, gt=0)
    """Control steps per episode; None fits the lead-in, the score, and the tail."""
    pressed_threshold: float = Field(default=0.5, gt=0, le=1)
    """Fraction of key travel that counts as a press."""
    finger_pitches: list[int] = Field(
        default_factory=lambda: list(HOME_PITCHES),
        min_length=len(HOME_PITCHES),
        max_length=len(HOME_PITCHES),
    )
    """The key under each finger, thumb to pinky, in the hand's starting position.

    finger_nearness wants each of these keys played by its own finger. The default,
    C4 D4 E4 F4 G4, matches HandPiano's default starting position.
    """

    # Reward settings.
    nearness_scale: float = Field(default=0.02, gt=0, allow_inf_nan=False)
    """Distance in meters for the nearness terms' 1 - tanh(distance / scale)."""
    target_hit_velocity: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    """Wanted key speed at the hit, in fractions of key travel per second.

    Every note uses this one value. Look at real hit_velocity values and tune this
    before turning key_intensity on.
    """
    intensity_scale: float = Field(default=0.2, gt=0, allow_inf_nan=False)
    """Hit speed error at which key_intensity falls to 1/e."""
    posture_scale: float = Field(default=0.3, gt=0, allow_inf_nan=False)
    """RMS joint error in radians at which finger_posture falls to 1/e."""
    orientation_scale: float = Field(default=0.3, gt=0, allow_inf_nan=False)
    """Base rotation in radians at which hand_orientation falls to 1/e."""

    # Reward weights and curriculum.
    reward_weights: dict[str, float] = Field(
        default_factory=lambda: (
            {name: 0.0 for name in REWARD_TERMS}
            | {"correct_notes": 1.0, "missed_notes": 1.0, "extra_notes": 1.0}
        )
    )
    """Nonnegative weight per reward term, at the start of the curriculum."""
    final_reward_weights: dict[str, float] | None = None
    """Weights at the end of the curriculum; None keeps reward_weights throughout."""
    curriculum_start: float = Field(default=0.1, ge=0, le=1)
    """Average note recall at which the weights start moving toward the final ones."""
    curriculum_end: float = Field(default=0.7, ge=0, le=1)
    """Average note recall at which the weights reach the final ones."""
    curriculum_rate: float = Field(default=0.2, gt=0, le=1)
    """How far one full batch of finished episodes moves the average note recall.

    Smaller is smoother; a reward that changes quickly destabilizes PPO's value
    estimates.
    """

    # Viewer colors.
    color_correctness: bool = True
    """Color environment 0's keys in the viewer: target_color for wanted keys that are
    up, correct_color for wanted keys that are down, wrong_color for unwanted keys that
    are down. Replaces the piano's color_pressed. Does nothing without a viewer."""
    target_color: RGBA = (0.2, 0.4, 1.0, 0.7)
    """Blue."""
    correct_color: RGBA = (0.2, 0.9, 0.3, 0.7)
    """Green."""
    wrong_color: RGBA = (1.0, 0.2, 0.2, 0.7)
    """Red."""

    @field_validator("reward_weights", "final_reward_weights")
    @classmethod
    def check_weights(cls, weights: dict[str, float] | None) -> dict[str, float] | None:
        if weights is None:
            return weights
        if set(weights) != set(REWARD_TERMS):
            raise ValueError(f"Specify all reward weights: {sorted(REWARD_TERMS)}")
        if any(not math.isfinite(w) or w < 0 for w in weights.values()):
            raise ValueError("Reward weights must be finite and nonnegative")
        return weights

    @field_validator("finger_pitches")
    @classmethod
    def check_finger_pitches(cls, pitches: list[int]) -> list[int]:
        if len(set(pitches)) != len(pitches) or not set(pitches) <= set(PIANO_PITCHES):
            raise ValueError(
                f"finger_pitches must be {len(HOME_PITCHES)} distinct piano pitches"
            )
        return pitches

    @model_validator(mode="after")
    def check_curriculum(self):
        if self.curriculum_end <= self.curriculum_start:
            raise ValueError("curriculum_end must be above curriculum_start")
        return self


class HandPianoEnv(VecEnv):
    """Play a timed score with the hand, judged by which keys are actually down."""

    def __init__(
        self,
        system: HandPianoSystem,
        cfg: HandPianoEnvConfig,
    ):
        self._system = system
        self.cfg: HandPianoEnvConfig = cfg  # pyright: ignore[reportIncompatibleVariableOverride]
        self.num_envs = system.num_envs
        self.num_actions = system.num_actions
        self.device = system.state.pressed.device
        if system.state.pressed.shape[-1] != len(PIANO_PITCHES):
            raise ValueError(f"The system's piano must have {len(PIANO_PITCHES)} keys")

        # Each step of the system is one control interval.
        self._dt = system.dt
        if not math.isfinite(self._dt) or self._dt <= 0:
            raise ValueError("The simulator timestep must be finite and positive")

        notes = load_notes(cfg.notes_path, PIANO_PITCHES)
        if not notes:
            raise ValueError(f"The score {cfg.notes_path} has no notes")
        self._score = Score.from_notes(notes, PIANO_PITCHES, self.device)
        seconds = cfg.lead_in_seconds + self._score.length() + cfg.release_tail_seconds
        needed = math.ceil(seconds / self._dt)
        if cfg.max_episode_length is not None and cfg.max_episode_length < needed:
            raise ValueError(
                f"Episodes need at least {needed} steps to fit the score and release tail"
            )
        self.max_episode_length = cfg.max_episode_length or needed
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        self._finger_keys = torch.tensor(
            [key_of(p) for p in cfg.finger_pitches], device=self.device
        )
        self._previous_action = torch.zeros(
            self.num_envs, self.num_actions, device=self.device
        )

        terms: list[RewardTerm[HandPianoEnvState, HandPianoAction]] = [
            piano.correct_notes,
            piano.missed_notes,
            piano.extra_notes,
            bind(piano.note_nearness, scale=cfg.nearness_scale),
            bind(piano.finger_nearness, scale=cfg.nearness_scale),
            bind(piano.key_intensity, scale=cfg.intensity_scale),
            piano.wrong_keys,
            motion.action_rate,
            bind(motion.finger_posture, scale=cfg.posture_scale),
            bind(motion.hand_orientation, scale=cfg.orientation_scale),
        ]
        self._reward = RewardFunction[HandPianoEnvState, HandPianoAction](
            terms,
            weights=cfg.reward_weights,
            batch_size=self.num_envs,
            device=self.device,
        )
        self.curriculum = Curriculum(
            cfg.reward_weights,
            cfg.final_reward_weights,
            recall_start=cfg.curriculum_start,
            recall_end=cfg.curriculum_end,
            rate=cfg.curriculum_rate,
        )
        self._recall = RecallTracker(self.num_envs, self.device)

    def _score_time(self) -> Tensor:
        """Each environment's position in its score, in seconds, as (envs, 1).

        Negative during the lead-in.
        """
        episode_time = self.episode_length_buf.to(torch.float64) * self._dt
        return (episode_time - self.cfg.lead_in_seconds).unsqueeze(-1)

    def state_at(
        self, score_time: Tensor, previous_action: Tensor
    ) -> HandPianoEnvState:
        """The system's current state, plus the score at `score_time` (envs, 1)."""
        system = self._system.state
        targets = self._score.targets(score_time, num_keys=len(PIANO_PITCHES))
        wanted = targets["target_active"]
        key_down = piano.keys_down(system.pressed, self.cfg.pressed_threshold)
        distances = system.fingertip_key_distances
        closest = closest_fingers(distances, wanted)
        home = home_fingers(wanted, self._finger_keys)
        return HandPianoEnvState(
            # The system's state.
            position=system.position,
            orientation=system.orientation,
            velocity=system.velocity,
            joint_positions=system.joint_positions,
            target_offsets=system.target_offsets,
            fingertip_positions=system.fingertip_positions,
            fingertip_key_distances=distances,
            pressed=system.pressed,
            hit_velocity=system.hit_velocity,
            # What the score wants, and how the keys compare.
            target_active=wanted,
            target_remaining=targets["target_remaining"],
            next_note_valid=targets["next_note_valid"],
            next_note_time=targets["next_note_time"],
            next_note_duration=targets["next_note_duration"],
            target_hit_velocity=wanted * self.cfg.target_hit_velocity,
            key_down=key_down,
            note_correctness=piano.score_notes(key_down, wanted).correctness.unsqueeze(
                -1
            ),
            closest_fingers=closest,
            closest_finger_distances=finger_distances(distances, closest),
            home_fingers=home,
            home_finger_distances=finger_distances(distances, home),
            previous_action=previous_action,
            posture_error=system.joint_positions - self._system.default_joint_positions,
            orientation_error=rotation_angle(
                system.orientation, self._system.default_orientation
            ),
            batch_size=[self.num_envs],
            device=self.device,
        )

    @override
    def get_observations(self) -> HandPianoEnvState:
        # Clone so later simulator steps cannot change a returned observation.
        return self.state_at(self._score_time(), self._previous_action).clone()

    @override
    def step(self, actions: Tensor):
        action = HandPianoAction.from_tensor(actions)

        # Judge the result against the score at the start of the interval just acted on.
        interval_start = self._score_time()
        self._system.step(action)
        state = self.state_at(interval_start, self._previous_action)
        reward = self._reward(state, action) * self._dt
        log = self._reward.log()  # Weighted rates, before multiplying by dt.
        log |= self._note_metrics(state)
        self._show_correctness(state)

        self._previous_action = motion.flat_action(action)
        self._recall.add(state.key_down, state.target_active)
        self.episode_length_buf += 1
        done = self.episode_length_buf >= self.max_episode_length
        self._reset(done.nonzero().flatten())
        log["curriculum/note_recall"] = self.curriculum.recall
        log["curriculum/progress"] = self.curriculum.progress

        # Finishing the score ends the task; it is not a time limit on a task that
        # would continue, so don't bootstrap a value past it.
        return (
            self.get_observations(),
            reward,
            done,
            {"time_outs": torch.zeros_like(done), "log": log},
        )

    def _reset(self, envs: Tensor):
        """Starts new episodes in `envs`."""
        if len(envs) == 0:
            return
        self.curriculum.record(self._recall.finish(envs), self.num_envs)
        self._reward.set_weights(self.curriculum.weights())
        self._system.reset(envs)
        self.episode_length_buf[envs] = 0
        self._previous_action[envs] = 0

    def _note_metrics(self, state: HandPianoEnvState) -> dict[str, float]:
        """Mean precision, recall, and F1 of the keys down, for logging."""
        precision, recall, f1 = piano.key_metrics(state.key_down, state.target_active)
        return {
            "notes/precision": precision.mean().item(),
            "notes/recall": recall.mean().item(),
            "notes/f1": f1.mean().item(),
        }

    def _show_correctness(self, state: HandPianoEnvState):
        """Colors environment 0's keys by what the rewards just scored."""
        if self.cfg.color_correctness:
            self._system.show_key_colors(
                piano.key_colors(
                    state.key_down[0],
                    state.target_active[0],
                    self.cfg.target_color,
                    self.cfg.correct_color,
                    self.cfg.wrong_color,
                )
            )

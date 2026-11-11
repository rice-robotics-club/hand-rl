"""Reward terms for playing a score on a piano, for any environment with a piano.

Each term is a pure function `term(state, action) -> (envs,) tensor`, typed against
the smallest state that has the fields it reads: KeyTargets or FingerTargets. An
environment can use these terms if its state type inherits from those classes.
Fixed settings such as distance scales are keyword arguments; bind them with
`src.reward.bind`.
"""

from typing import NamedTuple

import torch
from jaxtyping import Float, Int
from tensordict import TensorDictBase
from torch import Tensor

from src.music.keyboard import NUM_KEYS
from src.reward.shaping import bump, closeness
from src.sim.piano import PianoState

type RGBA = tuple[float, float, float, float]


class KeyTargets(PianoState):
    """The piano's keys, what the score wants from them, and how the two compare.

    Score fields describe the score at one time; see Score.targets.
    """

    target_active: Float[Tensor, f"*batch {NUM_KEYS}"]
    """1 for keys the score wants held now."""
    target_remaining: Float[Tensor, f"*batch {NUM_KEYS}"]
    """Seconds left in each active note; 0 for inactive keys."""
    next_note_valid: Float[Tensor, f"*batch {NUM_KEYS}"]
    """1 for keys with a note still to come."""
    next_note_time: Float[Tensor, f"*batch {NUM_KEYS}"]
    """Seconds until each key's next note; 0 if none."""
    next_note_duration: Float[Tensor, f"*batch {NUM_KEYS}"]
    """Duration of each key's next note; 0 if none."""
    target_hit_velocity: Float[Tensor, f"*batch {NUM_KEYS}"]
    """Wanted key speed at the hit for active keys; 0 for inactive keys."""
    key_down: Float[Tensor, f"*batch {NUM_KEYS}"]
    """1 for keys pressed far enough to count; see keys_down."""
    note_correctness: Float[Tensor, "*batch 1"]
    """Wanted keys down, divided by keys wanted or down (1 for a silent rest)."""


class FingerTargets(KeyTargets):
    """KeyTargets plus which finger should play each wanted key, and how far it is.

    See src/music/fingering.py. Fingers are 0 (thumb) to 4 (pinky), NO_FINGER if none.
    """

    closest_fingers: Int[Tensor, f"*batch {NUM_KEYS}"]
    """Finger closest to each active key, each key getting a different finger."""
    closest_finger_distances: Float[Tensor, f"*batch {NUM_KEYS}"]
    """Meters from each key to its closest finger."""
    home_fingers: Int[Tensor, f"*batch {NUM_KEYS}"]
    """Finger whose home-position key each active key is."""
    home_finger_distances: Float[Tensor, f"*batch {NUM_KEYS}"]
    """Meters from each key to its home finger, or the nearest finger if none."""


# ---------------------------------------------------------------------------
# Measurements, for building the state and for logging
# ---------------------------------------------------------------------------


class NoteScore(NamedTuple):
    """How the keys that are down compare with the keys the score wants, per env."""

    correct: Tensor
    """Fraction of wanted keys that are down."""
    missed: Tensor
    """Fraction of wanted keys that are up."""
    extra: Tensor
    """Number of unwanted keys that are down."""
    correctness: Tensor
    """Wanted keys down / keys wanted or down; 1 when no key is wanted or down."""


def score_notes(key_down: Tensor, target: Tensor) -> NoteScore:
    """Compares down keys with wanted keys, (envs, keys) each.

    Correct and missed are fractions of the chord, so a chord counts as much as a
    single note. Rests have nothing to get right or miss, but extra keys still count.
    """
    down, wanted = key_down.bool(), target.bool()
    hits = (down & wanted).sum(-1)
    misses = (~down & wanted).sum(-1)
    extras = (down & ~wanted).sum(-1)
    chord_size = wanted.sum(-1).clamp_min(1)
    anything = (down | wanted).any(-1)
    return NoteScore(
        correct=hits / chord_size,
        missed=misses / chord_size,
        extra=extras.float(),
        correctness=torch.where(
            anything, hits / (hits + misses + extras).clamp_min(1), 1.0
        ),
    )


def keys_down(pressed: Tensor, threshold: float) -> Tensor:
    """1.0 where a key has traveled at least `threshold` of the way down."""
    return (pressed >= threshold).float()


def key_metrics(
    active: Tensor, goal: Tensor, threshold: float = 0.5
) -> tuple[Tensor, Tensor, Tensor]:
    """Precision, recall, and F1 of the keys down against the wanted keys, per env.

    With nothing wanted, recall is 1, and precision is 1 only if nothing is down.
    """
    active_pressed = active >= threshold
    goal_pressed = goal >= threshold

    tp = (active_pressed & goal_pressed).sum(dim=-1).float()
    fp = (active_pressed & ~goal_pressed).sum(dim=-1).float()
    fn = (~active_pressed & goal_pressed).sum(dim=-1).float()

    precision = torch.where(
        tp + fp > 0, tp / (tp + fp).clamp(min=1), torch.where(tp + fn == 0, 1.0, 0.0)
    )
    recall = torch.where(tp + fn > 0, tp / (tp + fn).clamp(min=1), 1.0)
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall).clamp(min=1e-8),
        0.0,
    )
    return precision, recall, f1


def nearness(distances: Tensor, target: Tensor, scale: float) -> Tensor:
    """Mean `closeness` of the active keys to their fingers; 0 during rests."""
    if distances.shape != target.shape:
        raise ValueError("Fingertip/key distances must match the target key shape")
    if not torch.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("Fingertip/key distances must be finite and nonnegative")
    return (closeness(distances, scale) * target).sum(-1) / target.sum(-1).clamp_min(1)


def key_colors(
    key_down: Tensor,
    target: Tensor,
    target_color: RGBA,
    correct_color: RGBA,
    wrong_color: RGBA,
) -> Tensor:
    """An RGBA color per key showing what the note terms score: (..., keys, 4).

    Wanted keys are target_color while up and correct_color while down; unwanted keys
    that are down are wrong_color. Other keys are transparent.
    """
    down, wanted = key_down.bool().unsqueeze(-1), target.bool().unsqueeze(-1)

    def color(rgba: RGBA) -> Tensor:
        return torch.tensor(rgba, device=key_down.device)

    transparent = torch.zeros(4, device=key_down.device)
    return torch.where(
        wanted,
        torch.where(down, color(correct_color), color(target_color)),
        torch.where(down, color(wrong_color), transparent),
    )


# ---------------------------------------------------------------------------
# Reward terms
# ---------------------------------------------------------------------------


def correct_notes(state: KeyTargets, _: TensorDictBase) -> Tensor:
    """Fraction of wanted keys that are down."""
    return score_notes(state.key_down, state.target_active).correct


def missed_notes(state: KeyTargets, _: TensorDictBase) -> Tensor:
    """Minus the fraction of wanted keys that are up."""
    return -score_notes(state.key_down, state.target_active).missed


def extra_notes(state: KeyTargets, _: TensorDictBase) -> Tensor:
    """Minus the number of unwanted keys down, including early presses and late releases."""
    return -score_notes(state.key_down, state.target_active).extra


def wrong_keys(state: KeyTargets, _: TensorDictBase) -> Tensor:
    """-1 while any unwanted key is down, however many.

    A flat penalty rather than one per key: one of the papers we read found this
    keeps the policy bold enough to explore playing.
    """
    return -(score_notes(state.key_down, state.target_active).extra > 0).float()


def key_intensity(state: KeyTargets, _: TensorDictBase, *, scale: float) -> Tensor:
    """How close each wanted key's hit speed is to the target speed, averaged.

    hit_velocity is the speed on the step the key last crossed the press threshold;
    wanted keys not hit yet read zero and score low.
    """
    wanted = (state.target_hit_velocity > 0).float()
    error = state.hit_velocity.abs() - state.target_hit_velocity
    return (bump(error, scale) * wanted).sum(-1) / wanted.sum(-1).clamp_min(1)


def note_nearness(state: FingerTargets, _: TensorDictBase, *, scale: float) -> Tensor:
    """How close each wanted key is to the closest free fingertip."""
    return nearness(state.closest_finger_distances, state.target_active, scale)


def finger_nearness(state: FingerTargets, _: TensorDictBase, *, scale: float) -> Tensor:
    """How close each wanted key is to its home finger's fingertip."""
    return nearness(state.home_finger_distances, state.target_active, scale)

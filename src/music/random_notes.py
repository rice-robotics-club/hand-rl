"""Random notes to play, drawn live as an episode runs.

Why this exists. A fixed score is good for evaluating a policy. It is bad for
training, because the policy can memorize that one score. Here nothing is decided
ahead of time except the next set of notes. The policy has to learn to read the
notes and play them, not to recall a song.

The model. Notes come in sets that start together, like a chord. Each env holds a
window of two sets: the current one and the next one. When the next set's start
time passes, it becomes the current one and a new next set is drawn. The window
is shaped like a Score, so the same Score.targets code serves both modes. Holding
one set ahead also keeps next_note_valid / next_note_time / next_note_duration
filled in training, like they are when evaluating on a file.

Randomness comes from torch's global RNG. Call torch.manual_seed to repeat a run.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
from jaxtyping import Float, Int
from torch import Tensor

from src.music.keyboard import NUM_KEYS
from src.music.score import Score

NOTE_SETS_IN_WINDOW = 2
"""The current set plus the next one."""

CURRENT = 0
"""Index of the current set in the window."""

NEXT = 1
"""Index of the next set in the window."""

MIN_KEY_WEIGHT = 1e-12
"""A floor on how likely any key is to be picked. See `draw_note_set`."""


@dataclass(frozen=True)
class RandomNoteSettings:
    """How random notes are drawn. See HandPianoEnvConfig for what each means."""

    min_notes: int
    max_notes: int
    closeness: float
    min_hold: float
    max_hold: float
    min_rest: float
    max_rest: float


class NoteSet(NamedTuple):
    """Notes that start together, as (sets, max_notes) tensors.

    A set with fewer than max_notes notes is padded: start = inf and duration = 0.
    That pad never sounds and never counts as an upcoming note in Score.targets.
    Any other padding would leak into next_note_duration.
    """

    start: Float[Tensor, "sets notes"]
    duration: Float[Tensor, "sets notes"]
    key: Int[Tensor, "sets notes"]


def uniform(low: float, high: float, shape: tuple[int, ...], device: torch.device):
    """Floats drawn evenly from [low, high)."""
    return low + (high - low) * torch.rand(shape, device=device)


def draw_note_set(
    onset: Float[Tensor, " sets"], settings: RandomNoteSettings
) -> NoteSet:
    """One random set of notes for each onset, all starting at that onset.

    How many notes: uniform in [min_notes, max_notes].
    Which keys: all notes in a set cluster around one center key. The center is
    uniform over the whole keyboard, so training sees every region. Each key is
    weighted by a gaussian around the center with std `closeness` keys, and the
    notes are picked without replacement. So the keys in a set are always different,
    and a small closeness packs them together.
    How long: each note's hold time is drawn on its own, so notes in one set
    end at different times.
    """
    sets = onset.shape[0]
    device = onset.device
    count = torch.randint(
        settings.min_notes, settings.max_notes + 1, (sets, 1), device=device
    )
    center = torch.randint(0, NUM_KEYS, (sets, 1), device=device)
    offset = torch.arange(NUM_KEYS, device=device) - center  # (sets, keys)
    weights = torch.exp(-0.5 * (offset / settings.closeness) ** 2)
    # Hack. In float32 exp() hits exactly 0 past about 13 std. With a small closeness
    # that leaves fewer nonzero keys than max_notes, and multinomial throws. The
    # floor keeps every key possible. It is too small to change a normal draw.
    weights = weights + MIN_KEY_WEIGHT
    key = torch.multinomial(weights, settings.max_notes, replacement=False)

    # Slot i is a real note if i < count. The rest are padding.
    real = torch.arange(settings.max_notes, device=device) < count
    hold = uniform(settings.min_hold, settings.max_hold, key.shape, device)
    return NoteSet(
        start=torch.where(real, onset.unsqueeze(-1), math.inf),
        duration=torch.where(real, hold, 0.0),
        key=key,
    )


def draw_next_note_set(previous: NoteSet, settings: RandomNoteSettings) -> NoteSet:
    """A set that starts after every note of `previous` has ended, plus a rest.

    Waiting for the longest note means the previous set is always over before the
    next one starts. So at most one set sounds at a time, and the window of two sets
    is enough to describe everything that is happening or about to happen.
    """
    real = previous.duration > 0  # Holds are validated > 0, padding is exactly 0.
    end = torch.where(real, previous.start + previous.duration, -math.inf)
    rest = uniform(
        settings.min_rest, settings.max_rest, (end.shape[0],), previous.start.device
    )
    return draw_note_set(end.amax(-1) + rest, settings)


class NoteStream:
    """A rolling window of random notes for every env.

    This holds real state: the window changes as time passes. The env calls
    `advance` as its clock moves, and `reset` when an episode restarts.
    """

    def __init__(
        self, num_envs: int, settings: RandomNoteSettings, device: torch.device
    ):
        self._settings = settings
        shape = (num_envs, NOTE_SETS_IN_WINDOW, settings.max_notes)
        # Empty window: all padding. `reset` below fills it.
        self._start = torch.full(shape, math.inf, device=device)
        self._duration = torch.zeros(shape, device=device)
        self._key = torch.zeros(shape, dtype=torch.long, device=device)
        self.reset(torch.arange(num_envs, device=device))

    def reset(self, envs: Int[Tensor, " envs"]) -> None:
        """Draws a fresh window for `envs`. The first set starts at score time 0.

        Redrawing both sets (not just the next one) means a new episode never
        replays notes from the old one.
        """
        current = draw_note_set(
            torch.zeros(len(envs), device=self._start.device), self._settings
        )
        self._write(envs, CURRENT, current)
        self._write(envs, NEXT, draw_next_note_set(current, self._settings))

    def advance(self, now: Float[Tensor, "envs 1"]) -> None:
        """Moves each env's window forward to score time `now`.

        An env is due when its next set has started. Its next set becomes the
        current one and a new next set is drawn. This repeats until no env is due.
        One pass is not enough when an env's clock jumps a long way. That happens
        when training starts at a random episode length. Every pass moves the next
        set's start later by at least min_hold, so the loop ends.

        Cost: one GPU sync per pass to see if anything is due. It is cheap next to
        a physics step.
        """
        while True:
            due = (self._start[:, NEXT, 0] <= now[:, 0]).nonzero().flatten()
            if len(due) == 0:
                return
            # Slot 0 of every set is a real note, so slot 0's start is the set's start.
            promoted = NoteSet(
                self._start[due, NEXT], self._duration[due, NEXT], self._key[due, NEXT]
            )
            self._write(due, CURRENT, promoted)
            self._write(due, NEXT, draw_next_note_set(promoted, self._settings))

    def score(self) -> Score:
        """The window as a Score with (envs, notes) fields, for Score.targets."""
        return Score(
            start=self._start.flatten(1),
            duration=self._duration.flatten(1),
            key=self._key.flatten(1),
        )

    def _write(self, envs: Tensor, which: int, notes: NoteSet) -> None:
        self._start[envs, which] = notes.start
        self._duration[envs, which] = notes.duration
        self._key[envs, which] = notes.key

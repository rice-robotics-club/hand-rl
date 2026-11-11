"""A score to play, and what each key should be doing at a given time."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor

from src.music.notes import check_notes


@dataclass
class Score:
    """A score's notes as (notes,) tensors.

    Keys are indices into the keyboard's pitch list, not MIDI pitches.
    """

    start: Float[Tensor, " notes"]
    """Seconds (float64, so onsets are not rounded)."""
    duration: Float[Tensor, " notes"]
    """Seconds (float64)."""
    key: Int[Tensor, " notes"]
    """Key index into the keyboard's pitch list."""

    @property
    def end(self) -> Float[Tensor, " notes"]:
        return self.start + self.duration

    def length(self) -> float:
        """Seconds until the last note ends; 0 for an empty score."""
        return self.end.max().item() if len(self.end) else 0.0

    def targets(
        self, time: Float[Tensor, "envs 1"], num_keys: int
    ) -> dict[str, Float[Tensor, "envs keys"]]:
        """What each key should be doing at each environment's `time` (envs, 1).

        Returns float32 (envs, keys) tensors, zero where a value doesn't apply:
            target_active: 1 while a note on the key is sounding.
            target_remaining: Seconds left in the sounding note.
            next_note_valid: 1 if the key has a note still to come.
            next_note_time: Seconds until that next note starts.
            next_note_duration: How long that next note lasts.
        """
        time = time.to(torch.float64)
        num_envs = time.shape[0]
        key = self.key.expand(num_envs, -1)  # (envs, notes)

        def per_key(values: Tensor, reduce: str, empty: float) -> Tensor:
            """Combines the notes on each key: (envs, notes) -> (envs, keys)."""
            out = torch.full(
                (num_envs, num_keys), empty, dtype=values.dtype, device=values.device
            )
            return out.scatter_reduce(-1, key, values, reduce)

        sounding = (self.start <= time) & (time < self.end)
        active = per_key(sounding.double(), "amax", 0)
        remaining = per_key(torch.where(sounding, self.end - time, 0), "amax", 0)

        upcoming = self.start > time
        next_start = per_key(
            torch.where(upcoming, self.start, math.inf), "amin", math.inf
        )
        has_next = next_start.isfinite()
        until_next = torch.where(has_next, next_start - time, 0)
        # Notes on a key that start together count as one; use the longest.
        is_next = upcoming & (self.start == next_start.gather(-1, key))
        next_duration = per_key(torch.where(is_next, self.duration, 0), "amax", 0)

        return {
            "target_active": active.float(),
            "target_remaining": remaining.float(),
            "next_note_valid": has_next.float(),
            "next_note_time": until_next.float(),
            "next_note_duration": next_duration.float(),
        }

    @classmethod
    def from_notes(
        cls,
        notes: Sequence[Sequence[object]],
        pitches: list[int],
        device: torch.device | str,
    ) -> "Score":
        """A score from notes. `pitches` lists the keyboard's MIDI pitches in key order."""
        checked = check_notes(notes, pitches)
        key_of = {pitch: key for key, pitch in enumerate(pitches)}
        return cls(
            start=torch.tensor(
                [start for start, _, _ in checked], dtype=torch.float64, device=device
            ),
            duration=torch.tensor(
                [duration for _, duration, _ in checked],
                dtype=torch.float64,
                device=device,
            ),
            key=torch.tensor(
                [key_of[pitch] for _, _, pitch in checked],
                dtype=torch.long,
                device=device,
            ),
        )

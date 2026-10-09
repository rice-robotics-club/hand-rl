"""Sample a note score at simulation time without rounding its timestamps."""

import math

import torch


class PianoTargets:
    """Per-key targets from (start seconds, duration seconds, MIDI pitch) rows.

    Active intervals are [start, end). Overlapping notes of the same pitch
    use the latest release. Upcoming targets describe the next onset per key,
    including repeated notes while that key is already active.
    """

    def __init__(self, notes, pitches, device):
        rows = []
        for row in notes:
            if len(row) != 3:
                raise ValueError(
                    "Each note must contain start, duration, and MIDI pitch"
                )
            start, duration, pitch = row
            if not all(
                isinstance(x, (int, float))
                and not isinstance(x, bool)
                and math.isfinite(x)
                for x in row
            ):
                raise ValueError("Note values must be finite numbers")
            if (
                start < 0
                or duration <= 0
                or pitch != int(pitch)
                or pitch not in pitches
            ):
                raise ValueError(
                    "Notes require start >= 0, duration > 0, and a configured MIDI pitch"
                )
            rows.append((start, duration, pitch))
        self._notes = torch.tensor(rows, dtype=torch.float64, device=device).reshape(
            -1, 3
        )
        self._pitches = torch.tensor(pitches, device=device)

    def sample(self, time):
        """Return float32 [environments, keys] features; absent values are zero."""
        shape = (time.shape[0], self._pitches.numel())
        active = torch.zeros(shape, device=time.device)
        remaining = torch.zeros_like(active)
        upcoming = torch.zeros_like(active)
        until = torch.zeros_like(active)
        duration = torch.zeros_like(active)
        # Iterate over the small keyboard, keeping environments and notes batched.
        for key, pitch in enumerate(self._pitches):
            notes = self._notes[self._notes[:, 2] == pitch]
            if not len(notes):
                continue
            start, length = notes[:, 0], notes[:, 1]
            end = start + length
            sounding = (time >= start) & (time < end)
            active[:, key] = sounding.any(dim=-1)
            remaining[:, key] = torch.where(sounding, end - time, 0).amax(dim=-1)
            future = start > time
            next_start = torch.where(future, start, torch.inf).amin(
                dim=-1, keepdim=True
            )
            exists = torch.isfinite(next_start)
            upcoming[:, key] = exists[:, 0]
            until[:, key] = torch.where(exists, next_start - time, 0)[:, 0]
            # Simultaneous same-pitch notes use the longest duration.
            duration[:, key] = torch.where(
                future & (start == next_start), length, 0
            ).amax(dim=-1)
        return {
            "target_active": active,
            "target_remaining": remaining,
            "next_note_valid": upcoming,
            "next_note_time": until,
            "next_note_duration": duration,
        }

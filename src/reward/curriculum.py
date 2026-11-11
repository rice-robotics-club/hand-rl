"""Moves reward weights from a starting set to a final set as the policy plays better."""

import torch
from torch import Tensor


class Curriculum:
    """Tracks note recall across episodes and blends reward weights by it.

    Note recall is the fraction of wanted key-steps an episode actually played. Its
    running average sets progress: 0 at `recall_start`, 1 at `recall_end`. Each weight
    moves linearly from `start_weights` to `final_weights` with progress, and progress
    never goes down, so a dip in recall does not loosen penalties again.
    """

    def __init__(
        self,
        start_weights: dict[str, float],
        final_weights: dict[str, float] | None,
        recall_start: float,
        recall_end: float,
        rate: float,
    ):
        self.start_weights = start_weights
        self.final_weights = final_weights or start_weights
        self.recall_start = recall_start
        self.recall_end = recall_end
        self.rate = rate
        self.recall = 0.0
        """Running average of note recall over finished episodes."""
        self.progress = 0.0
        """0 to 1: how far the weights have moved toward final_weights."""

    def record(self, recalls: Tensor, num_envs: int):
        """Folds in the note recall of each finished episode.

        A full batch of `num_envs` episodes moves the average `rate` of the way toward
        their mean; fewer episodes move it proportionally less.
        """
        if len(recalls) == 0:
            return
        share = len(recalls) / num_envs
        self.recall += self.rate * share * (recalls.mean().item() - self.recall)
        span = self.recall_end - self.recall_start
        progress = min(max((self.recall - self.recall_start) / span, 0.0), 1.0)
        self.progress = max(self.progress, progress)

    def state_dict(self) -> dict[str, float]:
        """What a checkpoint needs to continue the curriculum where it left off."""
        return {"recall": self.recall, "progress": self.progress}

    def load_state_dict(self, state: dict[str, float]):
        """Continues from a state_dict, e.g. when resuming training."""
        self.recall = state["recall"]
        self.progress = state["progress"]

    def weights(self) -> dict[str, float]:
        """The reward weights at the current progress."""
        p = self.progress
        return {
            name: (1 - p) * start + p * self.final_weights[name]
            for name, start in self.start_weights.items()
        }


class RecallTracker:
    """Counts wanted and played key-steps per environment over an episode."""

    def __init__(self, num_envs: int, device: torch.device | str):
        self.played = torch.zeros(num_envs, device=device)
        self.wanted = torch.zeros(num_envs, device=device)

    def add(self, key_down: Tensor, target: Tensor):
        wanted = target.bool()
        self.played += (wanted & key_down.bool()).sum(-1)
        self.wanted += wanted.sum(-1)

    def finish(self, envs: Tensor) -> Tensor:
        """Note recall of the episodes in `envs` that wanted any notes; then clears them."""
        had_notes = self.wanted[envs] > 0
        recall = self.played[envs][had_notes] / self.wanted[envs][had_notes]
        self.played[envs] = 0
        self.wanted[envs] = 0
        return recall

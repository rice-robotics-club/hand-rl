"""Reward module for computing total rewards from a list of reward terms."""

from collections.abc import Callable

import torch
from tensordict import TensorClass, TensorDict, TypedTensorDict

from src.sim.vec_system import VecAction

_State = TensorClass | TypedTensorDict | TensorDict
_Action = TensorClass | TypedTensorDict | TensorDict | VecAction

type RewardTerm[S: _State, A: _Action] = Callable[[S, A], torch.Tensor]


class RewardFunction[S: _State, A: _Action]:
    """Computes the total reward for a given state using a list of reward terms."""

    def __init__(
        self,
        terms: list[RewardTerm[S, A]],
        weights: list[float] | dict[str, float] | None = None,
        batch_size: int = 1,
    ):
        """Initializes the reward function with a list of reward terms and optional weights.

        Args:
            terms: List of reward terms.
            weights: Optional tensor or dictionary of weights for each term.
            batch_size: Optional batch size for the reward terms.
        """
        self._terms = terms
        if weights is None:
            self._weights = torch.ones(len(terms))
        elif isinstance(weights, dict):
            self._weights = torch.tensor(
                [weights.get(term.__name__, 1.0) for term in terms]
            )
        else:
            self._weights = torch.tensor(weights)

        self._buf = torch.zeros((batch_size, len(terms)))

    def log(self) -> dict[str, float]:
        """Returns the mean reward for each term.

        Returns:
            A dictionary mapping term names to their mean reward values.
        """
        return {
            f"reward/{term.__name__}": self._buf[:, i].mean().item()
            for i, term in enumerate(self._terms)
        }

    def __call__(self, state: S, action: A) -> torch.Tensor:
        """Computes the total reward for the given state.

        Returns:
            The total reward as a per-env tensor.
        """
        self._buf[:] = (
            torch.stack([term(state, action) for term in self._terms], dim=-1)
            * self._weights
        )
        return self._buf.sum(dim=1)

"""Reward module for computing total rewards from a list of reward terms."""

import functools
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
        device: torch.device | str = "cpu",
    ):
        """Initializes the reward function with a list of reward terms and optional weights.

        Args:
            terms: List of reward terms.
            weights: Optional tensor or dictionary of weights for each term.
            batch_size: Optional batch size for the reward terms.
            device: Device on which to store the weights and reward buffer.
        """
        self._terms = terms
        if weights is None:
            self._weights = torch.ones(len(terms), device=device)
        elif isinstance(weights, dict):
            self._weights = torch.tensor(
                [weights.get(term.__name__, 1.0) for term in terms], device=device
            )
        else:
            self._weights = torch.tensor(weights, device=device)

        self._buf = torch.zeros((batch_size, len(terms)), device=device)

    def set_weights(self, weights: dict[str, float]) -> None:
        """Replaces the weights of the named terms, e.g. for a curriculum."""
        for i, term in enumerate(self._terms):
            if term.__name__ in weights:
                self._weights[i] = weights[term.__name__]

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


def bind[S: _State, A: _Action](
    term: Callable[..., torch.Tensor], **settings: object
) -> RewardTerm[S, A]:
    """Fixes a term's keyword settings, keeping its name for weights and logging.

    For example, `bind(note_nearness, scale=0.1)` is a term `(state, action)`. The
    type checker can't check the settings against the term, so a misspelled setting
    only fails when the term is first called.
    """
    return functools.update_wrapper(functools.partial(term, **settings), term)

"""Rewards shared between environments.

- `reward`: RewardFunction, which weights and sums reward terms, and `bind`.
- `piano`: terms for playing a score on a piano.
- `motion`: terms for smooth motion and holding a default pose.
- `shaping`: functions that turn errors and distances into bounded rewards.
- `curriculum`: moving reward weights as the policy improves.
"""

from .reward import RewardFunction, RewardTerm, bind

__all__ = ["RewardFunction", "RewardTerm", "bind"]

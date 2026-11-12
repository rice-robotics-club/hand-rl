"""Restore an RSL-RL v5 MLP actor and sample deployment inputs.

Transport, sensor subscriptions, and motor publishing are intentionally not
implemented here yet. Loading a policy never initializes a robot or simulator.
"""

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from threading import Event

import torch
import yaml
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from .input_loop import InputFrame, InputLoop, InputSource, training_control_period


class PolicyNode:
    """Load a policy using its training config and named observation layout.

    Checkpoint weights cannot establish feature meanings or action semantics.
    The supplied config and layout must belong to the checkpoint's training run.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        training_config: str | Path,
        layout: str | Path,
        device: str = "cpu",
    ):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Policy checkpoint not found: {checkpoint}")
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Supported inference devices are cpu and cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable; use --device cpu")

        config = yaml.safe_load(Path(training_config).read_text(encoding="utf-8-sig"))
        self.training_config = deepcopy(config)
        spec = json.loads(Path(layout).read_text(encoding="utf-8-sig"))
        runner = config["runner"]
        actor_config = deepcopy(runner["actor"])
        if actor_config.pop("class_name") != "MLPModel":
            raise ValueError(
                "This loader currently supports RSL-RL v5 MLPModel actors only"
            )
        groups = runner["obs_groups"]["actor"]
        if (
            not isinstance(groups, list)
            or not groups
            or len(set(groups)) != len(groups)
        ):
            raise ValueError(
                "Actor observation groups must be a nonempty list of unique names"
            )
        sizes = spec["observation_sizes"]
        self.num_actions = spec["num_actions"]
        if type(self.num_actions) is not int or self.num_actions <= 0:
            raise ValueError("num_actions must be a positive integer")
        if set(sizes) != set(groups):
            raise ValueError(
                "Layout observation names must exactly match the training actor groups"
            )
        if any(type(size) is not int or size <= 0 for size in sizes.values()):
            raise ValueError("Observation sizes must be positive integers")
        self.observation_sizes = {name: sizes[name] for name in groups}
        example = TensorDict(
            {
                name: torch.zeros(1, width)
                for name, width in self.observation_sizes.items()
            },
            batch_size=[1],
        )
        actor = MLPModel(
            example, {"actor": groups}, "actor", self.num_actions, **actor_config
        )
        # Restore on CPU first, including saved normalizer buffers. Never load
        # critic/optimizer state into the deployment actor or silently ignore keys.
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(saved, dict) or "actor_state_dict" not in saved:
            raise ValueError(
                "Expected an RSL-RL v5 checkpoint containing actor_state_dict"
            )
        try:
            actor.load_state_dict(saved["actor_state_dict"], strict=True)
        except RuntimeError as error:
            raise ValueError(
                "Checkpoint does not match the supplied actor config or observation/action layout"
            ) from error
        if any(
            not torch.isfinite(value).all() for value in actor.state_dict().values()
        ):
            raise ValueError(
                "Checkpoint contains non-finite actor weights or normalization statistics"
            )
        actor.eval()
        actor.requires_grad_(False)
        self.policy = actor.to(self.device)
        self.checkpoint = checkpoint.resolve()

    @property
    def control_rate_hz(self) -> float:
        """Policy input rate derived from the matching training configuration."""
        return 1 / training_control_period(self.training_config)

    def read_inputs(
        self,
        source: InputSource,
        on_frame: Callable[[InputFrame], None],
        *,
        max_frames: int | None = None,
        stop_event: Event | None = None,
    ) -> int:
        """Read sensors and goals at the training rate; transport is provided by source."""
        loop = InputLoop(training_control_period(self.training_config))
        return loop.run(source, on_frame, max_frames=max_frames, stop_event=stop_event)

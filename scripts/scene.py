"""Spins up the system described by a config file, without an environment or RL.

Useful for checking that a simulation builds and looks right before training on it.
"""

import argparse
from pathlib import Path

import torch
from tensordict import TensorDict

from src.util.run import build_system, get_config, setup_logging


class Args(argparse.Namespace):
    config: str = "config/piano.yaml"
    num_envs: int = 1
    steps: int | None = None
    random: bool = False
    headless: bool = False
    print_every: int | None = None
    save_state: str | None = None
    save_every: int = 1


def main():
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--config", type=str, default="config/piano.yaml")
    _ = parser.add_argument("--num-envs", type=int, default=1)
    _ = parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of steps to simulate. Defaults to running until interrupted.",
    )
    _ = parser.add_argument(
        "--random",
        action="store_true",
        help="Apply random actions instead of zero actions (the default pose).",
    )
    _ = parser.add_argument("--headless", action="store_true")
    _ = parser.add_argument(
        "--print-every",
        type=int,
        default=None,
        help="Print the state of environment 0 every N steps.",
    )
    _ = parser.add_argument(
        "--save-state",
        type=str,
        default=None,
        help="Record the state over time and save it to this .pt file on exit. Load it "
        + "with torch.load(path, weights_only=False), which gives {'step': [T], "
        + "'state': TensorDict[T, num_envs]}.",
    )
    _ = parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Record the state every N steps when using --save-state.",
    )
    args: Args = parser.parse_args(namespace=Args())

    setup_logging()

    cfg = get_config(args.config)
    cfg.system.num_envs = args.num_envs
    cfg.system.genesis.show_viewer = not args.headless

    sim = build_system(cfg)
    print(
        f"Built {type(sim).__name__}: {sim.num_envs} env(s), {sim.num_actions} actions"
    )
    for key, value in sim.state.items():
        print(f"  {key}: {tuple(value.shape)}")

    torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)
    device = sim.state.device
    step = 0
    steps: list[int] = []
    states: list[TensorDict] = []
    try:
        with torch.inference_mode():
            while args.steps is None or step < args.steps:
                shape = (sim.num_envs, sim.num_actions)
                action = (
                    torch.randn(shape, device=device)
                    if args.random
                    else torch.zeros(shape, device=device)
                )
                sim.step(action)
                step += 1

                if args.print_every and step % args.print_every == 0:
                    print(f"Step {step}:")
                    for key, value in sim.state.items():
                        print(f"  {key}: {value[0]}")
                if args.save_state and step % args.save_every == 0:
                    # Copy to the CPU, since the system updates its state in place.
                    state = TensorDict(dict(sim.state.items()), batch_size=sim.num_envs)
                    states.append(state.clone().cpu())
                    steps.append(step)
    except KeyboardInterrupt:
        pass
    print(f"Simulated {step} steps")

    if args.save_state and states:
        path = Path(args.save_state)
        path.parent.mkdir(parents=True, exist_ok=True)
        # TensorDict supports torch.stack, though torch's stubs only accept tensors.
        stacked = torch.stack(states)  # pyright: ignore[reportArgumentType]
        torch.save({"step": torch.tensor(steps), "state": stacked}, path)
        print(f"Saved {len(states)} states to {path}")


if __name__ == "__main__":
    main()

import argparse
import logging
import re
import warnings
from pathlib import Path

import torch
from rsl_rl.utils import resolve_class

from src.config import get_config

warnings.filterwarnings(
    "ignore", message=r".*torch\.jit\.script.*", category=FutureWarning
)


class Args(argparse.Namespace):
    config: str = "config/piano.yaml"
    checkpoint: str | None = None
    num_envs: int = 1
    episodes: int = 5
    headless: bool = False


def latest_checkpoint(log_dir: Path) -> Path:
    """Returns the checkpoint with the highest iteration number in `log_dir`."""
    checkpoints = list(log_dir.glob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {log_dir}")
    return max(checkpoints, key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))


def main():
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--config", type=str, default="config/piano.yaml")
    _ = parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a model checkpoint. Defaults to the latest one in ./logs/.",
    )
    _ = parser.add_argument("--num-envs", type=int, default=1)
    _ = parser.add_argument("--episodes", type=int, default=5)
    _ = parser.add_argument("--headless", action="store_true")
    args: Args = parser.parse_args(namespace=Args())

    logging.getLogger("genesis").propagate = False

    cfg = get_config(args.config)
    cfg.system.num_envs = args.num_envs
    cfg.system.genesis.show_viewer = not args.headless

    system_class, _ = resolve_class(cfg.system.model_dump())
    env_class, _ = resolve_class(cfg.env.model_dump())
    runner_class, _ = resolve_class(cfg.runner.model_dump())

    sim = system_class(cfg.system)
    env = env_class(sim, cfg.env)
    runner = runner_class(
        env,
        train_cfg=cfg.runner.model_dump(),
        log_dir=None,
        device="cuda",
    )

    checkpoint = Path(args.checkpoint or latest_checkpoint(Path("./logs/")))
    print(f"Loading checkpoint: {checkpoint}")
    _ = runner.load(str(checkpoint), load_cfg={"actor": True})
    policy = runner.get_inference_policy(device="cuda")

    obs = env.get_observations().to("cuda")
    episode_reward = torch.zeros(env.num_envs, device="cuda")
    completed = 0
    with torch.inference_mode():
        while completed < args.episodes:
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)
            obs = obs.to("cuda")
            episode_reward += rewards.to("cuda")

            for i in dones.nonzero().flatten().tolist():
                completed += 1
                terms = ", ".join(f"{k}: {v:.4f}" for k, v in extras["log"].items())
                print(
                    f"Episode {completed}: reward {episode_reward[i].item():.2f} "
                    + f"(final step, mean over envs: {terms})"
                )
                episode_reward[i] = 0.0


if __name__ == "__main__":
    main()

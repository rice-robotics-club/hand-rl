import argparse
from pathlib import Path

import torch

from src.util.run import (
    build_env,
    build_runner,
    build_system,
    get_config,
    latest_checkpoint,
    latest_run,
    setup_logging,
)


class Args(argparse.Namespace):
    config: str = "config/piano.yaml"
    log_dir: str | None = None
    checkpoint: str | None = None
    num_envs: int = 1
    episodes: int = 5
    headless: bool = False


def main():
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--config", type=str, default="config/piano.yaml")
    _ = parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a model checkpoint. Overrides --log-dir.",
    )
    _ = parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="A run directory, a run prefix (e.g. logs/piano), or a directory of runs. "
        + "Uses the latest checkpoint of the latest matching run. "
        + "Defaults to logs/<run_name>.",
    )
    _ = parser.add_argument("--num-envs", type=int, default=1)
    _ = parser.add_argument("--episodes", type=int, default=5)
    _ = parser.add_argument("--headless", action="store_true")
    args: Args = parser.parse_args(namespace=Args())

    setup_logging()

    cfg = get_config(args.config)

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    else:
        log_dir = Path(args.log_dir or f"logs/{cfg.runner.run_name}")
        checkpoint = latest_checkpoint(latest_run(log_dir))
    print(f"Loading checkpoint: {checkpoint}")

    cfg.system.num_envs = args.num_envs
    cfg.system.genesis.show_viewer = not args.headless

    sim = build_system(cfg)
    env = build_env(cfg, sim)
    runner = build_runner(cfg, env, log_dir=None)

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

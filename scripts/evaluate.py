import argparse
from pathlib import Path

import torch

from src.util.run import (
    build_env,
    build_runner,
    build_system,
    find_checkpoint,
    get_config,
    setup_logging,
)

# Import after src.util.run: it installs warning filters that rsl_rl needs first.
# isort: split
from src.env.hand_piano import HandPianoEnvConfig


class Args(argparse.Namespace):
    config: str = "config/piano.yaml"
    log_dir: str | None = None
    checkpoint: str | None = None
    num_envs: int = 1
    episodes: int = 5
    headless: bool = False
    audio: bool = False
    notes: str | None = None


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
    _ = parser.add_argument(
        "--audio",
        action="store_true",
        help="Play environment 0's key presses out loud, in real time.",
    )
    _ = parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Play this fixed score (a notes JSON file), e.g. "
        + "data/right_hand/expected.notes.json. Without it, the environment plays "
        + "random notes, the same as in training.",
    )
    args: Args = parser.parse_args(namespace=Args())

    setup_logging()

    cfg = get_config(args.config)
    if args.notes is not None:
        # Evaluating on a fixed score is the main reason this flag exists.
        if not isinstance(cfg.env, HandPianoEnvConfig):
            parser.error("--notes only works with the HandPiano environment")
        cfg.env.notes_path = args.notes

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    else:
        checkpoint = find_checkpoint(
            Path(args.log_dir or f"logs/{cfg.runner.run_name}")
        )
    print(f"Loading checkpoint: {checkpoint}")

    cfg.system.num_envs = args.num_envs
    cfg.system.genesis.show_viewer = not args.headless
    if args.audio:
        cfg.system.piano.audio = True

    sim = build_system(cfg)
    env = build_env(cfg, sim)
    runner = build_runner(cfg, env, log_dir=None)

    _ = runner.load(
        str(checkpoint), load_cfg={"actor": True}, map_location=str(env.device)
    )
    policy = runner.get_inference_policy(device=str(env.device))

    obs = env.get_observations().to(str(env.device))
    episode_reward = torch.zeros(env.num_envs, device=str(env.device))
    completed = 0
    with torch.inference_mode():
        while completed < args.episodes:
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)
            obs = obs.to(str(env.device))
            episode_reward += rewards.to(str(env.device))

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

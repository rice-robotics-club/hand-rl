import argparse
from pathlib import Path

from src.util.run import (
    build_env,
    build_runner,
    build_system,
    find_checkpoint,
    get_config,
    new_run_dir,
    setup_logging,
)


class Args(argparse.Namespace):
    config: str = "config/piano.yaml"
    resume: str | None = None


def main():
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--config", type=str, default="config/piano.yaml")
    _ = parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Continue training from a checkpoint: a .pt file, a run directory, a run "
        + "prefix (e.g. logs/piano), or a directory of runs, using the latest "
        + "checkpoint of the latest matching run. Logs go to a new run directory.",
    )
    args: Args = parser.parse_args(namespace=Args())

    setup_logging()

    cfg = get_config(args.config)
    sim = build_system(cfg)
    env = build_env(cfg, sim)
    runner = build_runner(cfg, env, log_dir=new_run_dir(cfg))
    if args.resume is not None:
        checkpoint = find_checkpoint(Path(args.resume))
        print(f"Resuming from checkpoint: {checkpoint}")
        # Loads the policy, critic, optimizer, iteration count, and environment state.
        _ = runner.load(str(checkpoint))
    runner.learn(num_learning_iterations=1000, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

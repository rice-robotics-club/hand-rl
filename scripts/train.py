import argparse

from src.util.run import (
    build_env,
    build_runner,
    build_system,
    get_config,
    new_run_dir,
    setup_logging,
)


class Args(argparse.Namespace):
    config: str = "config/piano.yaml"
    iterations: int = 1000
    num_envs: int | None = None
    viewer: bool = False


def main():
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--config", type=str, default="config/piano.yaml")
    _ = parser.add_argument("--iterations", type=int, default=1000)
    _ = parser.add_argument("--num-envs", type=int, default=None)
    _ = parser.add_argument("--viewer", action="store_true")
    args: Args = parser.parse_args(namespace=Args())

    setup_logging()

    cfg = get_config(args.config)
    if args.num_envs is not None:
        cfg.system.num_envs = args.num_envs
    if args.viewer:
        cfg.system.genesis.show_viewer = True
    sim = build_system(cfg)
    env = build_env(cfg, sim)
    runner = build_runner(cfg, env, log_dir=new_run_dir(cfg))
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

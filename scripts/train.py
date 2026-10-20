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


def main():
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--config", type=str, default="config/piano.yaml")
    args: Args = parser.parse_args(namespace=Args())

    setup_logging()

    cfg = get_config(args.config)
    sim = build_system(cfg)
    env = build_env(cfg, sim)
    runner = build_runner(cfg, env, log_dir=new_run_dir(cfg))
    runner.learn(num_learning_iterations=1000, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

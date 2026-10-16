import argparse
import logging
import warnings

from rsl_rl.utils import resolve_class

from src.config import get_config

warnings.filterwarnings(
    "ignore", message=r".*torch\.jit\.script.*", category=FutureWarning
)


class Args(argparse.Namespace):
    config: str = "config/piano.yaml"


def main():
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--config", type=str, default="config/piano.yaml")
    args: Args = parser.parse_args(namespace=Args())

    logging.getLogger("genesis").propagate = False

    cfg = get_config(args.config)
    system_class, _ = resolve_class(cfg.system.model_dump())
    env_class, _ = resolve_class(cfg.env.model_dump())
    runner_class, _ = resolve_class(cfg.runner.model_dump())

    sim = system_class(cfg.system)
    env = env_class(sim, cfg.env)
    runner = runner_class(
        env,
        train_cfg=cfg.runner.model_dump(),
        log_dir="./logs/",
        device="cuda",
    )
    runner.learn(num_learning_iterations=1000, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

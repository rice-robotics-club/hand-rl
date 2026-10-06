from rsl_rl.utils import resolve_class

from src.config import get_config


def main():
    cfg = get_config("config/ruka.yaml")
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
    runner.learn(num_learning_iterations=1000)


if __name__ == "__main__":
    main()

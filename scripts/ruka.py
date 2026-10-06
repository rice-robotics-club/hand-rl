import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from src.config import Config, get_config
from src.env.ruka_env import RukaEnv, RukaEnvConfig
from src.sim.ruka_sim import RukaSim, RukaSimConfig


def main():
    gs.init()
    cfg: Config[RukaEnvConfig, RukaSimConfig] = get_config("config/ruka.yaml")
    sim = RukaSim(cfg.system)
    env = RukaEnv(sim, cfg.env)
    runner = OnPolicyRunner(
        env, train_cfg=cfg.runner.model_dump(), log_dir="./logs/", device="cuda"
    )
    runner.learn(num_learning_iterations=1000)

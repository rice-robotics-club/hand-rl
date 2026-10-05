import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from src.env.ruka_env import RukaEnv
from src.sim.ruka_sim import RukaSim


def main():
    gs.init()

    obs_groups = {
        "actor": ["position", "orientation", "velocity"],
        "critic": ["position", "orientation", "velocity"],
    }

    train_cfg = {
        "num_steps_per_env": 24,
        "obs_groups": obs_groups,
        "save_interval": 100,
        "algorithm": {
            "class_name": "PPO",
        },
        "actor": {
            "class_name": "MLPModel",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
            },
        },
        "critic": {
            "class_name": "MLPModel",
        },
    }

    sim = RukaSim(num_envs=1)
    env = RukaEnv(sim)
    runner = OnPolicyRunner(env, train_cfg=train_cfg, log_dir="./logs/", device="cuda")
    runner.learn(num_learning_iterations=1000)

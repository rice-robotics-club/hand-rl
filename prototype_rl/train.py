"""PROTOTYPE ONLY: train RSL-RL on an abstract five-finger model."""

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

import torch
import yaml
from rsl_rl.runners import OnPolicyRunner

from .env import PrototypeFingerEnv

ROOT = Path(__file__).resolve().parent.parent
NOTICE = "PROTOTYPE ONLY: abstract bends, no CAD/contact physics, not a hardware controller."


def evaluate(env, policy=None):
    """Evaluate one full episode; held-key metrics are not acoustic accuracy."""
    obs = env.reset()
    reward_sum = tp = fp = fn = exact = frames = 0.0
    with torch.inference_mode():
        for _ in range(env.max_episode_length):
            actions = (torch.full((env.num_envs, 5), -1.0, device=env.device)
                       if policy is None else policy(obs))
            obs, reward, _, extras = env.step(actions)
            actual, target = extras["prototype_pressed"], extras["prototype_target"]
            tp += (actual & target).sum().item()
            fp += (actual & ~target).sum().item()
            fn += (~actual & target).sum().item()
            exact += (actual == target).all(dim=1).sum().item()
            frames += env.num_envs
            reward_sum += reward.sum().item()
    return {"mean_step_reward": reward_sum / frames,
            "held_key_f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "exact_key_state_fraction": exact / frames}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=Path, default=ROOT / "samples/right_hand/expected.notes.json")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.threads < 1:
        parser.error("iterations and threads must be positive")
    print(NOTICE, flush=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("prototype") is not True:
        parser.error("Configuration must explicitly declare prototype: true")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    env_cfg = config["environment"]
    if args.num_envs is not None:
        env_cfg["num_envs"] = args.num_envs
    notes = json.loads(args.notes.read_text(encoding="utf-8-sig"))
    env = PrototypeFingerEnv(notes, device=args.device, **env_cfg)
    output = args.output or ROOT / "prototype_runs" / datetime.now().strftime("prototype_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    baseline = evaluate(env)
    env.reset()
    (output / "PROTOTYPE_ONLY.txt").write_text(NOTICE + "\n", encoding="utf-8")
    (output / "target_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    (output / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    runner = OnPolicyRunner(env, deepcopy(config["runner"]), log_dir=str(output), device=args.device)
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
    result = evaluate(env, runner.get_inference_policy(device=args.device))
    summary = {"prototype": True, "notice": NOTICE, "seed": args.seed,
               "iterations": args.iterations, "evaluation": "same score used for training; not generalization",
               "raised_fingers_baseline": baseline, "trained_policy": result}
    runner.save(str(output / "PROTOTYPE_policy.pt"), infos=summary)
    (output / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Prototype results saved to {output}")


if __name__ == "__main__":
    main()

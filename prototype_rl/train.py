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
    # reset the environment and initialize metrics
    obs = env.reset()

    # initialize metrics for true positives, false positives, false negatives, exact matches, and total frames
    reward_sum = tp = fp = fn = exact = frames = 0.0
    with torch.inference_mode():
        for _ in range(env.max_episode_length):
            # 
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
            
    # return a dictionary containing the mean step reward, held key F1 score, and exact key state fraction
    return {"mean_step_reward": reward_sum / frames,
            "held_key_f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "exact_key_state_fraction": exact / frames}


def main():
    # all this argparse stuff is just things you can do when you run this script from the command line
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


    # not allowed to run with zero iterations or threads, because that wouldn't do anything
    if args.iterations < 1 or args.threads < 1:
        parser.error("iterations and threads must be positive")

    # what would appear to be a disclaimer
    print(NOTICE, flush=True)

    # load in the configuration file, which is a YAML file that specifies the environment and runner parameters
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("prototype") is not True:
        parser.error("Configuration must explicitly declare prototype: true")

    # ok now we can set up like threads and seeds and stuff, and then create the environment and runner
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    # our config file has a section for the environment, which we can override with command line arguments if desired
    env_cfg = config["environment"]
    if args.num_envs is not None:
        env_cfg["num_envs"] = args.num_envs

    # load in the notes from the specified JSON file, which will be used to define the target key presses for the prototype finger environment
    notes = json.loads(args.notes.read_text(encoding="utf-8-sig"))

    # boot the environment
    env = PrototypeFingerEnv(notes, device=args.device, **env_cfg)

    # create an output directory for the results, which will be named with a timestamp
    output = args.output or ROOT / "prototype_runs" / datetime.now().strftime("prototype_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)

    # evaluate the baseline performance of the environment without any trained policy
    baseline = evaluate(env)

    # reset the environment to start the fresh episode
    env.reset()

    # save the notice, target notes, and configuration to the output directory for reference
    (output / "PROTOTYPE_ONLY.txt").write_text(NOTICE + "\n", encoding="utf-8")
    (output / "target_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    (output / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    # create the runner, which will handle the training loop, and start learning
    runner = OnPolicyRunner(env, deepcopy(config["runner"]), log_dir=str(output), device=args.device)
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
    result = evaluate(env, runner.get_inference_policy(device=args.device))

    # save the results of the training and evaluation to the output directory, including a summary of the training parameters and evaluation metrics
    summary = {"prototype": True, "notice": NOTICE, "seed": args.seed,
               "iterations": args.iterations, "evaluation": "same score used for training; not generalization",
               "raised_fingers_baseline": baseline, "trained_policy": result}
    runner.save(str(output / "PROTOTYPE_policy.pt"), infos=summary)
    (output / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # print the summary of the training and evaluation results to the console for quick reference
    print(json.dumps(summary, indent=2))
    print(f"Prototype results saved to {output}")


if __name__ == "__main__":
    main()

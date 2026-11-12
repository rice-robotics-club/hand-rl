"""Load and validate a trained policy; no robot commands are sent."""

import argparse
from importlib import import_module

from src.deployment.policy_node import PolicyNode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True, help="Matching training YAML")
    parser.add_argument(
        "--layout", required=True, help="Matching observation/action layout JSON"
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument(
        "--input-source", help="Python module:factory returning a sensor/goal adapter"
    )
    parser.add_argument("--samples", type=int, help="Stop after this many input frames")
    args = parser.parse_args()
    if args.samples is not None and (args.samples <= 0 or not args.input_source):
        parser.error("--samples requires --input-source and a positive count")
    node = PolicyNode(args.checkpoint, args.config, args.layout, args.device)
    print(f"Loaded policy: {node.checkpoint}")
    print(f"Device: {node.device}; actions: {node.num_actions}")
    print(f"Actor observations (training order): {node.observation_sizes}")
    if not args.input_source:
        print("Checkpoint loading complete. No sensor/goal adapter was provided.")
        return
    module, separator, factory_name = args.input_source.partition(":")
    if not separator or not module or not factory_name:
        parser.error("--input-source must have the form module:factory")
    # Resolve timing before constructing a hardware adapter.
    print(f"Input sampling rate: {node.control_rate_hz:g} Hz")
    source = getattr(import_module(module), factory_name)()
    try:
        node.read_inputs(
            source,
            lambda frame: print(
                f"Input tick {frame.tick}, elapsed {frame.elapsed_seconds:.3f}s"
            ),
            max_frames=args.samples,
        )
    except KeyboardInterrupt:
        print("Input sampling stopped.")
    finally:
        close = getattr(source, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()

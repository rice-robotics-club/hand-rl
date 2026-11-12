# Robot policy node: checkpoint loading and input sampling

Step 1 is implemented in `src/deployment/policy_node.py`. It restores an RSL-RL
v5 MLP actor directly, without constructing Genesis, a training runner, or robot
connections. Middleware (including any ROS integration) is not selected yet.

Run from the repository root with the project's inference dependencies installed:

```powershell
python -m scripts.policy_node --checkpoint logs/YOUR_RUN/model_100.pt --config config/piano.yaml --layout config/policy_layout.example.json --device cpu
```

Use the **training configuration and feature layout belonging to the checkpoint**.
The example layout describes the current piano configuration, with 27 actions
(6 base + 21 joint), and four quaternion observation values. It is not evidence
that an arbitrary checkpoint or physical robot uses that layout. The earlier
five-action abstract prototype is not compatible with the piano layout.

The loader preserves observation concatenation order from the training YAML,
reconstructs the actor architecture, strictly loads actor weights and normalization
buffers, and sets evaluation mode with gradients disabled. In RSL-RL's MLPModel,
normal forward calls use deterministic output unless stochastic output is requested.
CPU is the default; CUDA can be explicitly selected. The node loads using PyTorch's
restricted weights-only loader; custom pickled checkpoint metadata may require
re-exporting a standard tensor/primitive checkpoint rather than disabling it.

Missing files, unsupported actor types, mismatched fields/dimensions, and
non-finite weights fail loading. Equal-sized layouts with different meanings
cannot be detected from weights alone; their training metadata must be verified.
Without an input adapter, the CLI reports successful loading and exits.

## Fixed-rate sensor and goal inputs

`PolicyNode.read_inputs(source, on_frame)` reads both inputs once per control
tick. The transport adapter implements `InputSource` from
`src/deployment/input_loop.py`:

```python
class RobotInputs:
    def read_sensors(self):
        # Return a mapping from real hardware readings or subscriber buffers.
        ...

    def read_goal(self, elapsed_seconds):
        # Return the current goal; elapsed_seconds starts at zero for this run.
        ...

# Once implemented:
# node.read_inputs(RobotInputs(), on_frame=consume, stop_event=stop)
```

The adapter must provide actual data, appropriate timestamps/freshness checks,
and bounded read timeouts. There is no default fake hardware source. Frames
contain independent copies of sensor and goal mappings, a scheduled tick index,
the monotonic acquisition-start timestamp, and elapsed time. Raw inputs are not
yet assembled into actor observations; inference and motor publishing are later
steps. Sensor and goal reads are sequential, not an atomic hardware snapshot.

Timing comes from the matching training config: `system.genesis.sim_options.dt`
for HandPiano, or the overriding `system.control_dt` for RukaSim. Physics
substeps do not change the policy input rate. The current piano config is **60 Hz**;
a config with `dt: 0.05` runs at **20 Hz**. Use that only with a policy trained at
20 Hz. The loop does not silently switch a 60 Hz checkpoint to the paper's rate.

The first sample is immediate. Absolute monotonic deadlines avoid adding read
and processing time to every period. Missed periods are skipped, not replayed in
a burst; goal time follows actual elapsed time. Tick-number gaps expose overruns.
This is best-effort Python scheduling, not a hard real-time guarantee. A stop
event interrupts the wait; it cannot cancel an already blocked hardware read.
Read errors propagate and stop the loop.

To use your implemented adapter factory from the CLI:

```powershell
python -m scripts.policy_node --checkpoint logs/YOUR_RUN/model_100.pt --config config/piano.yaml --layout config/policy_layout.example.json --input-source your_robot_adapter:create_inputs --samples 100
```

The factory takes no arguments and returns the input source. Omit `--samples`
to run until interrupted. The CLI prints tick/time information, and calls the
adapter's optional `close()` method on exit. Hardware/ROS adapters remain to be
provided once the robot's input transport is selected.

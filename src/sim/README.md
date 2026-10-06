# `sim` Module

This module defines the [`VecSystem`](./vec_system.py) interface, which provides a clean abstraction that could be used for both a vectorized simulator, such as gazebo, as well as an interface to actual robot hardware.

It uses generics (constraints upon Python's loose typing) to allow the user of the interface to define the structure of the state and action spaces for the given system. Such types may extend from those in the PyTorch ecosystem, such as bare Tensors, as well as TensorDicts.

## Rationale

The `VecEnv` class of the RSL_RL library utilizes the [tensordict](https://docs.pytorch.org/tensordict/stable/index.html) library to take in observations used by the policy, upon each call of the [`step` function](https://leggedrobotics.github.io/rsl_rl/api/env.html#rsl_rl.env.vec_env.VecEnv.step) that we implement as users of the library.

A tensordict is a simple way of providing named values to a neural network, such as a multi-layer perceptron.
Without it, one would need to manually map each individual input of a neural network to a position in the first layer of the network. A base `TensorDict` acts similar to a normal Python dictionary, with the added ability to specify a "batch size" that all contained tensors must conform to. The library also provides extensions to TensorDict, TensorClass and TypedTensorDict, which allows specifying named fields for the tensors contained within. This allows static type checking and autocompletion for the tensors contained within.

Thus, the `VecSystem` class allows the user to define the structure of the state and action spaces for the given system, using tensordicts to provide named values to the policy. Then, an implementation of `VecEnv` can request an implementation of a `VecSystem` to simulate the given system.

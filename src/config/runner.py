"""Training configuration for the hand RL project.

Note: Generated with Claude Sonnet 5.5
"""

from collections.abc import Callable
from typing import Annotated, Any, Literal

from pydantic import Field

from src.base_config import ConfigBase

Optimizer = Literal["adam", "adamw", "sgd", "rmsprop"]
"""Names accepted by `rsl_rl.utils.utils.resolve_optimizer`."""

Activation = Literal[
    "elu",
    "selu",
    "relu",
    "crelu",
    "lrelu",
    "tanh",
    "sigmoid",
    "softplus",
    "gelu",
    "swish",
    "mish",
    "identity",
]
"""Names accepted by `rsl_rl.utils.utils.resolve_nn_activation`."""


class LoggerConfig(ConfigBase):
    """Logging writer config for Weights & Biases and Neptune.

    Source: https://leggedrobotics.github.io/rsl_rl/guide/configuration.html#logger
    """

    class_name: Literal["WandbLogWriter", "NeptuneLogWriter"]
    """Logger class name."""

    project_name: str
    """Name of the project."""


class GaussianDistributionConfig(ConfigBase):
    """Gaussian distribution config.

    Source: https://leggedrobotics.github.io/rsl_rl/guide/configuration.html#gaussiandistribution
    """

    class_name: Literal["GaussianDistribution"] = "GaussianDistribution"
    """Distribution class name."""

    init_std: float = 1.0
    """Initial standard deviation for all dimensions."""

    std_range: tuple[float, float] = (1e-6, 1e6)
    """Minimum and maximum allowed values for the standard deviation for numerical stability."""

    std_type: Literal["scalar", "log"] = "scalar"
    """Whether the standard deviation is stored directly or in log-space."""

    learn_std: bool = True
    """Whether the standard deviation is learnable or fixed."""


class HeteroscedasticGaussianDistributionConfig(ConfigBase):
    """Heteroscedastic Gaussian distribution config.

    Source: https://leggedrobotics.github.io/rsl_rl/guide/configuration.html#heteroscedasticgaussiandistribution
    """

    class_name: Literal["HeteroscedasticGaussianDistribution"] = (
        "HeteroscedasticGaussianDistribution"
    )
    """Distribution class name."""

    init_std: float = 1.0
    """Initial standard deviation (used to initialize the std head bias)."""

    std_range: tuple[float, float] = (1e-6, 1e6)
    """Minimum and maximum allowed values for the standard deviation for numerical stability."""

    std_type: Literal["scalar", "log"] = "scalar"
    """Whether the standard deviation is stored directly or in log-space."""


class BetaDistributionConfig(ConfigBase):
    """Beta distribution config.

    Source: https://leggedrobotics.github.io/rsl_rl/guide/configuration.html#betadistribution
    """

    class_name: Literal["BetaDistribution"] = "BetaDistribution"
    """Distribution class name."""

    action_range: tuple[float, float] = (-1.0, 1.0)
    """Interval `(min, max)` to which samples are linearly rescaled. The Beta distribution naturally produces samples in `[0, 1]`, which are rescaled to this range."""


DistributionConfig = Annotated[
    GaussianDistributionConfig
    | HeteroscedasticGaussianDistributionConfig
    | BetaDistributionConfig,
    Field(discriminator="class_name"),
]
"""Any distribution config, selected by `class_name`."""


class ConstantWeightScheduleConfig(ConfigBase):
    """Constant RND weight schedule config.

    Source: https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/extensions/rnd.py
    """

    mode: Literal["constant"] = "constant"
    """Type of schedule to use for the RND weight parameter."""


class StepWeightScheduleConfig(ConfigBase):
    """Step RND weight schedule config.

    Source: https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/extensions/rnd.py
    """

    mode: Literal["step"] = "step"
    """Type of schedule to use for the RND weight parameter."""

    final_step: int
    """Step at which the weight parameter is set to the final value."""

    final_value: float
    """Final value of the weight parameter."""


class LinearWeightScheduleConfig(ConfigBase):
    """Linear RND weight schedule config.

    Source: https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/extensions/rnd.py
    """

    mode: Literal["linear"] = "linear"
    """Type of schedule to use for the RND weight parameter."""

    initial_step: int
    """Step at which the weight starts changing from its initial value."""

    final_step: int
    """Step at which the weight reaches the final value."""

    final_value: float
    """Final value of the weight parameter."""


WeightScheduleConfig = Annotated[
    ConstantWeightScheduleConfig
    | StepWeightScheduleConfig
    | LinearWeightScheduleConfig,
    Field(discriminator="mode"),
]
"""Any RND weight schedule config, selected by `mode`."""


class RNDConfig(ConfigBase):
    """Random Network Distillation extension config.

    Source: https://leggedrobotics.github.io/rsl_rl/guide/configuration.html#random-network-distillation
    """

    num_outputs: int
    """Number of outputs of the RND networks."""

    predictor_hidden_dims: tuple[int, ...] | list[int]
    """Hidden dimensions of the RND predictor network."""

    target_hidden_dims: tuple[int, ...] | list[int]
    """Hidden dimensions of the RND target network."""

    activation: Activation = "elu"
    """Activation function for the RND networks."""

    state_normalization: bool = False
    """Whether to normalize the RND state."""

    reward_normalization: bool = False
    """Whether to normalize the RND reward."""

    weight: float = 0.0
    """Initial weight of the RND reward."""

    weight_schedule: WeightScheduleConfig | None = None
    """Weight schedule for the RND reward."""

    learning_rate: float = 0.001
    """Learning rate for the RND optimizer."""


class SymmetryConfig(ConfigBase):
    """Symmetry augmentation extension config.

    Source: https://leggedrobotics.github.io/rsl_rl/guide/configuration.html#symmetry-augmentation
    """

    use_data_augmentation: bool
    """Whether to add symmetric trajectories to the batch."""

    data_augmentation_func: str | Callable[..., Any] | None
    """Function to generate symmetric trajectories. Resolved using `resolve_callable()`."""

    use_mirror_loss: bool
    """Whether to add a symmetry loss term to the loss function."""

    mirror_loss_coeff: float
    """Coefficient for the symmetry loss."""


class PPOConfig(ConfigBase):
    """Proximal Policy Optimization algorithm config.

    Source: https://leggedrobotics.github.io/rsl_rl/guide/configuration.html#ppo
    """

    class_name: Literal["PPO"] = "PPO"
    """Algorithm class name."""

    optimizer: Optimizer = "adam"
    """Optimizer used for policy/value updates."""

    learning_rate: float = 0.001
    """Optimizer learning rate."""

    num_learning_epochs: int = 5
    """Number of optimization epochs per iteration."""

    num_mini_batches: int = 4
    """Number of mini-batches per iteration."""

    schedule: Literal["adaptive", "fixed"] = "adaptive"
    """Learning rate schedule."""

    value_loss_coef: float = 1.0
    """Coefficient for the value-function loss."""

    clip_param: float = 0.2
    """PPO clipping parameter for surrogate/value clipping."""

    use_clipped_value_loss: bool = True
    """Whether to use clip the value loss."""

    desired_kl: float = 0.01
    """Target KL divergence used by the adaptive learning-rate schedule."""

    entropy_coef: float = 0.01
    """Entropy regularization coefficient."""

    gamma: float = 0.99
    """Discount factor."""

    lam: float = 0.95
    """GAE lambda parameter."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for gradient clipping."""

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize advantages for each mini-batch instead of across the entire rollout."""

    use_mixed_precision: bool = False
    """Whether to run the forward pass and loss computation in bfloat16 autocast. Backward, gradient clipping, and the optimizer step always stay in fp32."""

    share_cnn_encoders: bool = False
    """Whether to share the CNN networks between actor and critic in case the `CNNModel` is used."""

    rnd_cfg: RNDConfig | None = None
    """Optional RND extension configuration."""

    symmetry_cfg: SymmetryConfig | None = None
    """Optional symmetry extension configuration."""

    grad_reduce_bucket_mb: float = 25.0
    """Maximum size, in megabytes, of a single packed gradient buffer used when reducing gradients across GPUs during multi-GPU training. Matches `torch.nn.parallel.DistributedDataParallel`'s default `bucket_cap_mb`."""


class DistillationConfig(ConfigBase):
    """Student-teacher distillation algorithm config.

    Source: https://leggedrobotics.github.io/rsl_rl/guide/configuration.html#distillation
    """

    class_name: Literal["Distillation"] = "Distillation"
    """Algorithm class name."""

    optimizer: Optimizer = "adam"
    """Optimizer used for student updates."""

    learning_rate: float = 0.001
    """Optimizer learning rate."""

    num_learning_epochs: int = 1
    """Number of optimization epochs per iteration."""

    gradient_length: int = 15
    """Gradient backpropagation length."""

    max_grad_norm: float | None = None
    """Maximum gradient norm for gradient clipping."""

    loss_type: Literal["mse", "huber"] = "mse"
    """Loss type."""

    use_mixed_precision: bool = False
    """Whether to run the forward pass and loss computation in bfloat16 autocast. Backward, gradient clipping, and the optimizer step always stay in fp32."""

    grad_reduce_bucket_mb: float = 25.0
    """Maximum size, in megabytes, of a single packed gradient buffer used when reducing gradients across GPUs during multi-GPU training. Matches `torch.nn.parallel.DistributedDataParallel`'s default `bucket_cap_mb`."""


class _MLPModelFields(ConfigBase):
    """Keys shared by `MLPModel`, `RNNModel`, and `CNNModel`.

    Source: https://leggedrobotics.github.io/rsl_rl/guide/configuration.html#mlpmodel
    """

    distribution_cfg: DistributionConfig
    """Optional output distribution configuration. If provided, the model can output stochastic values sampled from the specified distribution."""

    hidden_dims: tuple[int, ...] | list[int] = Field(
        default_factory=lambda: [256, 256, 256]
    )
    """Hidden dimensions of the MLP."""

    activation: Activation = "elu"
    """Activation function of the MLP."""

    obs_normalization: bool = False
    """Whether to normalize the observations before passing them to the MLP."""


class MLPModelConfig(_MLPModelFields):
    """Multi-layer perceptron model config.

    Source: https://leggedrobotics.github.io/rsl_rl/guide/configuration.html#mlpmodel
    """

    class_name: Literal["MLPModel"] = "MLPModel"
    """Model class name."""


class RNNModelConfig(_MLPModelFields):
    """Recurrent neural network model config.

    Source: https://leggedrobotics.github.io/rsl_rl/guide/configuration.html#rnnmodel
    """

    class_name: Literal["RNNModel"] = "RNNModel"
    """Model class name."""

    rnn_type: Literal["lstm", "gru"] = "lstm"
    """Type of RNN network."""

    rnn_hidden_dim: int = 256
    """Hidden dimension of the RNN."""

    rnn_num_layers: int = 1
    """Number of RNN layers."""


class CNNEncoderConfig(ConfigBase):
    """Configuration of a single CNN encoder.

    Source: https://leggedrobotics.github.io/rsl_rl/guide/configuration.html#cnnmodel
    """

    output_channels: tuple[int, ...] | list[int]
    """Output channels for each convolutional layer."""

    kernel_size: int | tuple[int, ...] | list[int]
    """Kernel size for each convolutional layer or a single kernel size for all layers."""

    stride: int | tuple[int, ...] | list[int] = 1
    """Stride for each convolutional layer or a single stride for all layers."""

    dilation: int | tuple[int, ...] | list[int] = 1
    """Dilation for each convolutional layer or a single dilation for all layers."""

    padding: Literal["none", "zeros", "reflect", "replicate", "circular"] = "none"
    """Padding type to use."""

    norm: (
        Literal["none", "batch", "layer"]
        | tuple[Literal["none", "batch", "layer"], ...]
        | list[Literal["none", "batch", "layer"]]
    ) = "none"
    """Normalization type for each convolutional layer or a single normalization type for all layers."""

    activation: Activation = "elu"
    """Activation function to use."""

    max_pool: bool | tuple[bool, ...] | list[bool] = False
    """Whether to apply max pooling after each convolutional layer or a single boolean for all layers."""

    global_pool: Literal["none", "max", "avg"] = "none"
    """Global pooling type to apply at the end."""

    flatten: bool = True
    """Whether to flatten the output tensor."""


class CNNModelConfig(_MLPModelFields):
    """Convolutional neural network model config.

    Source: https://leggedrobotics.github.io/rsl_rl/guide/configuration.html#cnnmodel
    """

    class_name: Literal["CNNModel"] = "CNNModel"
    """Model class name."""

    cnn_cfg: CNNEncoderConfig | dict[str, CNNEncoderConfig] | None = None
    """Configuration of the CNN encoder(s). Either a single encoder config shared by all CNNs, or a mapping from observation name to the encoder config applying to it."""


ModelConfig = Annotated[
    MLPModelConfig | RNNModelConfig | CNNModelConfig,
    Field(discriminator="class_name"),
]
"""Any model config, selected by `class_name`."""


class _RunnerFields(ConfigBase):
    """Keys shared by `OnPolicyRunner` and `DistillationRunner`.

    Source: https://leggedrobotics.github.io/rsl_rl/guide/configuration.html#onpolicyrunner
    """

    obs_groups: dict[str, list[str]]
    """Mapping from observation sets to observation groups coming from the environment."""

    num_steps_per_env: int
    """Number of environment steps collected per iteration."""

    save_interval: int
    """Number of iterations between checkpoints."""

    run_name: str | None = None
    """Optional run label shown in the console output."""

    check_for_nan: bool = True
    """Whether to check for NaN values coming from the environment."""

    torch_compile_mode: Literal["default", "max-autotune-no-cudagraphs"] | None = None
    """Compile mode for the PyTorch models to accelerate training."""

    logger: Literal["tensorboard"] | LoggerConfig = "tensorboard"
    """Logging writer configuration."""


class OnPolicyRunnerConfig(_RunnerFields):
    """On-policy runner config.

    Source: https://leggedrobotics.github.io/rsl_rl/guide/configuration.html#onpolicyrunner
    """

    class_name: Literal["OnPolicyRunner"] = "OnPolicyRunner"
    """Class name of the runner to use."""

    algorithm: PPOConfig
    """RL algorithm configuration."""

    actor: ModelConfig
    """Actor model configuration."""

    critic: ModelConfig
    """Critic model configuration."""


class DistillationRunnerConfig(_RunnerFields):
    """Distillation runner config.

    Source: https://leggedrobotics.github.io/rsl_rl/guide/configuration.html#distillationrunner
    """

    class_name: Literal["DistillationRunner"] = "DistillationRunner"
    """Class name of the runner to use."""

    algorithm: DistillationConfig
    """RL algorithm configuration."""

    student: ModelConfig
    """Student model configuration."""

    teacher: ModelConfig
    """Teacher model configuration."""


RunnerConfig = Annotated[
    OnPolicyRunnerConfig | DistillationRunnerConfig, Field(discriminator="class_name")
]
"""Runner configuration."""

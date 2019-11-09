import tensorflow as tf  # type: ignore
import tensorflow.keras.layers as L  # type: ignore
from board import BOARD_SIZE, BOARD_SQUARES


def residual_block(
    x: tf.Tensor, filters: int, separable_convs: bool = False, prefix: str = ""
) -> tf.Tensor:
    """
    Make a residual block of layers.

    Parameters
    ----------
    x : tf.Tensor
        The input tensor.
    filters : int
        How many filters to use for each convolution.
    separable_convs : bool
        Whether to use depthwise separable convolution.
    prefix : str
        How to prefix layer names within the block.

    Returns
    -------
    tf.Tensor
        The output tensor.
    """

    if separable_convs:
        conv_fn = L.SeparableConv2D
    else:
        conv_fn = L.Conv2D

    y = conv_fn(filters, (3, 3), padding="same", name=prefix + "conv2d_1")(x)
    y = L.BatchNormalization(axis=-1, name=prefix + "batchnorm_1")(y)
    y = L.ReLU(name=prefix + "relu_1")(y)
    y = conv_fn(filters, (3, 3), padding="same", name=prefix + "conv2d_2")(y)
    y = L.BatchNormalization(axis=-1, name=prefix + "batchnorm_2")(y)
    y = L.add([x, y], name=prefix + "skip")
    y = L.ReLU(name=prefix + "relu_2")(y)
    return y


def get_network(
    filters: int,
    blocks: int,
    policy_head_filters: int = 2,
    value_head_filters: int = 1,
    value_dense_units: int = 64,
    separable_convs: bool = False,
    normalize_policy: bool = False,
) -> tf.keras.Model:
    """
    Make a (policy, value) network.

    Parameters
    ----------
    filters : int
        How many filters to use for the body convolutions.
    blocks : int
        How many residual blocks of depth to add.
    policy_head_filters : int
        How many filters to use for the initial policy head convolution.
    value_head_filters : int
        How many filters to use for the initial value head convolution.
    value_dense_units : int
        How many units to use for the value head's internal dense layer.
    separable_convs : bool
        If True, use depthwise-separable convolution.
    normalize_policy : bool
        If True, policy is run through a softmax. Do not use if masking.
    """

    if separable_convs:
        conv_fn = L.SeparableConv2D
    else:
        conv_fn = L.Conv2D

    inputs_ = L.Input(shape=(BOARD_SIZE, BOARD_SIZE, 2), name="board")
    x = conv_fn(filters, (3, 3), padding="same", name="input_conv2d")(inputs_)
    x = L.BatchNormalization(axis=-1, name="input_batchnorm")(x)
    x = L.ReLU(name="input_relu")(x)

    for block in range(blocks):
        x = residual_block(x, filters, separable_convs, prefix=f"block_{block + 1}_")

    # Policy head
    policy = L.Conv2D(policy_head_filters, (1, 1), name="policy_conv2d")(x)
    policy = L.BatchNormalization(axis=-1, name="policy_batchnorm")(policy)
    policy = L.ReLU(name="policy_relu")(policy)
    policy = L.Flatten(name="policy_flatten")(policy)
    policy = L.Dense(BOARD_SQUARES, name="policy_dense")(policy)
    if normalize_policy:
        policy = L.Softmax(name="policy_softmax")(policy)

    # Value head
    value = L.Conv2D(value_head_filters, (1, 1), name="value_conv2d")(x)
    value = L.BatchNormalization(axis=-1, name="value_batchnorm")(value)
    value = L.ReLU(name="value_relu_1")(value)
    value = L.Flatten(name="value_flatten")(value)
    value = L.Dense(value_dense_units, name="value_dense_1")(value)
    value = L.ReLU(name="value_relu_2")(value)
    value = L.Dense(1, name="value_dense_2")(value)
    value = tf.keras.activations.tanh(value)

    return tf.keras.Model(inputs=inputs_, outputs=[policy, value])

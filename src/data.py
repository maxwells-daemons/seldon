"""
Code for working with data.

In-memory format (as a list):
 - board: Tensor (8, 8, 2) [bool; one-hot]
 - move: Tensor (64,) [bool; one-hot]
 - value: Tensor () [float32]

On-disk format (to save space and quicken loading):
 - board: int64
 - move: int64
 - value: float32
"""

from typing import Dict, Tuple

import tensorflow as tf  # type: ignore
from board import BOARD_SHAPE, BOARD_SQUARES, Board, Loc

EXAMPLE_SPEC = {
    "board": tf.io.FixedLenFeature([2], tf.int64),
    "move": tf.io.FixedLenFeature([], tf.int64),
    "value": tf.io.FixedLenFeature([], tf.float32),
}


# Hack to allow storing bitboards efficiently as tf.Int64.
# Necessary because boards are all valid uint64 but not necessarily valid int64.
# Taken from: https://stackoverflow.com/questions/20766813/how-to-convert-signed-to-
#             unsigned-integer-in-python
def _signed_representation(unsigned: int) -> int:
    """Convert an "unsigned" int to its equivalent C "signed" representation."""
    return (unsigned & ((1 << 63) - 1)) - (unsigned & (1 << 63))


def _unsigned_representation(signed: int) -> int:
    """Convert a "signed" int to its equivalent C "unsigned" representation."""
    return signed & 0xFFFFFFFFFFFFFFFF


# See: https://stackoverflow.com/questions/48333210/tensorflow-how-to-convert-an-
#      integer-tensor-to-the-corresponding-binary-tensor
def decode_bitboard(encoded: tf.Tensor) -> tf.Tensor:
    """
    Convert from uint64 board representation to a tf.Tensor board.
    """
    flat = tf.math.mod(
        tf.bitwise.right_shift(encoded, tf.range(BOARD_SQUARES, dtype=tf.int64)), 2
    )
    board = tf.reshape(flat, BOARD_SHAPE)

    # Hack to allow using rot90 on a 2D tensor
    return tf.image.rot90(tf.expand_dims(board, axis=-1), k=2)[:, :, 0]


def serialize_example(board: Board, move: Loc, value: float) -> str:
    """
    Serialize a single training example into a string.
    """
    black = _signed_representation(int(board.black))
    white = _signed_representation(int(board.white))
    features = {
        "board": tf.train.Feature(int64_list=tf.train.Int64List(value=[black, white])),
        "move": tf.train.Feature(int64_list=tf.train.Int64List(value=[move.as_int])),
        "value": tf.train.Feature(float_list=tf.train.FloatList(value=[value])),
    }
    ex = tf.train.Example(features=tf.train.Features(feature=features))
    return ex.SerializeToString()


def preprocess_example(
    serialized: str
) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    """
    Turn a serialized example into the training-ready format.
    """

    example = tf.io.parse_single_example(serialized, EXAMPLE_SPEC)
    bitboards = example["board"]
    black_bb = bitboards[0]
    white_bb = bitboards[1]
    black = decode_bitboard(black_bb)
    white = decode_bitboard(white_bb)
    board = tf.stack([black, white], axis=-1)
    move = tf.one_hot(example["move"], BOARD_SQUARES)
    # TODO: better solution to multi-input Keras model training
    return (
        {"board": board},
        {"policy_softmax": move, "tf_op_layer_Tanh": example["value"]},
    )

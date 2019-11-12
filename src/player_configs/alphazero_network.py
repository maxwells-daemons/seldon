#!/usr/bin/env python
# cython: language_level=3, boundscheck=False, wraparound=False
from typing import Tuple

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from absl import app, flags  # type: ignore

import src.cs2_wrapper  # type: ignore
import src.players.alphazero_player  # type: ignore
from board import BOARD_SHAPE, PlayerColor

FLAGS = flags.FLAGS

flags.DEFINE_string("network", None, "Path to a saved Keras policy/value network.")
flags.DEFINE_float("C", 1.4, "MCTS exploration parameter.")
flags.DEFINE_integer(
    "solver_depth",
    18,
    "Depth to run deterministic endgame solver from. If -1, do not use a solver.",
)
flags.DEFINE_integer(
    "solver_time", 5000, "Milliseconds to reserve for endgame solving."
)
flags.DEFINE_integer(
    "sims", 100, "Number of simulations per turn if there is no time limit."
)
flags.DEFINE_integer("buffer_time", 80, "Milliseconds to reserve each turn.")


def main(argv):
    color = argv[0]
    model = tf.keras.models.load_model(FLAGS.network)

    def network_evaluator(board: np.ndarray) -> Tuple[np.ndarray, float]:
        policy, value = model.predict(np.expand_dims(board, axis=0).astype(np.float32))
        policy = np.reshape(policy, BOARD_SHAPE)
        return policy, value[0, 0]

    player = src.players.alphazero_player.AlphaZeroPlayer(
        evaluator=network_evaluator,
        explore_coeff=FLAGS.C,
        finalized=True,
        sims_per_turn=FLAGS.sims,
    )

    if FLAGS.solver_depth > 0:
        player = player.with_depth_solver(FLAGS.solver_depth, FLAGS.solver_time)

    p_color = {"Black": PlayerColor.BLACK, "White": PlayerColor.WHITE}[color]
    src.cs2_wrapper.run_player(player, color=p_color)


if __name__ == "__main__":
    app.run(main)

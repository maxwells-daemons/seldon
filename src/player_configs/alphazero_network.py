#!/usr/bin/env python
# cython: language_level=3, boundscheck=False, wraparound=False
from typing import Tuple

import click
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore

import src.cs2_wrapper  # type: ignore
import src.players.alphazero_player  # type: ignore
from board import BOARD_SHAPE, PlayerColor


@click.command()
@click.argument("color", type=click.Choice(["Black", "White"], case_sensitive=False))
@click.option(
    "--network",
    type=click.Path(exists=True),
    required=True,
    help="Path to a saved Keras policy/value network.",
)
@click.option("--C", type=float, default=1.4, help="MCTS exploration parameter.")
@click.option(
    "--solver-depth",
    type=int,
    default=18,
    help="Depth to run deterministic endgame solver from. If -1, do not use a solver.",
)
@click.option(
    "--solver-time",
    type=int,
    default=5000,
    help="Milliseconds to reserve for endgame solving.",
)
@click.option(
    "--sims",
    type=int,
    default=100,
    help="Number of simulations per turn if there is no time limit.",
)
@click.option(
    "--buffer_time", type=int, default=80, help="Milliseconds to reserve each turn."
)
def main(
    color: str,
    network: str,
    c: float = 1.4,
    solver_depth: int = 18,
    solver_time: int = 5000,
    sims: int = 100,
    buffer_time: int = 80,
):
    model = tf.keras.models.load_model(network)

    def network_evaluator(board: np.ndarray) -> Tuple[np.ndarray, float]:
        policy, value = model.predict(np.expand_dims(board, axis=0).astype(np.float32))
        policy = np.reshape(policy, BOARD_SHAPE)
        return policy, value[0, 0]

    player = src.players.alphazero_player.AlphaZeroPlayer(
        evaluator=network_evaluator, explore_coeff=c, finalized=True, sims_per_turn=sims
    )

    if solver_depth > 0:
        player = player.with_depth_solver(solver_depth, solver_time)

    p_color = {"Black": PlayerColor.BLACK, "White": PlayerColor.WHITE}[color]
    src.cs2_wrapper.run_player(player, color=p_color)


if __name__ == "__main__":
    main()

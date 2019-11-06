#!/usr/bin/env python
# cython: language_level=3, boundscheck=False, wraparound=False

import src.cs2_wrapper  # type: ignore
import src.players.mcts_player  # type: ignore

if __name__ == "__main__":
    player = src.players.mcts_player.MCTSPlayer(explore_coeff=4, turn_ms_buffer=40)
    src.cs2_wrapper.run_player(player)

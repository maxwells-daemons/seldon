#!/usr/bin/env python
# cython: language_level=3, boundscheck=False, wraparound=False
import src.cs2_wrapper  # type: ignore
import src.players.random_player  # type: ignore

if __name__ == "__main__":
    src.cs2_wrapper.run_player(src.players.random_player.RandomPlayer)

import random
import sys
from math import ceil, log, sqrt
from time import time
from typing import Dict, List, Optional

import numpy as np  # type: ignore

from board import BOARD_SQUARES, Board, GameOutcome, Loc, PlayerColor
from mcts_utils import random_rollout  # type: ignore
from player import PlayerABC

# TODO: debug mode

DEFAULT_N_TRAVERSALS = 100


class SearchTree:
    def __init__(self, board: Board, player: PlayerColor, explore_coeff: float):
        self.explore_coeff = explore_coeff
        self.player = player
        self.board: Board = board
        self.value: float = 0
        self.visits: int = 0
        self.unexplored: List[Loc] = board.find_moves(player).loc_list
        self.explored: Dict[Loc, SearchTree] = {}

    def _uct_score(self, parent_visits: int) -> float:
        return (self.value / self.visits) + self.explore_coeff * sqrt(
            log(parent_visits) / self.visits
        )

    def _apply_game_outcome(self, result: GameOutcome) -> None:
        self.visits += 1
        if result == GameOutcome.DRAW:
            self.value += 0.5
        elif result.value != self.player.value:
            self.value += 1

    def _random_rollout(self) -> GameOutcome:
        active, opp = self.board.player_view(self.player)
        result = random_rollout(active, opp, self.player)
        return result

    def traverse(self) -> GameOutcome:
        if self.unexplored:
            next_move = random.choice(self.unexplored)
            self.unexplored.remove(next_move)
            next_board = self.board.resolve_move(next_move, self.player)
            next_player = (
                self.player.opponent
                if next_board.has_moves(self.player.opponent)
                else self.player
            )

            next_node = SearchTree(next_board, next_player, self.explore_coeff)
            self.explored[next_move] = next_node
            result = next_node._random_rollout()
            next_node._apply_game_outcome(result)
        else:
            if self.explored == {}:  # Terminal node
                return self.board.winning_player

            # All children explored: pick one based on UCT scores, continue searching
            next_nodes = list(self.explored.values())
            uct_scores = [node._uct_score(self.visits) for node in next_nodes]
            next_node = next_nodes[np.argmax(uct_scores)]
            result = next_node.traverse()

        self._apply_game_outcome(result)
        return result


class MCTSPlayer(PlayerABC):
    def __init__(
        self,
        color: PlayerColor,
        ms_total: Optional[int],
        explore_coeff: float,
        turn_ms_buffer: int = 0,
    ) -> None:
        self.explore_coeff = explore_coeff
        self.search_tree: SearchTree = SearchTree(
            Board.starting_board(), PlayerColor.BLACK, self.explore_coeff
        )
        self.turn_ms_buffer = turn_ms_buffer
        super().__init__(color)

    def _get_move(
        self, board: Board, opponent_move: Optional[Loc], ms_left: Optional[int]
    ) -> Loc:
        if opponent_move is None:
            pass  # Opponent skips: keep the tree from our last move
        elif opponent_move in self.search_tree.explored.keys():
            self.search_tree = self.search_tree.explored[opponent_move]  # type: ignore
        else:
            print("WARNING: Opponent made a move we haven't explored.", file=sys.stderr)
            self.search_tree = SearchTree(board, self.color, self.explore_coeff)

        if ms_left:
            t1 = time() * 1000
            empties = BOARD_SQUARES - board.white.popcount - board.black.popcount
            n_moves_left = ceil(empties / 2)
            time_per_turn = ms_left / n_moves_left

            n = 0
            while time() * 1000 - t1 < time_per_turn - self.turn_ms_buffer:
                self.search_tree.traverse()
                n += 1
            print(f"SELDON: Searched {n} nodes.", file=sys.stderr)
        else:
            for _ in range(DEFAULT_N_TRAVERSALS):
                self.search_tree.traverse()

        moves_and_nodes = list(self.search_tree.explored.items())
        visit_counts = [node.visits for _, node in moves_and_nodes]
        move_idx = np.argmax(visit_counts)
        move, self.search_tree = moves_and_nodes[move_idx]

        win_rate = self.search_tree.value / self.search_tree.visits
        print(f"Move: {move}", file=sys.stderr)
        print(f"Expected win rate: {win_rate}.", file=sys.stderr)
        return move

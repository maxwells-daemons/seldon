from math import log, sqrt
from typing import List, Optional

import numpy as np  # type: ignore

from bitboard import _serialize, find_moves  # type: ignore
from game import (
    Board,
    GameOutcome,
    Move,
    PlayerABC,
    PlayerColor,
    player_make_move,
    starting_board,
)
from mcts_utils import random_rollout  # type: ignore
from utils import moves_list


class SearchTree:
    def __init__(
        self, board: Board, player: PlayerColor, exploration_parameter: float
    ) -> None:
        self.exploration_parameter = exploration_parameter
        self.player: PlayerColor = player
        self.board: Board = board
        self.value: float = 0
        self.visits: int = 0
        self.unexplored: List[Move] = moves_list(find_moves(*board.player_view(player)))
        self.explored: List[SearchTree] = []
        self.explored_moves: List[Move] = []

    def uct_score(self, parent_visits):
        return (self.value / self.visits) + self.exploration_parameter * sqrt(
            log(parent_visits) / self.visits
        )

    def _random_rollout(self) -> GameOutcome:
        active_bitboard, opponent_bitboard = self.board.player_view(self.player)
        result = random_rollout(
            _serialize(active_bitboard), _serialize(opponent_bitboard), self.player
        )
        self.visits += 1

        if result == GameOutcome.DRAW:
            self.value += 0.5
        elif result.value != self.player.value:
            self.value += 1

        return result

    def update(self) -> GameOutcome:
        if self.unexplored == []:
            if self.explored == []:  # Terminal node
                return self.board.winning_player()

            # Tree node: continue searching
            uct_scores = [child.uct_score(self.visits) for child in self.explored]
            next_node = self.explored[np.argmax(uct_scores)]
            result = next_node.update()
        else:  # Leaf node: expand and rollout
            next_move = self.unexplored.pop()
            next_board = player_make_move(self.board, next_move, self.player)

            if next_board.player_has_moves(self.player.opponent()):
                next_player = self.player.opponent()
            else:
                next_player = self.player

            next_node = SearchTree(next_board, next_player, self.exploration_parameter)
            self.explored.append(next_node)
            self.explored_moves.append(next_move)
            result = next_node._random_rollout()

        if result == GameOutcome.DRAW:
            self.value += 0.5
        elif result.value != self.player.value:
            self.value += 1

        self.visits += 1
        return result


class MCTSPlayer(PlayerABC):
    def __init__(
        self, color: PlayerColor, exploration_parameter: float, updates_per_move: int
    ) -> None:
        self.exploration_parameter = exploration_parameter
        self.updates_per_move = updates_per_move
        self._search_tree: SearchTree = SearchTree(
            starting_board(), PlayerColor.BLACK, self.exploration_parameter
        )
        super().__init__(color)

    # TODO: use opponent move
    def get_move(
        self,
        player_board: np.ndarray,
        opponent_board: np.ndarray,
        opponent_move: Optional[Move],
        ms_left: Optional[int],
    ) -> Move:
        # Our opponent has moved down the search tree by 1
        for node in self._search_tree.explored:
            player_board_, opponent_board_ = node.board.player_view(self.color)
            if np.array_equal(player_board, player_board_) and np.array_equal(
                opponent_board, opponent_board_
            ):
                self._search_tree = node
                break
        else:
            print("WARNING: Opponent made a move we haven't explored.")
            board = Board(
                **{
                    self.color.value: player_board,
                    self.color.opponent().value: opponent_board,
                }
            )
            self._search_tree = SearchTree(
                board, self.color, self.exploration_parameter
            )

        for _ in range(self.updates_per_move):
            self._search_tree.update()

        visit_counts = [node.visits for node in self._search_tree.explored]
        move_idx = np.argmax(visit_counts)

        move = self._search_tree.explored_moves[move_idx]
        self._search_tree = self._search_tree.explored[move_idx]
        return move

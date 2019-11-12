import math
from time import monotonic
from typing import Callable, Dict, Optional, Tuple

import numpy as np  # type: ignore
from scipy.special import softmax  # type: ignore

from board import BOARD_SQUARES, Board, Loc, PlayerColor
from player import PlayerABC

# Map from boards to (policy, value) pairs
Evaluator = Callable[[np.ndarray], Tuple[np.ndarray, float]]


# TODO: Dirichlet noise option
class SearchTree:
    board: Board
    move: Loc
    next_moves: Optional[Dict[Loc, "SearchTree"]]  # None when we've never been visited
    player: PlayerColor
    value: float
    visits: int
    policy_prior: float

    def __init__(
        self, board: Board, move: Loc, player: PlayerColor, policy_prior: float
    ):
        self.board = board
        self.move = move
        self.next_moves = None
        self.player = player
        self.value = board.score_for_player(player) if board.is_terminal else 0
        self.visits = 0
        self.policy_prior = policy_prior

    @property
    def _board_array(self) -> np.ndarray:
        mine, opp = self.board.player_view(self.player)
        return np.dstack([mine.piecearray, opp.piecearray])

    def _puct_score(self, parent_visits: int, explore_coeff: float) -> float:
        exploit = -self.value / self.visits if self.visits else 0
        explore = self.policy_prior * math.sqrt(parent_visits) / (1 + self.visits)
        return explore_coeff * explore + exploit

    def _make_child(self, move: Loc, policy_prior: float) -> "SearchTree":
        board = self.board.resolve_move(move, self.player)
        player = (
            self.player.opponent
            if board.has_moves(self.player.opponent)
            else self.player
        )
        return SearchTree(board, move, player, policy_prior)

    def _next_simulation_move(self, explore_coeff: float) -> "SearchTree":
        assert self.next_moves
        subtrees = list(self.next_moves.values())
        scores = map(lambda t: t._puct_score(self.visits, explore_coeff), subtrees)
        move_idx = np.argmax(list(scores))
        return subtrees[move_idx]

    def _expand(self, evaluator: Evaluator) -> float:
        assert self.next_moves is None, "Can only expand an unvisited node"

        policy, value = evaluator(self._board_array)

        # Mask illegal moves and re-normalize
        move_bitboard = self.board.find_moves(self.player)
        policy[np.where(~move_bitboard.piecearray)] = -math.inf
        policy = softmax(policy)
        self.next_moves = {
            loc: self._make_child(loc, policy[loc.y, loc.x])
            for loc in move_bitboard.loc_list
        }

        self.visits = 1
        self.value = value
        return value

    def simulate(self, evaluator: Evaluator, explore_coeff: float) -> float:
        if self.board.is_terminal:
            return self.value

        if self.next_moves is None:
            return self._expand(evaluator)

        next_move = self._next_simulation_move(explore_coeff)
        value = next_move.simulate(evaluator, explore_coeff)
        if self.player != next_move.player:
            value *= -1
        self.value += value
        self.visits += 1
        return value

    def best_move(self, deterministic: bool = True) -> "SearchTree":
        assert self.next_moves
        subtrees = list(self.next_moves.values())
        scores = list(map(lambda x: x.visits, subtrees))
        if deterministic:
            move_idx = np.argmax(scores)
        else:
            move_idx = np.random.choice(len(scores), p=scores)
        return subtrees[move_idx]

    def __repr__(self) -> str:
        if self.board.is_terminal:
            return f"{self.move}: terminal ({self.board.winning_player})"
        if not self.next_moves:
            return f"{self.move}: unvisited"

        return (
            f"{self.move}: visited (vs: {self.visits}, "
            f"ev: {self.value / self.visits:.3f}, "
            f"prior: {self.policy_prior:.3f})"
        )

    def summary(self) -> str:
        if self.board.is_terminal or not self.next_moves:
            return self.__repr__()

        move_visits = ", ".join([str(tree.visits) for tree in self.next_moves.values()])

        return f"""{self.move}: visited
    Visits: {self.visits}
    Expected value: {self.value / self.visits:.3f}
    Policy prior: {self.policy_prior:.3f}
    Subtree visits: [{move_visits}]"""


class AlphaZeroPlayer(PlayerABC):
    evaluator: Evaluator
    search_tree: "SearchTree"

    # Configuration
    explore_coeff: float
    finalized: bool
    time_buffer: int  # For fixed-time gameplay
    sims_per_turn: Optional[int]  # For fixed-moves gameplay

    def __init__(
        self,
        evaluator: Evaluator,
        explore_coeff: float,
        finalized: bool = False,
        time_buffer: int = 80,
        sims_per_turn: Optional[int] = None,
    ) -> None:
        self.evaluator = evaluator  # type: ignore
        self.explore_coeff = explore_coeff
        self.finalized = finalized
        self.time_buffer = time_buffer
        self.sims_per_turn = sims_per_turn
        self.search_tree = SearchTree(
            Board.starting_board(), Loc.pass_loc(), PlayerColor.BLACK, 1
        )

    def _get_move(
        self, board: Board, opponent_move: Optional[Loc], ms_left: Optional[int]
    ) -> Loc:
        if opponent_move:
            if self.search_tree.next_moves:
                self.search_tree = self.search_tree.next_moves[opponent_move]
            else:
                self.logger.warning(
                    "Opponent's move not considered. Generating a new tree..."
                )
                self.search_tree = SearchTree(board, opponent_move, self.color, 1)

            self.logger.debug(f"Opponent's node: {self.search_tree}")

            if not self.search_tree.visits:
                self.logger.warning("Off tree!")

        if not board.has_moves(self.color):
            return Loc.pass_loc()

        t1 = monotonic()
        if ms_left:  # Constant time
            empties = BOARD_SQUARES - board.white.popcount - board.black.popcount
            n_moves_left = math.ceil(empties / 2)
            time_per_turn = ms_left / n_moves_left

            n_sims = 0
            while monotonic() * 1000 - (t1 * 1000) < time_per_turn - self.time_buffer:
                self.search_tree.simulate(
                    self.evaluator, self.explore_coeff  # type: ignore
                )
                n_sims += 1

        else:  # Constant number of simulations per move
            assert self.sims_per_turn
            n_sims = self.sims_per_turn
            for _ in range(self.sims_per_turn):
                self.search_tree.simulate(
                    self.evaluator, self.explore_coeff  # type: ignore
                )

        self.search_tree = self.search_tree.best_move(deterministic=self.finalized)

        move_time = monotonic() - t1
        self.logger.debug(
            f"Completed {n_sims} simulations in {move_time:.2f} seconds "
            f"({n_sims / move_time:.2f} sims/sec)."
        )
        self.logger.debug(f"Chosen node: {self.search_tree.summary()}")

        return self.search_tree.move

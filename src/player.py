import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from time import monotonic
from types import MethodType
from typing import Optional

from board import BOARD_SQUARES, Board, Loc, PlayerColor
from solver import solve_game  # type: ignore
from termcolor import colored


"""
Definitions and functions related to Othello players.

Attributes
----------
PlayerABC : ABC
    Abstract base class for Othello players.
"""


class PlayerABC(ABC):
    """
    ABC for Othello players.

    Attributes
    ----------
    color : PlayerColor
        This player's color.
    logger : logging.Logger
        A logger for formatting and printing this player's output.
    """

    _initialized = False

    def initialize(self, color: PlayerColor, ms_total: Optional[int]) -> None:
        """
        Initialize a player's attributes. Used by the game framework; do not overwrite.

        Parameters
        ----------
        color : PlayerColor
            This player's color.
        ms_left : int or None
            Milliseconds total in this bot's time budget.
            If None, unlimited time is available.
        """
        self._initialized = True
        self.color: PlayerColor = color
        self.ms_total: Optional[int] = ms_total

        log_name = f"{self.__class__.__name__} ({self.color.value})"
        self.logger: logging.Logger = logging.getLogger(log_name)
        if not self.logger.handlers:
            log_handler = logging.StreamHandler()
            log_handler.setLevel(logging.DEBUG)
            log_handler.setFormatter(
                logging.Formatter(
                    fmt=f"{log_name} %(levelname)s >>- %(message)s", datefmt="%H:%M:%S"
                )
            )
            self.logger.addHandler(log_handler)
        self.logger.setLevel(logging.DEBUG)

        if ms_total:
            self.logger.debug(
                f"Time per move: {2 * ms_total / (BOARD_SQUARES * 1000):.2f} s."
            )
        self.logger.debug("Finished initializing player.")

    def get_move(
        self, board: Board, opponent_move: Optional[Loc], ms_left: Optional[int]
    ) -> Loc:
        assert self._initialized
        opp_fmt = "pass" if opponent_move is None else opponent_move.__repr__()
        self.logger.info(f"Opponent's move: {colored(opp_fmt, 'red')}.")

        t1 = monotonic()
        move = self._get_move(board, opponent_move, ms_left)
        t2 = monotonic()

        move_count = board.white.popcount + board.black.popcount - 3
        move_format = colored(move.__repr__(), "yellow")
        time_format = colored(f"{t2 - t1:.2f} s", "green")
        self.logger.info(f"Move {move_count}: {move_format} (time: {time_format}).\n")
        return move

    @abstractmethod
    def _get_move(
        self, board: Board, opponent_move: Optional[Loc], ms_left: Optional[int]
    ) -> Loc:
        """
        Get this player's next move.

        If there are no legal moves, the player must return Loc(-1, -1).

        Parameters
        ----------
        board : Board
            The current board.
        opponent_move : Loc or None
            Opponent's last move, if applicable. If this is the first move of the game
            or the opponent passed, this will be None.
        ms_left : int or None
            Milliseconds left in this bot's time budget.
            If None, unlimited time is available.

        Returns
        -------
        Loc
            The player's next move.
        """
        raise NotImplementedError

    def with_depth_solver(self, depth: int, time: int) -> "PlayerABC":
        """
        Wrap a player such that after few enough spots are left empty, moves are made
        with the endgame solver.

        Parameters
        ----------
        depth : int
            Depth to start solving endgame at.
        time : int
            How many milliseconds to reserve for endgame solving.

        Returns
        -------
        PlayerABC
            A copy of this player with a different _get_move function.
        """

        player = deepcopy(self)
        __get_move = player._get_move

        def _get_move(
            self: PlayerABC,
            board: Board,
            opponent_move: Optional[Loc],
            ms_left: Optional[int],
        ) -> Loc:
            empties = BOARD_SQUARES - board.white.popcount - board.black.popcount

            if empties <= depth:
                if not board.has_moves(self.color):
                    return Loc.pass_loc()

                self.logger.info(
                    f"Running solver at depth: {colored(str(empties), 'cyan')}."
                )
                mine, opp = board.player_view(self.color)
                x, y, _ = solve_game(mine.piecearray, opp.piecearray)
                return Loc(x, y)

            if ms_left:
                ms_left -= time

            return __get_move(board, opponent_move, ms_left)

        player._get_move = MethodType(_get_move, player)  # type: ignore
        return player

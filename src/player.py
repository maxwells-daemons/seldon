import logging
from abc import ABC, abstractmethod
from time import monotonic
from typing import Optional, Type

from termcolor import colored

from board import BOARD_SQUARES, Board, Loc, PlayerColor
from solver import solve_game  # type: ignore


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

    Parameters
    ----------
    color : PlayerColor
        This player's color.
    ms_left : int or None
        Milliseconds total in this bot's time budget.
        If None, unlimited time is available.

    Attributes
    ----------
    color : PlayerColor
        This player's color.
    logger : logging.Logger
        A logger for formatting and printing this player's output.
    """

    def __init__(self, color: PlayerColor, ms_total: Optional[int] = None) -> None:
        self.color: PlayerColor = color
        self.ms_total: Optional[int] = ms_total

        log_name = f"{self.__class__.__name__} ({self.color.value})"
        self.logger: logging.Logger = logging.getLogger(log_name)
        log_handler = logging.StreamHandler()
        log_handler.setLevel(logging.DEBUG)
        log_handler.setFormatter(
            logging.Formatter(
                fmt=f"{log_name} %(levelname)s >>- %(message)s", datefmt="%H:%M:%S"
            )
        )
        self.logger.addHandler(log_handler)
        self.logger.setLevel(logging.DEBUG)

        self.logger.debug("Finished initializing player.")

    def get_move(
        self, board: Board, opponent_move: Optional[Loc], ms_left: Optional[int]
    ) -> Loc:
        t1 = monotonic()
        move = self._get_move(board, opponent_move, ms_left)
        t2 = monotonic()

        move_format = colored(move.__repr__(), "yellow")
        time_format = colored(f"{t2 - t1:.2f} s", "green")
        self.logger.info(f"Made move: {move_format} (time: {time_format}).")
        return move

    @abstractmethod
    def _get_move(
        self, board: Board, opponent_move: Optional[Loc], ms_left: Optional[int]
    ) -> Loc:
        """
        Get this player's next move.

        There is guaranteed to be at least one legal move.

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

    @classmethod
    def with_depth_solver(cls, depth: int, time: int) -> Type["PlayerABC"]:
        """
        Wrap a player such that after few enough spots are left empty, moves are made
        with the endgame solver.

        Parameters
        ----------
        depth : int
            Depth to start solving endgame at.
        time : int
            How many milliseconds to reserve for endgame solving.
        """

        class _PlayerWithSolver(PlayerABC):
            def __init__(
                self, color: PlayerColor, ms_total: Optional[int] = None, **kwargs
            ) -> None:
                if ms_total:
                    ms_total -= time
                self.player = cls(color, ms_total, **kwargs)  # type: ignore
                super().__init__(color, ms_total)
                self.logger = self.player.logger
                self.logger.propagate = False

            def _get_move(
                self, board: Board, opponent_move: Optional[Loc], ms_left: Optional[int]
            ) -> Loc:
                empties = BOARD_SQUARES - board.white.popcount - board.black.popcount

                if empties <= depth:
                    self.logger.debug(
                        f"Running solver at depth: {colored(str(empties), 'red')}."
                    )
                    mine, opp = board.player_view(self.color)
                    x, y, _ = solve_game(mine.piecearray, opp.piecearray)
                    return Loc(x, y)

                if ms_left:
                    ms_left -= time

                return self.player._get_move(board, opponent_move, ms_left)

        return _PlayerWithSolver

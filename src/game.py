from abc import ABC, abstractmethod
from enum import Enum, auto
from string import ascii_lowercase
from typing import Dict, NamedTuple, Type

import numpy as np  # type: ignore

from backend import find_moves, resolve_move  # type: ignore

BOARD_SIZE = 8
BOARD_SHAPE = (BOARD_SIZE, BOARD_SIZE)


class Move(NamedTuple):
    x: int
    y: int

    def __repr__(self) -> str:
        return ascii_lowercase[self.x] + str(self.y + 1)

    def __lt__(self, other):
        return self.x < other.x or self.y < other.y


class Board(NamedTuple):
    black: np.ndarray
    white: np.ndarray

    def _string_array(self) -> np.ndarray:
        board = np.tile(" ", (BOARD_SIZE + 1, BOARD_SIZE + 1))
        board[0, 1:] = [x for x in ascii_lowercase[:BOARD_SIZE]]
        board[1:, 0] = range(1, BOARD_SIZE + 1)
        board[0, 0] = " "
        board[1:, 1:][np.where(self.black)] = "X"
        board[1:, 1:][np.where(self.white)] = "O"
        return board

    def __repr__(self) -> str:
        return np.array2string(self._string_array(), formatter={"numpystr": str})


class PlayerColor(Enum):
    BLACK = "black"  # Values align with Board tuple names to allow getting pieces
    WHITE = "white"

    def opponent(self):
        if self == PlayerColor.WHITE:
            return PlayerColor.BLACK
        return PlayerColor.WHITE


class GameOutcome(Enum):
    BLACK_WINS = auto()
    WHITE_WINS = auto()
    DRAW = auto()


class PlayerABC(ABC):
    """
    ABC for Othello players.

    Parameters
    ----------
    color : PlayerColor
        This player's color.

    Attributes
    ----------
    color : PlayerColor
        This player's color.
    """

    def __init__(self, color: PlayerColor) -> None:
        self.color = color

    @abstractmethod
    def get_move(self, player_board: np.ndarray, opponent_board: np.ndarray) -> Move:
        """
        Get this player's next move.

        There is guaranteed to be at least one legal move.

        Parameters
        ----------
        player_board : ndarray
            A bitboard of this player's pieces.
        opponent_board : ndarray
            A bitboard of the opponent's pieces.

        Returns
        -------
        Move
            The player's next mvoe.
        """
        raise NotImplementedError


def starting_board() -> Board:
    black = np.zeros(BOARD_SHAPE, dtype=bool)
    white = np.zeros(BOARD_SHAPE, dtype=bool)
    black[3, 4] = black[4, 3] = True
    white[3, 3] = white[4, 4] = True
    return Board(black, white)


def player_has_moves(player_board: Board, opp_board: Board) -> bool:
    return find_moves(player_board, opp_board).any()


def play_game(black: Type[PlayerABC], white: Type[PlayerABC]) -> GameOutcome:
    """
    Play a complete game of Othello between two players.

    Parameters
    ----------
    black : PlayerABC constructor
        Player class for the black player.
    white : PlayerABC constructor
        Player class for the white player.

    Returns
    -------
    GameOutcome
        The outcome of the game.
    """
    current_player: PlayerColor = PlayerColor.BLACK
    players: Dict[PlayerColor, PlayerABC] = {
        PlayerColor.BLACK: black(PlayerColor.BLACK),
        PlayerColor.WHITE: white(PlayerColor.WHITE),
    }
    board: Board = starting_board()
    just_passed: bool = False

    while True:
        player_board = board.__getattribute__(current_player.value)
        opponent_board = board.__getattribute__(current_player.opponent().value)

        if not player_has_moves(player_board, opponent_board):
            if just_passed:
                break  # Both players pass: game ends
            just_passed = True
            current_player = current_player.opponent()
            continue

        move = players[current_player].get_move(player_board, opponent_board)
        new_player_board, new_opponent_board = resolve_move(
            player_board, opponent_board, move.x, move.y
        )
        board = Board(
            **{
                current_player.value: new_player_board,
                current_player.opponent().value: new_opponent_board,
            }
        )

        current_player = current_player.opponent()

    black_score = np.count_nonzero(board.black)
    white_score = np.count_nonzero(board.white)
    if black_score > white_score:
        return GameOutcome.BLACK_WINS
    if white_score > black_score:
        return GameOutcome.WHITE_WINS
    return GameOutcome.DRAW

from abc import ABC, abstractmethod
from enum import Enum, auto
from string import ascii_lowercase
from typing import Dict, NamedTuple, Tuple, Type

import numpy as np  # type: ignore

from bitboard import find_moves, resolve_move  # type: ignore

BOARD_SIZE = 8
BOARD_SHAPE = (BOARD_SIZE, BOARD_SIZE)


class PlayerColor(Enum):
    BLACK = "black"  # Values align with Board tuple names to allow getting pieces
    WHITE = "white"

    def opponent(self):
        if self == PlayerColor.WHITE:
            return PlayerColor.BLACK
        return PlayerColor.WHITE


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

    def for_side(self, player: PlayerColor) -> Tuple[np.ndarray, np.ndarray]:
        player_board = self.__getattribute__(player.value)
        opponent_board = self.__getattribute__(player.opponent().value)
        return player_board, opponent_board

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


class GameOutcome(Enum):
    BLACK_WINS = "black"  # Values align with PlayerColor to compare
    WHITE_WINS = "white"
    DRAW = auto()


def player_wins(player: PlayerColor) -> GameOutcome:
    if player == PlayerColor.BLACK:
        return GameOutcome.BLACK_WINS
    return GameOutcome.WHITE_WINS


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
        self.color: PlayerColor = color

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


def winning_player(board: Board) -> GameOutcome:
    black_score = np.count_nonzero(board.black)
    white_score = np.count_nonzero(board.white)
    if black_score > white_score:
        return GameOutcome.BLACK_WINS
    if white_score > black_score:
        return GameOutcome.WHITE_WINS
    return GameOutcome.DRAW


def player_make_move(board: Board, move: Move, player: PlayerColor) -> Board:
    player_board, opponent_board = board.for_side(player)
    new_player_board, new_opponent_board = resolve_move(
        player_board, opponent_board, move.x, move.y
    )
    return Board(
        **{player.value: new_player_board, player.opponent().value: new_opponent_board}
    )


def play_game_from_state(
    board: Board,
    current_player: PlayerColor,
    black: Type[PlayerABC],
    white: Type[PlayerABC],
) -> GameOutcome:
    """
    Play a game of Othello between two players, starting from a given state.

    Parameters
    ----------
    board : Board
        The starting board.
    current_player : PlayerColor
        The starting player.
    black : PlayerABC constructor
        Player class for the black player.
    white : PlayerABC constructor
        Player class for the white player.

    Returns
    -------
    GameOutcome
        The outcome of the game.
    """

    players: Dict[PlayerColor, PlayerABC] = {
        PlayerColor.BLACK: black(PlayerColor.BLACK),
        PlayerColor.WHITE: white(PlayerColor.WHITE),
    }
    just_passed: bool = False

    while True:
        player_board, opponent_board = board.for_side(current_player)

        if not player_has_moves(player_board, opponent_board):
            if just_passed:
                break  # Both players pass: game ends
            just_passed = True
            current_player = current_player.opponent()
            continue

        just_passed = False
        move = players[current_player].get_move(player_board, opponent_board)
        board = player_make_move(board, move, current_player)
        current_player = current_player.opponent()

    return winning_player(board)


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
    return play_game_from_state(starting_board(), PlayerColor.BLACK, black, white)

from abc import ABC, abstractmethod
from enum import Enum, auto
from string import ascii_lowercase
from time import time
from typing import Dict, NamedTuple, Optional, Tuple, Type, cast

import numpy as np  # type: ignore

from bitboard import find_moves, resolve_move  # type: ignore

BOARD_SIZE = 8
BOARD_SHAPE = (BOARD_SIZE, BOARD_SIZE)


# NOTE: values align between these enums to allow comparison
class GameOutcome(Enum):
    BLACK_WINS = "black"
    WHITE_WINS = "white"
    DRAW = auto()


class PlayerColor(Enum):
    BLACK = "black"
    WHITE = "white"

    def opponent(self):
        if self == PlayerColor.BLACK:
            return PlayerColor.WHITE
        return PlayerColor.BLACK

    def winning_outcome(self) -> GameOutcome:
        if self == PlayerColor.BLACK:
            return GameOutcome.BLACK_WINS
        return GameOutcome.WHITE_WINS


class Board(NamedTuple):
    black: np.ndarray
    white: np.ndarray

    def player_view(self, player: PlayerColor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve (my board, opponent board) tuple.
        """
        if player == PlayerColor.BLACK:
            return self.black, self.white
        return self.white, self.black

    def player_has_moves(self, player: PlayerColor) -> bool:
        return find_moves(*self.player_view(player)).any()

    def winning_player(self) -> GameOutcome:
        black_score = np.count_nonzero(self.black)
        white_score = np.count_nonzero(self.white)
        if black_score > white_score:
            return GameOutcome.BLACK_WINS
        if white_score > black_score:
            return GameOutcome.WHITE_WINS
        return GameOutcome.DRAW

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


class Move(NamedTuple):
    x: int
    y: int

    def __repr__(self) -> str:
        return ascii_lowercase[self.x] + str(self.y + 1)

    def __lt__(self, other):
        return self.x < other.x or self.y < other.y


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
    def get_move(
        self,
        player_board: np.ndarray,
        opponent_board: np.ndarray,
        opponent_move: Optional[Move],
        ms_left: Optional[int],
    ) -> Move:
        """
        Get this player's next move.

        There is guaranteed to be at least one legal move.

        Parameters
        ----------
        player_board : ndarray
            A bitboard of this player's pieces.
        opponent_board : ndarray
            A bitboard of the opponent's pieces.
        opponent_move : Move or None
            Opponent's last move, if applicable. If this is the first move of the game
            or the opponent passed, this will be None.
        ms_left : int or None
            Milliseconds left in this bot's time budget.
            If None, unlimited time is available.

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


def player_make_move(board: Board, move: Move, player: PlayerColor) -> Board:
    player_board, opponent_board = board.player_view(player)
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
    black_time: Optional[int],
    white_time: Optional[int],
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
    black_time : int or None
        Milliseconds left in the black bot's time budget.
        If None, unlimited time is available.
    white_time : int or None
        Milliseconds left in the white bot's time budget.
        If None, unlimited time is available.

    Returns
    -------
    GameOutcome
        The outcome of the game.
    """

    players: Dict[PlayerColor, PlayerABC] = {
        PlayerColor.BLACK: black(PlayerColor.BLACK),
        PlayerColor.WHITE: white(PlayerColor.WHITE),
    }
    times: Dict[PlayerColor, Optional[int]] = {
        PlayerColor.BLACK: black_time,
        PlayerColor.WHITE: white_time,
    }
    just_passed: bool = False
    last_move: Optional[Move] = None

    while True:
        if board.player_has_moves(current_player):
            just_passed = False
            player_board, opponent_board = board.player_view(current_player)

            t1 = time()
            move = players[current_player].get_move(
                player_board, opponent_board, last_move, times[current_player]
            )
            t2 = time()

            last_move = move
            board = player_make_move(board, move, current_player)

            if times[current_player] is not None:
                times[current_player] -= int((t2 - t1) * 1000)  # type: ignore
                if times[current_player] < 0:  # type: ignore
                    print(f"Player {current_player} timed out.")
                    return current_player.opponent().winning_outcome()
        else:
            if just_passed:
                break  # Both players pass: game ends
            just_passed = True
            last_move = None
        current_player = current_player.opponent()

    return board.winning_player()


def play_game(
    black: Type[PlayerABC], white: Type[PlayerABC], max_time: Optional[int] = None
) -> GameOutcome:
    """
    Play a complete game of Othello between two players.

    Parameters
    ----------
    black : PlayerABC constructor
        Player class for the black player.
    white : PlayerABC constructor
        Player class for the white player.
    max_time : int or None
        Maximum number of milliseconds to allow each bot to compute for.
        If None, unlimited time is available.

    Returns
    -------
    GameOutcome
        The outcome of the game.
    """
    return play_game_from_state(
        starting_board(), PlayerColor.BLACK, black, white, max_time, max_time
    )

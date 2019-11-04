"""
Python object-level interface to board representation objects.

Implemented in terms of efficient low-level C operations. Python code should use this
and not the wrapped C functions.

Attributes
----------
PlayerColor : Enum
    A player's color.
GameOutcome : Enum
    The outcome of an Othello game.
Loc : NamedTuple
    A pretty-printed and named (x, y) tuple representing a location on the board.
Bitboard : class
    A single player's set of pieces.
Board : class
    A complete Othello board.
"""

from enum import Enum, auto
from string import ascii_lowercase
from typing import List, NamedTuple, Tuple

import numpy as np  # type: ignore

from bitboard import (  # type: ignore
    bitboard_find_moves,
    bitboard_stability,
    deserialize_piecearray,
    make_singleton_bitboard,
    popcount,
    resolve_move,
    serialize_piecearray,
)

BOARD_SIZE = 8
BOARD_SHAPE = (BOARD_SIZE, BOARD_SIZE)
BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE


# NOTE: values align between these two enums to allow comparison
class GameOutcome(Enum):
    BLACK_WINS = "black"
    WHITE_WINS = "white"
    DRAW = auto()

    @property
    def opponent(self) -> "GameOutcome":
        if self == GameOutcome.BLACK_WINS:
            return GameOutcome.WHITE_WINS
        if self == GameOutcome.WHITE_WINS:
            return GameOutcome.BLACK_WINS
        return GameOutcome.DRAW


class PlayerColor(Enum):
    BLACK = "black"
    WHITE = "white"

    @property
    def opponent(self) -> "PlayerColor":
        if self == PlayerColor.BLACK:
            return PlayerColor.WHITE
        return PlayerColor.BLACK

    @property
    def winning_outcome(self) -> GameOutcome:
        if self == PlayerColor.BLACK:
            return GameOutcome.BLACK_WINS
        return GameOutcome.WHITE_WINS


class Loc(NamedTuple):
    "A pretty-printed and named (x, y) tuple representing a location on the board."
    x: int
    y: int

    @staticmethod
    def pass_loc() -> "Loc":
        return Loc(-1, -1)

    def __repr__(self) -> str:
        if self == Loc.pass_loc():
            return "pass"
        return ascii_lowercase[self.x] + str(self.y + 1)

    def __lt__(self, other):
        return self.x < other.x or self.y < other.y


class Bitboard(int):
    "A single player's set of pieces."

    @staticmethod
    def singleton(x: int, y: int) -> "Bitboard":
        return Bitboard(make_singleton_bitboard(x, y))

    @staticmethod
    def from_piecearray(piecearray: np.ndarray) -> "Bitboard":
        return Bitboard(serialize_piecearray(piecearray))

    @property
    def popcount(self) -> int:
        return popcount(self)

    @property
    def piecearray(self) -> np.ndarray:
        return deserialize_piecearray(self)

    @property
    def loc_list(self) -> List[Loc]:
        return [Loc(x, y) for y, x in np.argwhere(self.piecearray)]

    def __repr__(self) -> str:
        return np.array2string(self.piecearray.astype("int"))


class Board(NamedTuple):
    "A complete Othello board."
    black: Bitboard
    white: Bitboard

    @staticmethod
    def from_player_view(
        player_bitboard: Bitboard, opponent_bitboard: Bitboard, player: PlayerColor
    ):
        """
        Given a player color, bitboard, and opponent's bitboard, make a Board.
        """
        return Board(
            **{player.value: player_bitboard, player.opponent.value: opponent_bitboard}
        )

    @staticmethod
    def starting_board() -> "Board":
        return Board(Bitboard(0x0000000810000000), Bitboard(0x0000001008000000))

    def player_view(self, player: PlayerColor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve (my board, opponent board) tuple.
        """
        if player == PlayerColor.BLACK:
            return self.black, self.white
        return self.white, self.black

    def resolve_move(self, move: Loc, player: PlayerColor) -> "Board":
        player_board, opponent_board = self.player_view(player)
        new_player_board, new_opponent_board = resolve_move(
            player_board, opponent_board, move.x, move.y
        )
        return Board.from_player_view(
            Bitboard(new_player_board), Bitboard(new_opponent_board), player
        )

    def find_moves(self, player: PlayerColor) -> Bitboard:
        player_board, opponent_board = self.player_view(player)
        return Bitboard(bitboard_find_moves(player_board, opponent_board))

    def has_moves(self, player: PlayerColor) -> bool:
        return self.find_moves(player) != 0

    def find_stability(self, player: PlayerColor) -> Bitboard:
        player_board, opponent_board = self.player_view(player)
        return Bitboard(bitboard_stability(player_board, opponent_board))

    @property
    def winning_player(self) -> GameOutcome:
        black_score = self.black.popcount
        white_score = self.white.popcount
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
        board[1:, 1:][np.where(self.black.piecearray)] = "X"
        board[1:, 1:][np.where(self.white.piecearray)] = "O"
        return board

    def __repr__(self) -> str:
        return np.array2string(self._string_array(), formatter={"numpystr": str})

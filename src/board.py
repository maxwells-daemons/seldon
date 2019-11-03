"""
Python object-level interface to board representation objects.

Implemented in terms of efficient low-level C operations. Python code should use this
and not the wrapped C functions.

Attributes
----------
Loc : NamedTuple
    A pretty-printed and named (x, y) tuple representing a location on the board.
Bitboard : class
    A single player's set of pieces.
Board : class
    A complete Othello board.
"""

from string import ascii_lowercase
from typing import List, NamedTuple, Tuple

import numpy as np  # type: ignore

from bitboard import (  # type: ignore
    bitboard_find_moves,
    bitboard_resolve_move,
    bitboard_stability,
    deserialize_piecearray,
    make_singleton_bitboard,
    popcount,
    serialize_piecearray,
)
from player import PlayerColor

BOARD_SIZE = 8
BOARD_SHAPE = (BOARD_SIZE, BOARD_SIZE)


class Loc(NamedTuple):
    "A pretty-printed and named (x, y) tuple representing a location on the board."
    x: int
    y: int

    def __repr__(self) -> str:
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
        return Board(Bitboard(34628173824), Bitboard(68853694464))

    def player_view(self, player: PlayerColor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve (my board, opponent board) tuple.
        """
        if player == PlayerColor.BLACK:
            return self.black, self.white
        return self.white, self.black

    def resolve_move(self, player: PlayerColor, move: Loc) -> "Board":
        player_board, opponent_board = self.player_view(player)
        new_player_board, new_opponent_board = bitboard_resolve_move(
            player_board, opponent_board, move.x, move.y
        )
        return Board.from_player_view(new_player_board, new_opponent_board, player)

    def find_moves(self, player: PlayerColor) -> Bitboard:
        player_board, opponent_board = self.player_view(player)
        return Bitboard(bitboard_find_moves(player_board, opponent_board))

    def has_moves(self, player: PlayerColor) -> bool:
        return self.find_moves(player) != 0

    def find_stability(self, player: PlayerColor) -> Bitboard:
        player_board, opponent_board = self.player_view(player)
        return Bitboard(bitboard_stability(player_board, opponent_board))

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

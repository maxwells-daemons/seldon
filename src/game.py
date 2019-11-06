"""
Functions to run Othello games.
"""


import logging
from time import monotonic
from typing import Dict, Optional

from board import Board, GameOutcome, Loc, PlayerColor
from player import PlayerABC


def play_game_from_state(
    board: Board,
    current_player: PlayerColor,
    black: PlayerABC,
    white: PlayerABC,
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
    black : PlayerABC
        The black player.
    white : PlayerABC
        The white player.
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

    black.initialize(PlayerColor.BLACK, black_time)
    white.initialize(PlayerColor.WHITE, white_time)

    players: Dict[PlayerColor, PlayerABC] = {
        PlayerColor.BLACK: black,
        PlayerColor.WHITE: white,
    }
    times: Dict[PlayerColor, Optional[int]] = {
        PlayerColor.BLACK: black_time,
        PlayerColor.WHITE: white_time,
    }
    just_passed: bool = False
    last_move: Optional[Loc] = None

    while True:
        t1 = monotonic()
        move = players[current_player].get_move(board, last_move, times[current_player])
        t2 = monotonic()

        if move == Loc.pass_loc():
            if just_passed:
                break
            just_passed = True
            last_move = None
        else:
            just_passed = False
            last_move = move
            board = board.resolve_move(move, current_player)

        if times[current_player] is not None:
            times[current_player] -= int((t2 - t1) * 1000)  # type: ignore
            if times[current_player] < 0:  # type: ignore
                logging.error(f"Player {current_player} timed out.")
                return current_player.opponent.winning_outcome

        current_player = current_player.opponent

    return board.winning_player


def play_game(
    black: PlayerABC, white: PlayerABC, max_time: Optional[int] = None
) -> GameOutcome:
    """
    Play a complete game of Othello between two players.

    Parameters
    ----------
    black : PlayerABC
        The black player.
    white : PlayerABC
        The white player.
    max_time : int or None
        Maximum number of milliseconds to allow each bot to compute for.
        If None, unlimited time is available.

    Returns
    -------
    GameOutcome
        The outcome of the game.
    """
    return play_game_from_state(
        Board.starting_board(), PlayerColor.BLACK, black, white, max_time, max_time
    )

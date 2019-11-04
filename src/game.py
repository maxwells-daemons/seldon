"""
Functions to run Othello games.
"""


import logging
from time import monotonic
from typing import Dict, Optional, Type

from board import Board, GameOutcome, Loc, PlayerColor
from player import PlayerABC


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
        PlayerColor.BLACK: black(PlayerColor.BLACK, black_time),
        PlayerColor.WHITE: white(PlayerColor.WHITE, white_time),
    }
    times: Dict[PlayerColor, Optional[int]] = {
        PlayerColor.BLACK: black_time,
        PlayerColor.WHITE: white_time,
    }
    just_passed: bool = False
    last_move: Optional[Loc] = None

    while True:
        if board.has_moves(current_player):
            just_passed = False

            t1 = monotonic()
            move = players[current_player].get_move(
                board, last_move, times[current_player]
            )
            t2 = monotonic()

            last_move = move
            board = board.resolve_move(move, current_player)

            if times[current_player] is not None:
                times[current_player] -= int((t2 - t1) * 1000)  # type: ignore
                if times[current_player] < 0:  # type: ignore
                    logging.error(f"Player {current_player} timed out.")
                    return current_player.opponent.winning_outcome
        else:
            if just_passed:
                break  # Both players pass: game ends
            just_passed = True
            last_move = None
        current_player = current_player.opponent

    return board.winning_player


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
        Board.starting_board(), PlayerColor.BLACK, black, white, max_time, max_time
    )

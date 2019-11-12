#!/usr/bin/env python

from typing import Optional

import click

from game import play_game
from players.external_player import ExternalPlayer  # type: ignore


@click.command()
@click.argument("black", type=str)
@click.argument("white", type=str)
@click.option(
    "--max_time",
    type=click.IntRange(min=1),
    default=None,
    help="Milliseconds to give each player. If unspecified, there is no maximum time.",
)
def main(black: str, white: str, max_time: Optional[int] = None):
    black_player = ExternalPlayer(black)
    white_player = ExternalPlayer(white)
    outcome = play_game(black_player, white_player, max_time, show_board=True)
    print(f"Outcome: {outcome}")


if __name__ == "__main__":
    main()

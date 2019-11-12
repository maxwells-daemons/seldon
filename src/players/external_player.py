"""
A wrapper for external players that conform to the CS2 standard.
"""

import os
import subprocess
from typing import Optional

from board import Board, Loc, PlayerColor
from player import PlayerABC


class ExternalPlayer(PlayerABC):
    player_command: str

    def __init__(self, player_command: str):
        self.player_command = player_command
        self.__class__.__name__ = os.path.basename(player_command.split()[0])

    def initialize(self, color: PlayerColor, ms_total: Optional[int]) -> None:
        self.process = subprocess.Popen(
            f"{self.player_command} {color.value.capitalize()}",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            shell=True,
        )
        super().initialize(color, ms_total)
        self.logger.info(self.process.stdout.readline().decode("utf-8"))

    def _get_move(
        self, board: Board, opponent_move: Optional[Loc], ms_left: Optional[int] = None
    ) -> Loc:
        if opponent_move is None:
            opponent_move = Loc.pass_loc()

        if ms_left is None:
            ms_left = -1

        self.process.stdin.write(
            bytes(f"{opponent_move.x} {opponent_move.y} {ms_left}\n", "utf-8")
        )
        self.process.stdin.flush()

        x, y = self.process.stdout.readline().decode("utf-8").strip().split()
        return Loc(int(x), int(y))

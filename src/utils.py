from typing import List

import numpy as np  # type: ignore

from game import Move


def moves_list(bitboard: np.ndarray) -> List[Move]:
    """
    Convert a bitboard of moves into a list of moves.
    """
    return [Move(x, y) for y, x in np.argwhere(bitboard)]

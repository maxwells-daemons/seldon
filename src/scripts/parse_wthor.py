"""
Code for working with the WThor database, available at
http://www.ffothello.org/informatique/la-base-wthor/.
"""
import logging
import os
from glob import glob
from typing import List, NamedTuple, Tuple, Union

import click
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore

from board import Bitboard, Board, GameOutcome, Loc, PlayerColor
from data import serialize_example

DB_HEADER_BYTES = 16
GAME_BYTES = 68
GAME_HEADER_BYTES = 8


class GameState(NamedTuple):
    board: Board
    player: PlayerColor
    move: Loc

    # Format: (board [8x8x2 ndarray of (mine, opp)], move [x, y], value)
    def to_data(self, winner: GameOutcome) -> Tuple[np.ndarray, Tuple[int, int], int]:
        mine, opp = self.board.player_view(self.player)
        board = np.dstack([mine.piecearray, opp.piecearray])

        if winner == GameOutcome.DRAW:
            value = 0
        elif winner.value == self.player.value:
            value = 1
        else:
            value = -1

        return (board, (self.move.x, self.move.y), value)

    def __repr__(self) -> str:
        return f"Next move: {self.player.value} plays {self.move}\n{self.board}"


class GameSummary(NamedTuple):
    real_score: int
    theoretical_score: int
    states: List[GameState]
    outcome: GameOutcome


def parse_move(move_encoding: int) -> Loc:
    x = move_encoding % 10 - 1
    y = move_encoding // 10 - 1
    return Loc(x, y)


def parse_game(game_bytes: bytes) -> GameSummary:
    assert len(game_bytes) == GAME_BYTES

    header_bytes = game_bytes[:GAME_HEADER_BYTES]
    real_score = int(header_bytes[6])
    theoretical_score = int(header_bytes[7])

    move_bytes = game_bytes[GAME_HEADER_BYTES:]
    board = Board.starting_board()
    moves = list(map(parse_move, move_bytes))
    player = PlayerColor.BLACK
    states: List[GameState] = []

    for move in moves:
        if move == Loc.pass_loc():
            break

        states.append(GameState(board, player, move))
        board = board.resolve_move(move, player)

        if board.has_moves(player.opponent):
            player = player.opponent

    return GameSummary(
        real_score=real_score,
        theoretical_score=theoretical_score,
        states=states,
        outcome=board.winning_player,
    )


def parse_db(filename: str) -> List[GameSummary]:
    logging.info(f"Parsing database: {filename}")

    with open(filename, "rb") as f:
        db_bytes = f.read()
    data_bytes = db_bytes[DB_HEADER_BYTES:]

    summaries = []
    for i in range(len(data_bytes) // GAME_BYTES):
        game_bytes = data_bytes[i * GAME_BYTES : (i + 1) * GAME_BYTES]  # noqa
        summaries.append(parse_game(game_bytes))
    return summaries


def make_dataset(
    boards: np.ndarray, moves: np.ndarray, values: np.ndarray
) -> tf.data.Dataset:
    def gen():
        for i in range(boards.shape[0]):
            black_bb = Bitboard.from_piecearray(boards[i, :, :, 0])
            white_bb = Bitboard.from_piecearray(boards[i, :, :, 1])
            board = Board.from_player_view(black_bb, white_bb, PlayerColor.BLACK)
            move = Loc(moves[i, 0], moves[i, 1])
            yield serialize_example(board, move, values[i])

    return tf.data.Dataset.from_generator(gen, output_types=tf.string, output_shapes=())


@click.command()
@click.option("--wthor_glob", default="resources/wthor/game_data/*.wtb")
@click.option("--out_dir", default="resources/wthor/preprocessed/")
def main(wthor_glob: str, out_dir: str):
    logging.basicConfig(level=logging.INFO)
    os.makedirs(out_dir, exist_ok=True)

    db_files = glob(wthor_glob)
    boards: Union[List[np.ndarray], np.ndarray] = []
    moves: Union[List[Tuple[int, int]], np.ndarray] = []
    values: Union[List[int], np.ndarray] = []

    logging.info(f"Reading files: {db_files}")
    logging.info(f"Writing files to: {out_dir}")

    for filename in db_files:
        games = parse_db(filename)

        for game in games:
            data_samples = map(lambda x: x.to_data(game.outcome), game.states)
            new_boards, new_moves, new_values = zip(*data_samples)
            boards.extend(new_boards)
            moves.extend(new_moves)
            values.extend(new_values)

    boards = np.array(boards)
    moves = np.array(moves)
    values = np.array(values)

    legal_indices = np.where(moves[:, 0] != -1)
    boards = boards[legal_indices]
    moves = moves[legal_indices]
    values = values[legal_indices]

    np.save(os.path.join(out_dir, "boards.npy"), boards)
    np.save(os.path.join(out_dir, "moves.npy"), moves)
    np.save(os.path.join(out_dir, "values.npy"), values)

    logging.info("Writing TFRecord")
    ds = make_dataset(boards, moves, values)
    writer = tf.data.experimental.TFRecordWriter(
        os.path.join(out_dir, "wthor.tfrecord")
    )
    writer.write(ds)

    print("Done!")


if __name__ == "__main__":
    main()

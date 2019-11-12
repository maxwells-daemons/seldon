#!/usr/bin/env python

from absl import app, flags  # type: ignore

from game import play_game
from players.external_player import ExternalPlayer  # type: ignore

FLAGS = flags.FLAGS

flags.DEFINE_string("black", None, "Command to run black player.")
flags.DEFINE_string("white", None, "Command to run white player.")
flags.DEFINE_integer(
    "max_time",
    None,
    "Milliseconds to give each player. If unspecified, there is no maximum time.",
)

flags.mark_flags_as_required(["black", "white"])


def main(_):
    black_player = ExternalPlayer(FLAGS.black)
    white_player = ExternalPlayer(FLAGS.white)
    outcome = play_game(black_player, white_player, FLAGS.max_time, show_board=True)
    print(f"Outcome: {outcome}")


if __name__ == "__main__":
    app.run(main)

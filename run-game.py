import argparse
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

from game.main import start_game
from game.play import DEFAULT_TICK_RATE


def build_parser() -> argparse.ArgumentParser:
	"""The command line, which is the display and simulation half of run-ai's."""
	parser = argparse.ArgumentParser(
		prog='run-game',
		description='Play the level yourself, on the same simulation the agents are trained on.',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument('--map', default='maps/level_1.txt', help='level file to play')
	parser.add_argument('--tick-rate', type=int, default=DEFAULT_TICK_RATE, help='simulation ticks per in-game second')
	parser.add_argument('--fps', type=int, default=0, help='target framerate, 0 uses the display refresh rate')
	parser.add_argument('--spawn', type=int, default=0, help='spawn point to start on, 0 is the start and the rest are the checkpoints')
	return parser


def main() -> None:
	args = build_parser().parse_args()
	start_game(map_path=args.map, tick_rate=args.tick_rate, fps=args.fps, spawn=args.spawn)


if __name__ == '__main__':
	main()

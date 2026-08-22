import argparse
import os
from typing import override

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

import torch

from ai.generation import (
	DEFAULT_ACTION_REPEAT, DEFAULT_ELITE_COUNT, DEFAULT_EPISODE_SECONDS, DEFAULT_MUTATION_RATE,
	DEFAULT_MUTATION_STRENGTH, DEFAULT_POPULATION_SIZE, DEFAULT_TICK_RATE, MAX_SPEED, Generation,
)
from ai.population import DEFAULT_HIDDEN_SIZES

DEFAULT_THREADS: int = 4  # More threads than this only adds synchronisation overhead on batches this small


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
	"""Shows the default of every option except the on/off flags, where printing one only misleads."""

	@override
	def _get_help_string(self, action: argparse.Action) -> str | None:
		if isinstance(action, argparse._StoreTrueAction | argparse._StoreFalseAction):
			return action.help
		return super()._get_help_string(action)


def speed_value(text: str) -> float | str:
	"""Parses --speed: a multiplier of real time, or 'max' to run as fast as the framerate allows."""
	if text.lower() == MAX_SPEED:
		return MAX_SPEED
	value = float(text)
	if value <= 0:
		raise argparse.ArgumentTypeError('speed must be positive or "max"')
	return value


def build_parser() -> argparse.ArgumentParser:
	"""The command line, grouped by what each knob actually touches."""
	parser = argparse.ArgumentParser(
		prog='run-ai',
		description='Evolve a population of agents through the level.',
		formatter_class=HelpFormatter,
	)

	evolution = parser.add_argument_group('evolution', 'How a generation is selected and bred')
	evolution.add_argument('--population-size', type=int, default=DEFAULT_POPULATION_SIZE, help='agents per generation, a tick at 300 costs about 1.4x a tick at 100')
	evolution.add_argument('--elite-count', type=int, default=DEFAULT_ELITE_COUNT, help='agents carried over untouched and used as parents')
	evolution.add_argument('--mutation-rate', type=float, default=DEFAULT_MUTATION_RATE, help="probability that a child's weight tensor is mutated at all")
	evolution.add_argument('--mutation-strength', type=float, default=DEFAULT_MUTATION_STRENGTH, help='scale of the noise added to a mutated tensor')
	evolution.add_argument('--sampled', dest='deterministic', action='store_false', default=True, help="sample actions instead of playing the argmax, which turns a generation's scores into a lottery")

	network = parser.add_argument_group('network', 'Shape and placement of the policy')
	network.add_argument('--hidden-sizes', type=int, nargs=3, default=list(DEFAULT_HIDDEN_SIZES), metavar=('N1', 'N2', 'N3'), help='sizes of the three hidden layers, smaller is faster and dumber')
	network.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='where the population runs')
	network.add_argument('--threads', type=int, default=DEFAULT_THREADS, help='torch CPU threads')

	simulation = parser.add_argument_group('simulation', 'The level and the episode played on it')
	simulation.add_argument('--map', default='maps/level_1.txt', help='level file to train on')
	simulation.add_argument('--episode-seconds', type=float, default=DEFAULT_EPISODE_SECONDS, help='in-game time budget per spawn point')
	simulation.add_argument('--tick-rate', type=int, default=DEFAULT_TICK_RATE, help='simulation ticks per in-game second')
	simulation.add_argument('--action-repeat', type=int, default=DEFAULT_ACTION_REPEAT, help='physics ticks a chosen action is held for')
	simulation.add_argument('--checkpoints', action='store_true', help='use the level checkpoints as extra spawn points')

	run = parser.add_argument_group('run', 'Where the run starts and when it stops')
	run.add_argument('--generations', type=int, default=0, help='stop after N generations, 0 runs forever')
	run.add_argument('--seed', type=int, default=None, help='seed python, numpy and torch so a run replays exactly')
	run.add_argument('--load-latest-generation-weights', action='store_true', help='start from the most recent weight file')

	display = parser.add_argument_group('display', 'Only meaningful together with --show-window')
	display.add_argument('--show-window', action='store_true', help='render the run instead of training headless')
	display.add_argument('--speed', type=speed_value, default=1.0, metavar='S', help="simulation speed multiplier, or 'max' to run as fast as the framerate survives")
	display.add_argument('--fps', type=int, default=0, help='target framerate, 0 uses the display refresh rate')

	return parser


def main() -> None:
	args = build_parser().parse_args()

	torch.set_num_threads(max(1, min(args.threads, os.cpu_count() or 1)))
	if not args.show_window:
		os.environ['SDL_VIDEODRIVER'] = 'dummy'

	generation = Generation(
		args.population_size,
		elite_count=args.elite_count,
		mutation_rate=args.mutation_rate,
		mutation_strength=args.mutation_strength,
		deterministic_actions=args.deterministic,
		hidden_sizes=tuple(args.hidden_sizes),
		device=args.device,
		map_path=args.map,
		episode_seconds=args.episode_seconds,
		tick_rate=args.tick_rate,
		action_repeat=args.action_repeat,
		use_checkpoints=args.checkpoints,
		seed=args.seed,
		load_latest_generation_weights=args.load_latest_generation_weights,
		show_window=args.show_window,
		speed=args.speed,
		fps=args.fps,
	)

	print(f"--- Generation {generation.generation}, {args.population_size} agents on {generation.population.device}, "
		  f"mutation rate: {generation.mutation_rate:.3f} - strength: {generation.mutation_strength:.3f} ---")

	try:
		while True:
			generation.play_agents()
			rewards = generation.rewards
			print(f"--- Generation {generation.generation} ---")
			print(f"  Best: {rewards.max():.2f} | Avg: {rewards.mean():.2f} | Worst: {rewards.min():.2f}")
			print(f"  Speed: {generation.last_speed:,.0f} ticks/s ({generation.last_speed * args.population_size:,.0f} agent-steps/s)")
			generation.evolve_generation()
			if args.generations and generation.generation > args.generations:
				break
	except (KeyboardInterrupt, SystemExit):
		print("Stopped.")
	finally:
		generation.quit()


if __name__ == '__main__':
	main()

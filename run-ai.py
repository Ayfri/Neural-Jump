import argparse
import os
import time
from typing import override

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

import torch

from ai.generation import (
	DEFAULT_ACTION_REPEAT, DEFAULT_ELITE_COUNT, DEFAULT_ENV_COUNT, DEFAULT_EPISODE_SECONDS,
	DEFAULT_MUTATION_RATE, DEFAULT_MUTATION_STRENGTH, DEFAULT_POPULATION_SIZE, DEFAULT_SPAWN_MODE,
	DEFAULT_SPAWN_SPACING, DEFAULT_TICK_RATE, DEFAULT_TRAINER, MAX_SPEED, SPAWN_MODES, TRAINERS, Generation,
)
from ai.population import DEFAULT_HIDDEN_SIZES
from ai.ppo import (
	DEFAULT_CLIP_RANGE, DEFAULT_ENTROPY_COEF, DEFAULT_EPOCHS, DEFAULT_GAE_LAMBDA, DEFAULT_GAMMA,
	DEFAULT_LEARNING_RATE, DEFAULT_MINIBATCHES, DEFAULT_ROLLOUT_STEPS, DEFAULT_TARGET_KL,
)

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

	parser.add_argument('--trainer', choices=TRAINERS, default=DEFAULT_TRAINER, help='ppo trains one policy on every environment at once, ga selects and breeds a population')
	parser.add_argument('--population-size', type=int, default=None, metavar='N', help=f'agents under ga, parallel environments under ppo (defaults to {DEFAULT_POPULATION_SIZE} and {DEFAULT_ENV_COUNT})')

	ppo = parser.add_argument_group('ppo', 'The policy gradient, ignored under --trainer ga')
	ppo.add_argument('--rollout-steps', type=int, default=DEFAULT_ROLLOUT_STEPS, help='decisions per environment between two updates')
	ppo.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
	ppo.add_argument('--gamma', type=float, default=DEFAULT_GAMMA, help='discount per decision, not per tick')
	ppo.add_argument('--gae-lambda', type=float, default=DEFAULT_GAE_LAMBDA, help='bias against variance in the advantage estimate')
	ppo.add_argument('--clip-range', type=float, default=DEFAULT_CLIP_RANGE, help='how far one update may move the policy')
	ppo.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='passes over each rollout, cut short by --target-kl')
	ppo.add_argument('--minibatches', type=int, default=DEFAULT_MINIBATCHES, help='minibatches each pass is split into')
	ppo.add_argument('--entropy-coef', type=float, default=DEFAULT_ENTROPY_COEF, help='exploration bonus, annealed down over the run')
	ppo.add_argument('--target-kl', type=float, default=DEFAULT_TARGET_KL, help='divergence from the rollout that stops the extra passes')
	ppo.add_argument('--spawn', choices=SPAWN_MODES, default=DEFAULT_SPAWN_MODE, dest='spawn_mode', help='curriculum starts at the rung nearest the flag and works backwards, uniform draws any, start always uses the level spawn')
	ppo.add_argument('--spawn-spacing', type=int, default=DEFAULT_SPAWN_SPACING, metavar='T', help='tiles between two rungs of the curriculum ladder, which is built from the level floor')

	evolution = parser.add_argument_group('evolution', 'How a generation is selected and bred, ignored under --trainer ppo')
	evolution.add_argument('--elite-count', type=int, default=DEFAULT_ELITE_COUNT, help='agents carried over untouched and used as parents')
	evolution.add_argument('--mutation-rate', type=float, default=DEFAULT_MUTATION_RATE, help="probability that a child's weight tensor is mutated at all")
	evolution.add_argument('--mutation-strength', type=float, default=DEFAULT_MUTATION_STRENGTH, help='scale of the noise added to a mutated tensor')

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
	if args.population_size is None:
		args.population_size = DEFAULT_ENV_COUNT if args.trainer == 'ppo' else DEFAULT_POPULATION_SIZE

	torch.set_num_threads(max(1, min(args.threads, os.cpu_count() or 1)))
	if not args.show_window:
		os.environ['SDL_VIDEODRIVER'] = 'dummy'

	generation = Generation(
		args.population_size,
		trainer=args.trainer,
		elite_count=args.elite_count,
		mutation_rate=args.mutation_rate,
		mutation_strength=args.mutation_strength,
		hidden_sizes=tuple(args.hidden_sizes),
		device=args.device,
		map_path=args.map,
		episode_seconds=args.episode_seconds,
		tick_rate=args.tick_rate,
		action_repeat=args.action_repeat,
		use_checkpoints=args.checkpoints,
		spawn_mode=args.spawn_mode,
		spawn_spacing=args.spawn_spacing,
		rollout_steps=args.rollout_steps,
		learning_rate=args.learning_rate,
		gamma=args.gamma,
		gae_lambda=args.gae_lambda,
		clip_range=args.clip_range,
		epochs=args.epochs,
		minibatches=args.minibatches,
		entropy_coef=args.entropy_coef,
		target_kl=args.target_kl,
		seed=args.seed,
		load_latest_generation_weights=args.load_latest_generation_weights,
		show_window=args.show_window,
		speed=args.speed,
		fps=args.fps,
	)

	if args.trainer == 'ppo':
		print(f"--- PPO, {args.population_size} environments on {generation.population.device}, "
			  f"{args.rollout_steps * args.population_size:,} transitions per update ---")
	else:
		print(f"--- Generation {generation.generation}, {args.population_size} agents on {generation.population.device}, "
			  f"mutation rate: {generation.mutation_rate:.3f} - strength: {generation.mutation_strength:.3f} ---")

	ticks = generation.total_ticks
	started = time.perf_counter()
	try:
		while True:
			generation.play_agents()
			print(f"--- {'Update' if args.trainer == 'ppo' else 'Generation'} {generation.generation} ---")
			if args.trainer == 'ppo':
				generation.evolve_generation()
				stats = generation.last_stats
				record = f"{generation.best_fitness_ever:.2f}" if generation.best_fitness_ever > -float('inf') else '-'
				print(f"  Episodes: {stats['episodes']:,.0f} | Wins: {stats['wins']:,.0f} | Return: {stats['mean_return']:.2f} | Best: {record}"
					  + (f" | Fastest: {generation.best_time_ever:.2f}s" if generation.best_time_ever else ''))
				front = f"{stats['front_cleared']:,.0f}/{stats['front_episodes']:,.0f}"
				print(f"  Front: rung {generation.curriculum.front} of {len(generation.curriculum.spawns) - 1}, cleared {front} | Entropy: {stats['entropy']:.3f}")
				print(f"  Policy: {stats['policy_loss']:+.4f} | Value: {stats['value_loss']:.4f} | KL: {stats['approx_kl']:.4f} | Clipped: {stats['clip_fraction']:.1%} | Scale: {stats['return_scale']:.1f}")
			else:
				rewards = generation.rewards
				print(f"  Best: {rewards.max():.2f} | Avg: {rewards.mean():.2f} | Worst: {rewards.min():.2f}")
				generation.evolve_generation()
			# Two numbers under PPO: what the simulation runs at, and what the wall clock sees once the
			# update in front of it is paid for, which is most of a cycle at these batch sizes
			elapsed = time.perf_counter() - started
			overall = (generation.total_ticks - ticks) / elapsed if elapsed else 0.0
			ticks, started = generation.total_ticks, time.perf_counter()
			print(f"  Speed: {generation.last_speed:,.0f} ticks/s ({generation.last_speed * args.population_size:,.0f} agent-steps/s)"
				  + (f" | {overall:,.0f} ticks/s counting the update" if args.trainer == 'ppo' else ''))
			if args.generations and generation.generation > args.generations:
				break
	except (KeyboardInterrupt, SystemExit):
		print("Stopped.")
	finally:
		generation.quit()


if __name__ == '__main__':
	main()

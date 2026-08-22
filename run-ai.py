import argparse
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

import torch

from ai.generation import DEFAULT_ELITE_COUNT, DEFAULT_EPISODE_SECONDS, DEFAULT_TICK_RATE, MAX_SPEED, Generation
from ai.population import DEFAULT_HIDDEN_SIZES

DEFAULT_THREADS = 4  # More threads than this only adds synchronisation overhead on batches this small


def speed_value(text: str) -> float | str:
	"""Parses --speed: a multiplier of real time, or 'max' to run as fast as the framerate allows."""
	if text.lower() == MAX_SPEED:
		return MAX_SPEED
	value = float(text)
	if value <= 0:
		raise argparse.ArgumentTypeError('speed must be positive or "max"')
	return value


def main() -> None:
	argparser = argparse.ArgumentParser()
	argparser.add_argument("--population-size", type=int, default=100)
	argparser.add_argument("--elite-count", type=int, default=DEFAULT_ELITE_COUNT)
	argparser.add_argument("--mutation-rate", type=float, default=0.8)
	argparser.add_argument("--mutation-strength", type=float, default=0.015)
	argparser.add_argument("--hidden-sizes", type=int, nargs=3, default=list(DEFAULT_HIDDEN_SIZES), help="Sizes of the three shared hidden layers, smaller is faster and dumber")
	argparser.add_argument("--device", default='auto', choices=['auto', 'cpu', 'cuda'])
	argparser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="Torch CPU threads")
	argparser.add_argument("--tick-rate", type=int, default=DEFAULT_TICK_RATE, help="Simulation ticks per in-game second (also caps FPS when rendering)")
	argparser.add_argument("--episode-seconds", type=float, default=DEFAULT_EPISODE_SECONDS)
	argparser.add_argument("--generations", type=int, default=0, help="Stop after N generations, 0 runs forever")
	argparser.add_argument("--speed", type=speed_value, default=1.0, help="Simulation speed multiplier, or 'max' to run as fast as the target framerate survives (only with --show-window)")
	argparser.add_argument("--fps", type=int, default=0, help="Target framerate, 0 uses the display refresh rate (only with --show-window)")
	argparser.add_argument("--map", default='maps/level_1.txt')
	argparser.add_argument("--load-latest-generation-weights", action="store_true")
	argparser.add_argument("--show-window", action="store_true")
	argparser.add_argument("--checkpoints", action="store_true", help="Use checkpoints as spawn points")
	argparser.add_argument("--no-use-a2c", dest="use_a2c", action="store_false", default=True, help="Disable A2C and use only genetic algorithm")
	args = argparser.parse_args()

	torch.set_num_threads(max(1, min(args.threads, os.cpu_count() or 1)))
	if not args.show_window:
		os.environ['SDL_VIDEODRIVER'] = 'dummy'

	generation = Generation(
		args.population_size,
		elite_count=args.elite_count,
		mutation_rate=args.mutation_rate,
		mutation_strength=args.mutation_strength,
		load_latest_generation_weights=args.load_latest_generation_weights,
		show_window=args.show_window,
		use_checkpoints=args.checkpoints,
		use_a2c_learning=args.use_a2c,
		hidden_sizes=tuple(args.hidden_sizes),
		device=args.device,
		tick_rate=args.tick_rate,
		episode_seconds=args.episode_seconds,
		speed=args.speed,
		fps=args.fps,
		map_path=args.map,
	)

	learning_method = "A2C + Genetic Algorithm" if args.use_a2c else "Genetic Algorithm only"
	print(f"--- Training with: {learning_method} ---")
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

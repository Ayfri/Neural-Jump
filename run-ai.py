import argparse
import os
from ai.generation import Generation

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


def main() -> None:
	argparser = argparse.ArgumentParser()
	argparser.add_argument("--population_size", type=int, default=30)
	argparser.add_argument("--mutation_rate", type=float, default=0.06)
	argparser.add_argument("--mutation_strength", type=float, default=0.025)
	argparser.add_argument("--load_latest_generation_weights", action="store_true")
	argparser.add_argument("--show_window", action="store_true")
	argparser.add_argument("--checkpoints", action="store_true", help="Use checkpoints as spawn points")
	args = argparser.parse_args()

	generation = Generation(
		args.population_size,
		mutation_rate=args.mutation_rate,
		mutation_strength=args.mutation_strength,
		load_latest_generation_weights=args.load_latest_generation_weights,
		show_window=args.show_window,
		use_checkpoints=args.checkpoints
	)

	print(f"--- Generation {generation.generation}, mutation rate: {generation.mutation_rate} - mutation strength: {generation.mutation_strength} ---")
	while True:
		generation.play_agents()
		best_agent = generation.get_best_agent()
		print(f"--- Generation {generation.generation + 1} - Best agent reward: {best_agent.current_reward} - Mutation rate: {generation.mutation_rate} - Mutation strength: {generation.mutation_strength} ---")
		generation.evolve_generation()


if __name__ == '__main__':
	main()

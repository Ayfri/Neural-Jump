import argparse
import os
from ai.generation import Generation

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


def main() -> None:
	argparser = argparse.ArgumentParser()
	argparser.add_argument("--population-size", type=int, default=60)
	argparser.add_argument("--mutation-rate", type=float, default=0.3)
	argparser.add_argument("--mutation-strength", type=float, default=0.02)
	argparser.add_argument("--load-latest-generation-weights", action="store_true")
	argparser.add_argument("--show-window", action="store_true")
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

	print(f"--- Generation {generation.generation}, mutation rate: {generation.mutation_rate:.3f} - mutation strength: {generation.mutation_strength:.3f} ---")
	while True:
		generation.play_agents()
		best_agent = generation.get_best_agent()
		avg_reward = sum(agent.current_reward for agent in generation.agents) / len(generation.agents)
		worst_reward = min(agent.current_reward for agent in generation.agents)
		print(f"--- Generation {generation.generation + 1} ---")
		print(f"  Best reward: {best_agent.current_reward:.2f} | Avg: {avg_reward:.2f} | Worst: {worst_reward:.2f}")
		print(f"  Mutation - rate: {generation.mutation_rate:.3f}, strength: {generation.mutation_strength:.3f}")
		generation.evolve_generation()


if __name__ == '__main__':
	main()

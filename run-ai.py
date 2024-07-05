import os
from ai.generation import Generation

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


def main() -> None:
	generation = Generation(30,  mutation_rate=0.06, mutation_strength=0.025, load_latest_generation_weights=True)
	print(f"--- Generation {generation.generation}, mutation rate: {generation.mutation_rate} - mutation strength: {generation.mutation_strength} ---")
	while True:
		generation.play_agents()
		best_agent = generation.get_best_agent()
		print(f"--- Generation {generation.generation + 1} - Best agent reward: {best_agent.current_reward} - Mutation rate: {generation.mutation_rate} - Mutation strength: {generation.mutation_strength} ---")
		generation.evolve_generation()


if __name__ == '__main__':
	main()

# main.py

from ai.generation import Generation

def main() -> None:
	generation = Generation(5)
	print(f"--- Generation {generation.generation}, mutation rate: {generation.mutation_rate} ---")
	while True:
		generation.play_agents()
		best_agent = generation.get_best_agent()
		print(f"--- Generation {generation.generation + 1} - Best agent reward: {best_agent.current_reward} - Mutation rate: {generation.mutation_rate} ---")
		generation.evolve_generation()


if __name__ == '__main__':
	main()

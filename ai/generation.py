import os

import numpy as np
from ai.agent import Agent


class Generation:
	def __init__(self, population_size: int, elite_fraction: float = 0.1) -> None:
		self.population_size = population_size
		self.elite_fraction = elite_fraction
		self.tick_rate = 2000
		self.show_window = True
		self.running_time = 2  # Time in seconds to run the game
		self.generation = 1
		self.mutation_rate = 0.07
		self.agents = [Agent(self.tick_rate, self.show_window, self.running_time, generation = self) for _ in range(population_size)]

	def evolve_generation(self) -> None:
		"""
		Evolves agent generation by generating new weights.
		"""
		# Sort agents in descending order of reward and select the best agent
		self.agents.sort(key=lambda agent: agent.current_reward, reverse=True)
		num_elites = max(1, int(self.elite_fraction * self.population_size))
		elites = self.agents[:num_elites]

		# Generate new agents with mutations based on the best agent
		new_agents = elites.copy()
		while len(new_agents) < self.population_size:
			parent1, parent2 = np.random.choice(elites, 2)
			child_weights = self.crossover(parent1.weights, parent2.weights)
			child_weights = self.mutate(child_weights)
			new_agents.append(Agent(self.tick_rate, self.show_window, self.running_time, weights=child_weights, generation=self))

		self.agents = new_agents
		self.generation += 1

		os.makedirs("weights", exist_ok=True)

		with open(f"weights/generation_{self.generation}.npy", "wb") as f:
			np.save(f, elites[0].weights)

	def crossover(self, weights1: np.ndarray, weights2: np.ndarray) -> np.ndarray:
		"""
		Performs cross-over between two sets of weights.
		"""
		mask = np.random.rand(*weights1.shape) > 0.5
		return np.where(mask, weights1, weights2)

	def mutate(self, weights: np.ndarray) -> np.ndarray:
		"""
		Mutates weights with Gaussian noise.
		"""
		return weights + np.random.randn(*weights.shape) * self.mutation_rate

	def get_best_agent(self) -> Agent:
		"""
		Retourne l'agent avec le meilleur reward dans la génération actuelle.
		"""
		return max(self.agents, key=lambda agent: agent.current_reward)

	def play_agents(self) -> None:
		"""
		Play games with all agents in the generation.
		"""
		for index, agent in enumerate(self.agents):
			print(f"Playing game with agent {index + 1}, generation {self.generation}.")
			agent.play_game()
			print(f"Finished game with agent {index + 1}, reward: {agent.current_reward}")

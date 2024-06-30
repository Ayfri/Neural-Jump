import os
import re

import numpy as np
import torch
from torch import nn

from ai.agent import Agent


class Generation:
	def __init__(
		self,
		population_size: int,
		elite_fraction: float = 0.1,
		mutation_rate: float = 0.01,
		mutation_strength: float = 0.1,
		load_latest_generation_weights: bool = False
	) -> None:
		self.population_size = population_size
		self.elite_fraction = elite_fraction
		self.mutation_strength = mutation_strength
		self.mutation_rate = mutation_rate
		self.tick_rate = 2000
		self.show_window = True
		self.running_time = 6  # Time in seconds to run the game
		self.generation = 1
		self.agents = [Agent(self.tick_rate, self.show_window, self.running_time, generation=self) for _ in range(population_size)]

		if load_latest_generation_weights:
			self.load_latest_generation_weights()

	def evolve_generation(self) -> None:
		"""
		Evolves agent generation by generating new weights.
		"""
		# Sort agents in descending order of reward and select the best agent
		self.agents.sort(key=lambda agent: agent.current_reward, reverse=True)
		num_elites = max(1, int(self.elite_fraction * self.population_size))
		elites = self.agents[:num_elites]

		# Ensure we have at least one elite to be preserved without mutation
		new_agents = elites[:]

		# Generate new agents with crossover and mutations based on the elites
		while len(new_agents) < self.population_size:
			parent1, parent2 = np.random.choice(elites, 2)
			child_weights = self.crossover(parent1.model.state_dict(), parent2.model.state_dict())
			new_agent = Agent(self.tick_rate, self.show_window, self.running_time, generation=self)
			new_agent.model.load_state_dict(child_weights)
			self.mutate(new_agent.model)
			new_agents.append(new_agent)

		self.agents = new_agents[:self.population_size]
		self.generation += 1

		os.makedirs("weights", exist_ok=True)

		torch.save(elites[0].model.state_dict(), f"weights/generation_{self.generation}.pth")

	def crossover(self, parent1_weights: dict, parent2_weights: dict) -> dict:
		"""
		Cross over the weights of two parents to create a child.
		"""
		child_weights = {}
		for key in parent1_weights:
			if key in parent2_weights:
				child_weights[key] = (parent1_weights[key] + parent2_weights[key]) / 2
			else:
				child_weights[key] = parent1_weights[key]
		return child_weights

	def mutate(self, model: nn.Module) -> None:
		"""
		Mutate the weights of an agent.
		"""
		for param in model.parameters():
			if param.requires_grad and np.random.rand() < self.mutation_rate:
				param.data += torch.randn_like(param) * self.mutation_strength

	def get_best_agent(self) -> Agent:
		"""
		Retourne l'agent avec le meilleur reward dans la génération actuelle.
		"""
		return max(self.agents, key=lambda agent: agent.current_reward)

	def load_latest_generation_weights(self) -> None:
		"""
		Load the weights of the latest generation from the weights directory.
		"""
		try:
			latest_generation = max([
				int(filename.split('_')[1].split('.')[0])
				for filename in os.listdir("weights")
				if re.match(r"generation_\d+\.pth", filename)
			])
			weights_path = f"weights/generation_{latest_generation}.pth"
			with open(weights_path, 'rb') as file:
				weights = torch.load(file)
				for agent in self.agents:
					agent.model.load_state_dict(weights)

			self.generation = latest_generation
			print(f"Loaded weights for generation {latest_generation}.")
		except:
			print("No weights found, starting with random weights.")

	def play_agents(self) -> None:
		"""
		Play games with all agents in the generation.
		"""
		for index, agent in enumerate(self.agents):
			print(f"Playing game with agent {index + 1}, generation {self.generation}.")
			agent.play_game()
			print(f"Finished game with agent {index + 1}, reward: {agent.current_reward}")

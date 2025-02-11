import os
import re
import traceback
from copy import deepcopy

import numpy as np
import torch
from pygame import Surface
from torch import nn

from ai.agent import Agent
from concurrent.futures import ProcessPoolExecutor, as_completed

from game.game import Game
from game.settings import BLACK


class Generation:
	def __init__(
		self,
		population_size: int,
		elite_count: int = 2,
		mutation_rate: float = 0.01,
		mutation_strength: float = 0.1,
		load_latest_generation_weights: bool = False,
		show_window: bool = True,
		use_checkpoints: bool = False
	) -> None:
		self.population_size = population_size
		self.elite_count = elite_count
		self.mutation_strength = mutation_strength
		self.mutation_rate = mutation_rate
		self.show_window = show_window
		self.tick_rate = 2000
		self.running_time = 20  # Time in seconds to run the game
		self.generation = 1
		self.use_checkpoints = use_checkpoints
		self.agents = [Agent(self.tick_rate, self.show_window, self.running_time, generation=self) for _ in range(population_size)]

		if load_latest_generation_weights:
			self.load_latest_generation_weights()

	def evolve_generation(self) -> None:
		"""
		Evolves agent generation by generating new weights.
		"""
		# Sort agents in descending order of reward and select the best agent
		self.agents.sort(key=lambda agent: agent.current_reward, reverse=True)
		elites = self.agents[:self.elite_count]
		print(f"Selected {len(elites)} elites: {[agent.current_reward for agent in elites]}")

		# Ensure we have at least one elite to be preserved without mutation
		new_agents = elites[:]

		# Generate new agents with crossover and mutations based on the elites
		random_generator = np.random.default_rng()
		while len(new_agents) < self.population_size:
			parent1: Agent
			parent2: Agent
			parent1, parent2 = random_generator.choice(elites, 2)
			child_weights = self.crossover(parent1.model.state_dict().copy(), parent2.model.state_dict().copy())
			new_agent = Agent(self.tick_rate, self.show_window, self.running_time, generation=self)
			new_agent.model.load_state_dict(child_weights)
			self.mutate(new_agent.model)
			new_agents += [new_agent]

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
		Returns the agent with the best reward in the current generation.
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
			self.evolve_generation()
		except:
			print("No weights found, starting with random weights.")

	def play_agents(self) -> None:
		"""
		Play games with all agents in the generation.
		If use_checkpoints is True, each agent will be tested from each checkpoint and their rewards will be summed.
		"""
		game = Game(self.population_size, self.tick_rate, self.show_window, has_playable_player=True)
		game.use_checkpoints = self.use_checkpoints  # Activer l'utilisation des checkpoints
		game.init()

		# Get all spawn points (checkpoints + initial spawn)
		spawn_points = [(game.level.spawn_point[0], game.level.spawn_point[1])]
		if self.use_checkpoints and game.level.checkpoints:
			spawn_points.extend(game.level.checkpoints)

		print(f"Testing agents on {len(spawn_points)} spawn points (1 initial + {len(spawn_points)-1} checkpoints)")

		# Initialize agents with zero rewards
		for agent in self.agents:
			agent.current_reward = 0

		# For each spawn point, create a copy of each agent and test it
		for spawn_x, spawn_y in spawn_points:
			# Reset game state for this spawn point
			for i, agent in enumerate(self.agents):
				agent.player = game.players[i]
				agent.player.rect.x = spawn_x
				agent.player.rect.y = spawn_y
				agent.player.dead = False
				agent.player.win = False
				agent.player.finished_reward = None
				agent.player.change_x = 0
				agent.player.change_y = 0
				agent.player.revive()  # Réinitialise l'état visuel du joueur

			# Time limit for the game in seconds
			time_limit = self.running_time
			tick = 0

			while not all([player.win or player.dead for player in game.players]) and tick / 1000 < time_limit:
				for agent in self.agents:
					if not agent.player.dead and not agent.player.win:
						# Get the surrounding grid
						grid = agent.player.get_surrounding_tiles()

						# Calculate the move using the agent
						direction = agent.calculate_move(grid)

						# Execute the move
						agent.player.execute_move(direction)

				game.handle_inputs()
				game.update()
				tick += game.clock.get_time()

				if game.display_window:
					best_agent = self.get_best_agent()
					best_player = game.players[best_agent.current_index]

					elapsed_time = tick / 1000
					game.draw_text(
						f"Best Agent: {best_agent.current_index + 1}/{self.population_size}, Generation: {self.generation}",
						10,
						10,
						font_size=24,
						color=BLACK
					)

					game.draw_text(f"FPS: {1000 / (game.clock.get_time() or 1):.1f}", 10, 70, font_size=24, color=BLACK)
					game.draw_text(f"Time: {elapsed_time:.2f}/{time_limit}", 10, 100, font_size=24, color=BLACK)
					game.draw_text(f"Player: X: {best_player.rect.x}, Y: {best_player.rect.y}", 10, 130, font_size=24, color=BLACK)
					game.draw_text(f"Total Reward: {best_agent.current_reward:.2f}", 10, 160, font_size=24, color=BLACK)
					game.draw_text(f"Checkpoint: {spawn_points.index((spawn_x, spawn_y)) + 1}/{len(spawn_points)}", 10, 190, font_size=24, color=BLACK)

					game.draw()

			# Calculate and add rewards for this checkpoint run
			for agent in self.agents:
				agent.current_reward += agent.calculate_reward()

		game.quit()

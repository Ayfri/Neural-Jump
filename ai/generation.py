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

import pygame


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
		self.tick_rate = 90
		self.generation = 1
		self.use_checkpoints = use_checkpoints
		self.agents = [Agent(self.tick_rate, self.show_window, generation=self) for _ in range(population_size)]
		self.should_skip_checkpoint = False
		self.manual_stop = False
		self.agent_positions = {}
		self.last_position_check = 0
		self.best_fitness_ever = 0

		if load_latest_generation_weights:
			self.load_latest_generation_weights()

	def evolve_generation(self) -> None:
		"""
		Evolves agent generation by generating new weights.
		Adapts mutation parameters based on progress.
		"""
		# Sort agents in descending order of reward and select the best agent
		self.agents.sort(key=lambda agent: agent.current_reward, reverse=True)
		elites = self.agents[:self.elite_count]
		best_reward = elites[0].current_reward
		print(f"Selected {len(elites)} elites: {[agent.current_reward for agent in elites]}")

		# Adapt mutation parameters based on progress
		# If agents are consistently reaching the flag (reward > 100), focus on speed optimization
		if best_reward > 100:
			# Reduce mutation rate and strength to fine-tune for speed
			self.mutation_rate = max(0.001, self.mutation_rate * 0.95)
			self.mutation_strength = max(0.001, self.mutation_strength * 0.95)
			print(f"Speed optimization mode - Reduced mutation: rate={self.mutation_rate:.4f}, strength={self.mutation_strength:.4f}")
		elif best_reward < self.best_fitness_ever * 0.8:
			# Performance dropped significantly, increase exploration
			self.mutation_rate = min(0.1, self.mutation_rate * 1.05)
			self.mutation_strength = min(0.05, self.mutation_strength * 1.05)
			print(f"Exploration mode - Increased mutation: rate={self.mutation_rate:.4f}, strength={self.mutation_strength:.4f}")

		# Ensure we have at least one elite to be preserved without mutation
		new_agents = elites[:]

		# Generate new agents with crossover and mutations based on the elites
		random_generator = np.random.default_rng()
		while len(new_agents) < self.population_size:
			parent1: Agent
			parent2: Agent
			parent1, parent2 = random_generator.choice(elites, 2)
			child_weights = self.crossover(parent1.model.state_dict().copy(), parent2.model.state_dict().copy())
			new_agent = Agent(self.tick_rate, self.show_window, generation=self)
			new_agent.model.load_state_dict(child_weights)
			self.mutate(new_agent.model)
			new_agents += [new_agent]

		self.agents = new_agents[:self.population_size]
		self.generation += 1

		# Reset position history for the new generation
		self.agent_positions = {}
		self.last_position_check = 0
		self.manual_stop = False

		os.makedirs("weights", exist_ok=True)
		torch.save({
			'weights': elites[0].model.state_dict(),
			'best_fitness': self.best_fitness_ever,
			'mutation_rate': self.mutation_rate,
			'mutation_strength': self.mutation_strength
		}, f"weights/generation_{self.generation}.pth")

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
				weights_data = torch.load(file)
				if isinstance(weights_data, dict) and 'weights' in weights_data:
					# New format with additional metadata
					for agent in self.agents:
						agent.model.load_state_dict(weights_data['weights'])
					self.best_fitness_ever = weights_data.get('best_fitness', 0)
					
					# Load mutation parameters if available
					if 'mutation_rate' in weights_data:
						self.mutation_rate = weights_data['mutation_rate']
					if 'mutation_strength' in weights_data:
						self.mutation_strength = weights_data['mutation_strength']
				else:
					# Old format, weights_data is the state_dict
					for agent in self.agents:
						agent.model.load_state_dict(weights_data)

			self.generation = latest_generation
			print(f"Loaded weights for generation {latest_generation}.")
			print(f"Mutation parameters: rate={self.mutation_rate:.4f}, strength={self.mutation_strength:.4f}")
			self.evolve_generation()
		except:
			print("No weights found, starting with random weights.")

	def skip_checkpoint(self) -> None:
		"""Skip to the next checkpoint."""
		self.should_skip_checkpoint = True

	def check_agent_positions(self, tick: int) -> None:
		"""
		Check if agents are stuck or moving backwards.
		- Stuck: Same X position for 3 seconds
		- Moving backwards: X position lower than 6 seconds ago
		"""
		current_time = tick / 1000

		# Only check every second
		if current_time - self.last_position_check < 1:
			return

		self.last_position_check = current_time

		for agent in self.agents:
			# Skip dead or winning agents
			if agent.player.dead or agent.player.win:
				continue

			current_x = agent.player.rect.x
			agent_key = agent.current_index

			# Initialize position history
			if agent_key not in self.agent_positions:
				self.agent_positions[agent_key] = []

			# Add current position
			self.agent_positions[agent_key].append((current_time, current_x))

			# Remove positions older than 6 seconds
			self.agent_positions[agent_key] = [
				pos for pos in self.agent_positions[agent_key]
				if pos[0] >= current_time - 6
			]

			positions = self.agent_positions[agent_key]

			# Check if stuck (same position for 3 seconds)
			if len(positions) >= 3:
				recent_positions = [pos[1] for pos in positions[-3:]]
				if len(set(recent_positions)) == 1:  # All positions are the same
					agent.player.set_dead()
					continue

			# Check if moving backwards (X position decreased over 6 seconds)
			if len(positions) >= 6:
				x_6_seconds_ago = positions[0][1]
				if current_x < x_6_seconds_ago:
					agent.player.set_dead()

	def play_agents(self) -> None:
		"""
		Play games with all agents in the generation.
		If use_checkpoints is True, each agent will be tested from each checkpoint and their rewards will be summed.
		"""
		game = Game(self.population_size, self.tick_rate, self.show_window, has_playable_player=False)
		game.use_checkpoints = self.use_checkpoints
		game.init()

		# Add skip checkpoint key action
		game.add_key_action(pygame.K_g, self.skip_checkpoint, "Skip Checkpoint")
		game.add_key_action(pygame.K_s, lambda: setattr(self, 'manual_stop', True), "Stop Generation")

		# Get all spawn points (checkpoints + initial spawn)
		spawn_points = [(game.level.spawn_point[0], game.level.spawn_point[1])]
		if self.use_checkpoints and game.level.checkpoints:
			spawn_points.extend(game.level.checkpoints)

		print(f"Testing agents on {len(spawn_points)} spawn points (1 initial + {len(spawn_points)-1} checkpoints)")

		# Initialize agents with zero rewards
		for agent in self.agents:
			agent.current_reward = 0

		# For each spawn point, test all agents
		for checkpoint_idx, (spawn_x, spawn_y) in enumerate(spawn_points):
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
				agent.player.revive()

			tick = 0
			max_ticks = 30 * self.tick_rate  # 30 seconds time limit

			# Run until all agents die/win, skip checkpoint, manual stop, or timeout
			while (tick < max_ticks and
				   not self.should_skip_checkpoint and
				   not self.manual_stop and
				   not any(player.win for player in game.players) and
				   not all(player.dead for player in game.players)):
				
				# Update living agents
				for agent in self.agents:
					if not agent.player.dead and not agent.player.win:
						grid = agent.player.get_surrounding_tiles()
						direction = agent.calculate_move(grid)
						agent.player.execute_move(direction)

				game.handle_inputs()
				game.update(tick)
				tick += 1

				# Check for stuck agents every second
				self.check_agent_positions(tick)

				if game.display_window:
					best_agent = self.get_best_agent()
					best_player = game.players[best_agent.current_index]
					elapsed_time = tick / self.tick_rate
					living_agents = sum(1 for agent in self.agents if not agent.player.dead and not agent.player.win)
					
					# Calculate current fitness
					current_fitness = max(
						(agent.player.finished_reward if agent.player.finished_reward is not None else agent.player.rect.x / 10)
						for agent in self.agents
					) if self.agents else 0

					# Draw all info
					game.draw_text(f"Best Agent: {best_agent.current_index + 1}/{self.population_size}, Generation: {self.generation}", 10, 10, font_size=24, color=BLACK)
					game.draw_text(f"FPS: {1000 / (game.clock.get_time() or 1):.1f}", 10, 70, font_size=24, color=BLACK)
					game.draw_text(f"Time: {elapsed_time:.2f}s", 10, 100, font_size=24, color=BLACK)
					game.draw_text(f"Player: X: {best_player.rect.x}, Y: {best_player.rect.y}", 10, 130, font_size=24, color=BLACK)
					game.draw_text(f"Checkpoint: {checkpoint_idx + 1}/{len(spawn_points)}", 10, 160, font_size=24, color=BLACK)
					game.draw_text(f"Living agents: {living_agents}/{self.population_size}", 10, 190, font_size=24, color=BLACK)
					game.draw_text(f"Best Fitness: {self.best_fitness_ever:.2f}", 10, 220, font_size=24, color=BLACK)
					game.draw_text(f"Current Fitness: {current_fitness:.2f}", 10, 250, font_size=24, color=BLACK)
					game.draw_text("Press S to skip generation", 10, 280, font_size=24, color=BLACK)

					game.draw()

			# Calculate and add rewards for this checkpoint run
			for agent in self.agents:
				agent.current_reward += agent.calculate_reward()

			# Reset skip checkpoint flag
			self.should_skip_checkpoint = False

		# Update best fitness ever
		if self.agents:
			current_max_reward = max(agent.current_reward for agent in self.agents)
			self.best_fitness_ever = max(self.best_fitness_ever, current_max_reward)

		game.quit()

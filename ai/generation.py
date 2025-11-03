import os
import random
import re
from typing import Final

import pygame
import torch
from torch import nn
import numpy as np

from ai.agent import Agent
from game.game import Game
from game.settings import BLACK

# Generation constants
DEFAULT_ELITE_COUNT: Final[int] = 4
DEFAULT_MUTATION_RATE: Final[float] = 0.01
DEFAULT_MUTATION_STRENGTH: Final[float] = 0.1
DEFAULT_TICK_RATE: Final[int] = 90
RANDOM_AGENTS_COUNT: Final[int] = 5  # Number of random agents to add for diversity
POSITION_CHECK_INTERVAL: Final[float] = 2.0  # Seconds between position checks
STUCK_CHECK_WINDOW: Final[float] = 6.0  # Seconds to check if agent is stuck


class Generation:
	def __init__(
		self,
		population_size: int,
		elite_count: int = DEFAULT_ELITE_COUNT,
		mutation_rate: float = DEFAULT_MUTATION_RATE,
		mutation_strength: float = DEFAULT_MUTATION_STRENGTH,
		load_latest_generation_weights: bool = False,
		show_window: bool = True,
		use_checkpoints: bool = False
	) -> None:
		self.population_size = population_size
		self.elite_count = elite_count
		self.mutation_strength = mutation_strength
		self.mutation_rate = mutation_rate
		self.show_window = show_window
		self.tick_rate = DEFAULT_TICK_RATE
		self.generation = 1
		self.use_checkpoints = use_checkpoints
		self.agents = [Agent(self.tick_rate, self.show_window, generation=self) for _ in range(population_size)]
		self.should_skip_checkpoint = False
		self.manual_stop = False
		self.agent_positions: dict[int, list[tuple[float, int]]] = {}
		self.last_position_check = 0.0
		self.best_fitness_ever = 0.0

		if load_latest_generation_weights:
			self.load_latest_generation_weights()

	def evolve_generation(self) -> None:
		"""Evolve generation by selecting elites and creating new offspring"""
		self.agents.sort(key=lambda agent: agent.current_reward, reverse=True)
		elites = self.agents[:self.elite_count]
		best_reward = elites[0].current_reward
		print(f"Selected {len(elites)} elites: {[f'{agent.current_reward:.2f}' for agent in elites]}")

		# Preserve elite weights (without mutation)
		elite_weights = [elite.model.state_dict() for elite in elites]
		new_agents: list[Agent] = []
		
		# Create new agents with preserved elite weights
		for elite_weight in elite_weights:
			new_agent = Agent(self.tick_rate, self.show_window, generation=self)
			new_agent.model.load_state_dict(elite_weight)
			new_agents.append(new_agent)
		
		# Create offspring from elites
		while len(new_agents) < self.population_size - RANDOM_AGENTS_COUNT:
			parent1 = random.choice(elites)
			parent2 = random.choice(elites)
			child_weights = self.crossover(parent1.model.state_dict(), parent2.model.state_dict())
			new_agent = Agent(self.tick_rate, self.show_window, generation=self)
			new_agent.model.load_state_dict(child_weights)
			self.mutate(new_agent.model)
			new_agents.append(new_agent)
		
		# Add random agents to maintain diversity
		while len(new_agents) < self.population_size:
			random_agent = Agent(self.tick_rate, self.show_window, generation=self)
			new_agents.append(random_agent)

		self.agents = new_agents[:self.population_size]
		self.generation += 1
		self.agent_positions.clear()
		self.last_position_check = 0.0
		self.manual_stop = False

		os.makedirs("weights", exist_ok=True)
		torch.save({
			'weights': elites[0].model.state_dict(),
			'best_fitness': self.best_fitness_ever,
			'mutation_rate': self.mutation_rate,
			'mutation_strength': self.mutation_strength
		}, f"weights/generation_{self.generation}.pth")

	def crossover(self, parent1_weights: dict[str, torch.Tensor], parent2_weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
		"""Cross over weights of two parents to create a child"""
		return {
			key: (parent1_weights[key] + parent2_weights.get(key, parent1_weights[key])) / 2
			for key in parent1_weights
		}

	def mutate(self, model: nn.Module) -> None:
		"""Mutate model weights"""
		for param in model.parameters():
			if param.requires_grad and np.random.rand() < self.mutation_rate:
				param.data += torch.randn_like(param) * self.mutation_strength

	def get_best_agent(self) -> Agent:
		"""Return agent with highest reward"""
		return max(self.agents, key=lambda agent: agent.current_reward)

	def load_latest_generation_weights(self) -> None:
		"""Load weights from latest generation file"""
		try:
			latest_generation = max(
				int(filename.split('_')[1].split('.')[0])
				for filename in os.listdir("weights")
				if re.match(r"generation_\d+\.pth", filename)
			)
			weights_path = f"weights/generation_{latest_generation}.pth"
			weights_data = torch.load(weights_path, weights_only=False)
			
			if isinstance(weights_data, dict) and 'weights' in weights_data:
				weights = weights_data['weights']
				if not isinstance(weights, dict):
					raise ValueError("Invalid weights format")
					
				for agent in self.agents:
					agent.model.load_state_dict(weights)
				self.best_fitness_ever = weights_data.get('best_fitness', 0.0)
				self.mutation_rate = weights_data.get('mutation_rate', self.mutation_rate)
				self.mutation_strength = weights_data.get('mutation_strength', self.mutation_strength)
			else:
				# Old format - direct state dict
				if not isinstance(weights_data, dict):
					raise ValueError("Invalid weights format")
				for agent in self.agents:
					agent.model.load_state_dict(weights_data)

			self.generation = latest_generation
			print(f"Loaded weights for generation {latest_generation}")
			print(f"Mutation parameters: rate={self.mutation_rate:.4f}, strength={self.mutation_strength:.4f}")
			self.evolve_generation()
		except (FileNotFoundError, ValueError) as e:
			print(f"No weights found, starting with random weights: {e}")

	def skip_checkpoint(self) -> None:
		"""Skip to next checkpoint"""
		self.should_skip_checkpoint = True

	def check_agent_positions(self, tick: int) -> None:
		"""Check if agents are stuck or moving backwards, kill them if so"""
		current_time = tick / 1000.0

		if current_time - self.last_position_check < POSITION_CHECK_INTERVAL:
			return

		self.last_position_check = current_time

		for agent in self.agents:
			if agent.player and not agent.player.dead and not agent.player.win:
				current_x = agent.player.rect.x
				agent_key = agent.current_index

				if agent_key not in self.agent_positions:
					self.agent_positions[agent_key] = []

				self.agent_positions[agent_key].append((current_time, current_x))
				self.agent_positions[agent_key] = [
					pos for pos in self.agent_positions[agent_key]
					if pos[0] >= current_time - STUCK_CHECK_WINDOW
				]

				positions = self.agent_positions[agent_key]

				# Check if stuck (same position for POSITION_CHECK_INTERVAL seconds)
				if len(positions) >= 2:
					recent_x = [pos[1] for pos in positions[-2:]]
					if len(set(recent_x)) == 1:
						agent.player.set_dead()
						continue

				# Check if moving backwards (X decreased over STUCK_CHECK_WINDOW seconds)
				if len(positions) >= 4:
					if current_x < positions[0][1]:
						agent.player.set_dead()

	def play_agents(self) -> None:
		"""Play games with all agents in the generation"""
		game = Game(self.population_size, self.tick_rate, self.show_window, has_playable_player=False)
		game.use_checkpoints = self.use_checkpoints
		game.init()

		game.add_key_action(pygame.K_g, self.skip_checkpoint, "Skip Checkpoint")
		game.add_key_action(pygame.K_s, lambda: setattr(self, 'manual_stop', True), "Stop Generation")

		spawn_points = [(game.level.spawn_point[0], game.level.spawn_point[1])]
		if self.use_checkpoints and game.level.checkpoints:
			spawn_points.extend(game.level.checkpoints)

		print(f"Testing agents on {len(spawn_points)} spawn points (1 initial + {len(spawn_points)-1} checkpoints)")

		for agent in self.agents:
			agent.current_reward = 0.0

		for checkpoint_idx, (spawn_x, spawn_y) in enumerate(spawn_points):
			for i, agent in enumerate(self.agents):
				player = game.players[i]
				agent.player = player
				player.rect.x = spawn_x
				player.rect.y = spawn_y
				player.dead = False
				player.win = False
				player.finished_reward = None
				player.change_x = 0
				player.change_y = 0
				player.revive()
				agent.reset_continuous_reward_tracking()

			tick = 0
			max_ticks = 30 * self.tick_rate

			while (tick < max_ticks and
				   not self.should_skip_checkpoint and
				   not self.manual_stop and
				   not any(p.win for p in game.players) and
				   not all(p.dead for p in game.players)):
				
				for agent in self.agents:
					if agent.player and not agent.player.dead and not agent.player.win:
						grid = agent.player.get_surrounding_tiles()
						direction = agent.calculate_move(grid)
						agent.player.execute_move(direction)
						agent.current_reward += agent.calculate_continuous_reward()

				game.handle_inputs()
				game.update(tick)
				tick += 1

				self.check_agent_positions(tick)

				if game.display_window:
					best_agent = self.get_best_agent()
					elapsed_time = tick / self.tick_rate
					living_agents = sum(1 for a in self.agents if a.player and not a.player.dead and not a.player.win)
					
					current_fitness = max(
						(a.player.finished_reward if a.player and a.player.finished_reward is not None else (a.player.rect.x / 10 if a.player else 0))
						for a in self.agents
					) if self.agents else 0.0

					game.draw_text(f"Best Agent: {best_agent.current_index + 1}/{self.population_size}, Generation: {self.generation}", 10, 10, font_size=24, color=BLACK)
					game.draw_text(f"FPS: {1000 / (game.clock.get_time() or 1):.1f}", 10, 70, font_size=24, color=BLACK)
					game.draw_text(f"Time: {elapsed_time:.2f}s", 10, 100, font_size=24, color=BLACK)
					if best_agent.player:
						game.draw_text(f"Player: X: {best_agent.player.rect.x}, Y: {best_agent.player.rect.y}", 10, 130, font_size=24, color=BLACK)
					game.draw_text(f"Checkpoint: {checkpoint_idx + 1}/{len(spawn_points)}", 10, 160, font_size=24, color=BLACK)
					game.draw_text(f"Living agents: {living_agents}/{self.population_size}", 10, 190, font_size=24, color=BLACK)
					game.draw_text(f"Best Fitness: {self.best_fitness_ever:.2f}", 10, 220, font_size=24, color=BLACK)
					game.draw_text(f"Current Fitness: {current_fitness:.2f}", 10, 250, font_size=24, color=BLACK)
					game.draw_text("Press S to skip generation", 10, 280, font_size=24, color=BLACK)

					game.draw()

			for agent in self.agents:
				agent.current_reward += agent.calculate_reward()

			self.should_skip_checkpoint = False

		if self.agents:
			current_max_reward = max(agent.current_reward for agent in self.agents)
			if current_max_reward > self.best_fitness_ever:
				self.best_fitness_ever = current_max_reward

		game.quit()
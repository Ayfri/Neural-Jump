import os
import re
import time
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray

from ai.a2c_trainer import A2CTrainer, RolloutBuffer
from ai.population import DEFAULT_HIDDEN_SIZES, Population, pick_device
from game.world import OBSERVATION_SIZE, World

# Generation constants
DEFAULT_ELITE_COUNT: Final[int] = 4
DEFAULT_MUTATION_RATE: Final[float] = 0.8
DEFAULT_MUTATION_STRENGTH: Final[float] = 0.015
DEFAULT_TICK_RATE: Final[int] = 90
DEFAULT_EPISODE_SECONDS: Final[float] = 20.0
RANDOM_AGENTS_COUNT: Final[int] = 5  # Number of random agents to add for diversity
POSITION_CHECK_INTERVAL: Final[float] = 2.0  # Seconds between position checks
STUCK_CHECK_WINDOW: Final[float] = 6.0  # Seconds to check if agent is stuck
ROLLOUT_MEMORY_BUDGET: Final[int] = 512 * 1024 * 1024  # Cap on the observations kept for one A2C update

# Reward constants
FORWARD_MOVEMENT_REWARD: Final[float] = 0.02
NEW_MAX_POSITION_BONUS: Final[float] = 0.1
BACKWARD_MOVEMENT_PENALTY: Final[float] = -0.1
STATIONARY_PENALTY: Final[float] = -0.05
STATIONARY_THRESHOLD: Final[int] = 5  # Ticks before penalty kicks in
FALLING_PENALTY: Final[float] = -0.02
FALLING_THRESHOLD: Final[int] = 5  # Y distance before penalty

DEATH_PENALTY: Final[float] = -20.0
WIN_TIME_BONUS_MULTIPLIER: Final[float] = 100.0
WIN_TIME_BONUS_BASE: Final[float] = 10.0
DISTANCE_REWARD_DIVISOR: Final[float] = 10.0
PROGRESS_REWARD_DIVISOR: Final[float] = 20.0
MIN_REWARD: Final[float] = -30.0

WEIGHTS_FOLDER: Final[str] = 'weights'


class Generation:
	"""Training loop: one batched World and one batched Population, stepped together and bred every generation."""

	def __init__(
		self,
		population_size: int,
		elite_count: int = DEFAULT_ELITE_COUNT,
		mutation_rate: float = DEFAULT_MUTATION_RATE,
		mutation_strength: float = DEFAULT_MUTATION_STRENGTH,
		load_latest_generation_weights: bool = False,
		show_window: bool = True,
		use_checkpoints: bool = False,
		use_a2c_learning: bool = True,
		hidden_sizes: tuple[int, int, int] = DEFAULT_HIDDEN_SIZES,
		device: str = 'auto',
		tick_rate: int = DEFAULT_TICK_RATE,
		episode_seconds: float = DEFAULT_EPISODE_SECONDS,
		render_every: int = 1,
		map_path: str = 'maps/level_1.txt',
	) -> None:
		self.population_size = population_size
		self.elite_count = min(elite_count, population_size)
		self.mutation_rate = mutation_rate
		self.mutation_strength = mutation_strength
		self.show_window = show_window
		self.use_checkpoints = use_checkpoints
		self.use_a2c_learning = use_a2c_learning
		self.tick_rate = tick_rate
		self.episode_seconds = episode_seconds
		self.render_every = max(1, render_every)
		self.generation = 1
		self.best_fitness_ever = 0.0
		self.should_skip_checkpoint = False
		self.manual_stop = False
		self.last_speed = 0.0

		self.world = World(map_path, population_size)
		self.population = Population(population_size, hidden_sizes, pick_device(device))
		self.rewards = np.zeros(population_size, dtype=np.float64)

		self.max_x_reached = np.zeros(population_size, dtype=np.float64)
		self.ticks_stationary = np.zeros(population_size, dtype=np.int32)
		self.position_history: list[NDArray[np.float64]] = []
		self.last_position_check = 0.0

		self.a2c_trainer: A2CTrainer | None = None
		self.rollout: RolloutBuffer | None = None
		if self.use_a2c_learning:
			self.a2c_trainer = A2CTrainer(self.population)
			self.rollout = RolloutBuffer(self.rollout_capacity(), population_size)
			print(self.a2c_trainer.get_training_summary())

		self.renderer = None
		if show_window:
			import pygame

			from game.render import Renderer

			self.renderer = Renderer(self.world)
			self.renderer.add_key_action(pygame.K_g, self.skip_checkpoint, 'Skip Checkpoint')
			self.renderer.add_key_action(pygame.K_s, self.stop_generation, 'Stop Generation')

		if load_latest_generation_weights:
			self.load_latest_generation_weights()

	def rollout_capacity(self) -> int:
		"""Number of transitions kept for one A2C update, bounded by the memory budget."""
		wanted = int(self.episode_seconds * self.tick_rate) * len(self.spawn_points())
		per_step = self.population_size * OBSERVATION_SIZE * 2  # float16 observations dominate the buffer
		return max(1, min(wanted, ROLLOUT_MEMORY_BUDGET // per_step))

	def skip_checkpoint(self) -> None:
		self.should_skip_checkpoint = True

	def stop_generation(self) -> None:
		self.manual_stop = True

	def spawn_points(self) -> list[tuple[int, int]]:
		points = [self.world.spawn_point]
		if self.use_checkpoints:
			points.extend(self.world.checkpoints)
		return points

	def play_agents(self) -> None:
		"""Runs the whole population through every spawn point and fills `self.rewards`."""
		points = self.spawn_points()
		self.rewards.fill(0.0)
		max_ticks = int(self.episode_seconds * self.tick_rate)
		started = time.perf_counter()
		ticks_done = 0

		for checkpoint_index, (spawn_x, spawn_y) in enumerate(points):
			self.world.reset(spawn_x, spawn_y)
			self.max_x_reached[:] = self.world.x
			self.ticks_stationary.fill(0)
			self.position_history.clear()
			self.last_position_check = 0.0

			for tick in range(max_ticks):
				if self.should_skip_checkpoint or self.manual_stop or self.world.win.any() or not self.world.alive().any():
					break

				alive = self.world.alive()
				observations = self.world.observe()
				actions = self.population.act(observations)
				previous_x = self.world.x.copy()
				previous_y = self.world.y.copy()
				self.world.step(actions, tick)

				step_rewards = self.continuous_rewards(alive, previous_x, previous_y)
				self.rewards += step_rewards
				if self.rollout is not None:
					self.rollout.add(observations, actions, step_rewards, alive)

				self.check_agent_positions(tick)
				ticks_done += 1

				if self.renderer is not None and tick % self.render_every == 0:
					self.render(checkpoint_index, len(points), tick)

			self.rewards += self.final_rewards()
			self.should_skip_checkpoint = False

		self.best_fitness_ever = max(self.best_fitness_ever, float(self.rewards.max()))
		elapsed = time.perf_counter() - started
		self.last_speed = ticks_done / elapsed if elapsed else 0.0

	def continuous_rewards(self, alive: NDArray[np.bool_], previous_x: NDArray[np.float64], previous_y: NDArray[np.float64]) -> NDArray[np.float64]:
		"""Per-tick micro rewards, computed for the whole population at once."""
		x_delta = self.world.x - previous_x
		rewards = np.zeros(self.population_size, dtype=np.float64)

		forward = x_delta > 0
		rewards += forward * FORWARD_MOVEMENT_REWARD
		new_max = forward & (self.world.x > self.max_x_reached)
		rewards += new_max * NEW_MAX_POSITION_BONUS
		np.maximum(self.max_x_reached, self.world.x, out=self.max_x_reached, where=forward)

		rewards += (x_delta < 0) * BACKWARD_MOVEMENT_PENALTY

		still = x_delta == 0
		self.ticks_stationary = np.where(still, self.ticks_stationary + 1, 0)
		rewards += (still & (self.ticks_stationary > STATIONARY_THRESHOLD)) * STATIONARY_PENALTY

		rewards += ((self.world.y - previous_y) > FALLING_THRESHOLD) * FALLING_PENALTY
		return rewards * alive

	def final_rewards(self) -> NDArray[np.float64]:
		"""End of episode reward: win bonus, reward tile value, or distance travelled minus the death penalty."""
		world = self.world
		time_taken = np.where(world.win_tick >= 0, world.win_tick / self.tick_rate, WIN_TIME_BONUS_BASE)
		win_reward = world.x / DISTANCE_REWARD_DIVISOR + np.maximum(0.0, WIN_TIME_BONUS_BASE - time_taken) * WIN_TIME_BONUS_MULTIPLIER
		progress = np.maximum(MIN_REWARD, self.max_x_reached / PROGRESS_REWARD_DIVISOR + world.dead * DEATH_PENALTY)

		rewards = np.where(world.finished_reward != 0, world.finished_reward * DISTANCE_REWARD_DIVISOR, progress)
		return np.where(world.win, win_reward, rewards)

	def check_agent_positions(self, tick: int) -> None:
		"""Kills agents that are stuck in place or crawling backwards."""
		current_time = tick / self.tick_rate
		if current_time - self.last_position_check < POSITION_CHECK_INTERVAL:
			return

		self.last_position_check = current_time
		self.position_history.append(self.world.x.copy())
		window = int(STUCK_CHECK_WINDOW / POSITION_CHECK_INTERVAL) + 1
		del self.position_history[:-window]

		alive = self.world.alive()
		if len(self.position_history) >= 2:
			self.world.kill(alive & (self.position_history[-1] == self.position_history[-2]))
		if len(self.position_history) >= 4:
			self.world.kill(alive & (self.world.x < self.position_history[0]))

	def render(self, checkpoint_index: int, checkpoint_count: int, tick: int) -> None:
		assert self.renderer is not None
		self.renderer.poll_events()
		best = int(np.argmax(np.where(self.world.alive(), self.rewards, -np.inf)))
		living = int(self.world.alive().sum())
		self.renderer.draw(
			best,
			[
				f'Generation: {self.generation} | Agent {best + 1}/{self.population_size}',
				f'Time: {tick / self.tick_rate:.2f}s | Checkpoint {checkpoint_index + 1}/{checkpoint_count}',
				f'Living agents: {living}/{self.population_size}',
				f'Best fitness ever: {self.best_fitness_ever:.2f} | Current: {self.rewards.max():.2f}',
				f'X: {int(self.world.x[best])} Y: {int(self.world.y[best])}',
			],
			self.tick_rate,
		)

	def evolve_generation(self) -> None:
		"""Runs the A2C update, selects the elites, breeds the next generation and saves the best weights."""
		if self.a2c_trainer is not None and self.rollout is not None:
			print('Performing A2C learning step...')
			stats = self.a2c_trainer.train_step(self.rollout)
			print(f'A2C Training Stats: {stats}')
			self.a2c_trainer.decay_entropy(self.generation)

		elites = self.population.evolve(
			self.rewards.astype(np.float32),
			self.elite_count,
			min(RANDOM_AGENTS_COUNT, max(0, self.population_size - self.elite_count)),
			self.mutation_rate,
			self.mutation_strength,
		)
		print(f'Selected {len(elites)} elites: {[f"{self.rewards[i]:.2f}" for i in elites]}')

		if self.a2c_trainer is not None:
			# The agents behind each slot just changed, so the Adam moments no longer describe them
			self.a2c_trainer.reset_optimizer()

		self.generation += 1
		self.manual_stop = False

		os.makedirs(WEIGHTS_FOLDER, exist_ok=True)
		torch.save({
			'weights': self.population.state_dict(0),
			'hidden_sizes': self.population.hidden_sizes,
			'best_fitness': self.best_fitness_ever,
			'mutation_rate': self.mutation_rate,
			'mutation_strength': self.mutation_strength,
		}, f'{WEIGHTS_FOLDER}/generation_{self.generation}.pth')

	def load_latest_generation_weights(self) -> None:
		"""Loads the most recent weight file into every agent, then immediately breeds from it."""
		try:
			latest = max(
				int(filename.split('_')[1].split('.')[0])
				for filename in os.listdir(WEIGHTS_FOLDER)
				if re.match(r'generation_\d+\.pth', filename)
			)
			data = torch.load(f'{WEIGHTS_FOLDER}/generation_{latest}.pth', weights_only=True, map_location='cpu')
			weights = data['weights'] if isinstance(data, dict) and 'weights' in data else data
			if not isinstance(weights, dict):
				raise ValueError('invalid weights format')

			self.population.load_state_dict(weights)
			if isinstance(data, dict):
				self.best_fitness_ever = float(data.get('best_fitness', 0.0))
				self.mutation_rate = float(data.get('mutation_rate', self.mutation_rate))
				self.mutation_strength = float(data.get('mutation_strength', self.mutation_strength))

			self.generation = latest
			print(f'Loaded weights for generation {latest}')
			print(f'Mutation parameters: rate={self.mutation_rate:.4f}, strength={self.mutation_strength:.4f}')
			self.evolve_generation()
		except (FileNotFoundError, ValueError, KeyError) as error:
			print(f'No usable weights found, starting with random weights: {error}')

	def quit(self) -> None:
		if self.renderer is not None:
			self.renderer.quit()

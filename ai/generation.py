import os
import re
import time
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray

from ai.population import DEFAULT_HIDDEN_SIZES, Population, pick_device, seed_everything
from game.world import World

# Generation constants
DEFAULT_ELITE_COUNT: Final[int] = 4
DEFAULT_MUTATION_RATE: Final[float] = 0.8
DEFAULT_MUTATION_STRENGTH: Final[float] = 0.03  # Now that the elites are preserved, this is the only exploration the GA has left
DEFAULT_TICK_RATE: Final[int] = 90
DEFAULT_EPISODE_SECONDS: Final[float] = 30.0
DEFAULT_ACTION_REPEAT: Final[int] = 2  # Physics ticks a chosen action is held for, measurably better than 1 or 4 here
MAX_SPEED: Final[str] = 'max'  # `speed='max'` tunes itself to the fastest rate the target framerate survives
MAX_TICKS_PER_FRAME: Final[float] = 4096.0
SPEED_HOLD_RATIO: Final[float] = 0.95  # Share of the target framerate above which the simulation asks for more work
SPEED_DROP_RATIO: Final[float] = 0.88  # Share under which it backs off
SPEED_ADAPT_FRAMES: Final[int] = 12  # Frames between two adjustments, so the framerate average settles first
SPEED_SAMPLE_SECONDS: Final[float] = 0.25
RANDOM_AGENTS_COUNT: Final[int] = 5  # Number of random agents to add for diversity
POSITION_CHECK_INTERVAL: Final[float] = 2.0  # Seconds between position checks
STUCK_CHECK_WINDOW: Final[float] = 6.0  # Seconds to check if agent is stuck

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
		deterministic_actions: bool = True,
		seed: int | None = None,
		hidden_sizes: tuple[int, int, int] = DEFAULT_HIDDEN_SIZES,
		device: str = 'auto',
		tick_rate: int = DEFAULT_TICK_RATE,
		episode_seconds: float = DEFAULT_EPISODE_SECONDS,
		action_repeat: int = DEFAULT_ACTION_REPEAT,
		speed: float | str = 1.0,
		fps: int = 0,
		map_path: str = 'maps/level_1.txt',
	) -> None:
		self.population_size = population_size
		self.elite_count = min(elite_count, population_size)
		self.mutation_rate = mutation_rate
		self.mutation_strength = mutation_strength
		self.show_window = show_window
		self.use_checkpoints = use_checkpoints
		# Sampling makes the measured fitness a lottery, which is fatal to selection, so the population
		# plays its argmax and an elite re-scores exactly what it scored before
		self.deterministic_actions = deterministic_actions
		self.seed = seed
		if seed is not None:
			seed_everything(seed)
		self.tick_rate = tick_rate
		self.episode_seconds = episode_seconds
		self.action_repeat = max(1, action_repeat)
		self.auto_speed = speed == MAX_SPEED
		self.speed = 1.0 if self.auto_speed else float(speed)
		self.generation = 1
		self.best_fitness_ever = 0.0
		self.should_skip_checkpoint = False
		self.manual_stop = False
		self.last_speed = 0.0
		self.live_speed = 0.0
		self.total_ticks = 0
		self.fitness_history: list[float] = []

		self.world = World(map_path, population_size)
		self.population = Population(population_size, hidden_sizes, pick_device(device))
		self.rewards = np.zeros(population_size, dtype=np.float64)

		self.max_x_reached = np.zeros(population_size, dtype=np.float64)
		self.ticks_stationary = np.zeros(population_size, dtype=np.int32)
		self.position_history: list[NDArray[np.float64]] = []
		self.last_position_check = 0.0
		self._previous_x = np.zeros(population_size, dtype=np.float64)
		self._previous_y = np.zeros(population_size, dtype=np.float64)
		self._actions = np.zeros(population_size, dtype=np.int64)

		self.renderer = None
		self.ticks_per_frame = 1.0
		self._tick_budget = 0.0
		self._speed_time = 0.0
		self._speed_ticks = 0
		self._frames_since_adapt = 0
		if show_window:
			import pygame

			from game.render import Renderer

			self.renderer = Renderer(self.world, fps)
			self.renderer.add_key_action(pygame.K_g, self.skip_checkpoint, 'Skip Checkpoint')
			self.renderer.add_key_action(pygame.K_s, self.stop_generation, 'Stop Generation')
			self.ticks_per_frame = 8.0 if self.auto_speed else self.tick_rate * self.speed / self.renderer.target_fps

		if load_latest_generation_weights:
			self.load_latest_generation_weights()

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
		self._speed_time = started
		self._speed_ticks = self.total_ticks
		ticks_done = 0

		for checkpoint_index, (spawn_x, spawn_y) in enumerate(points):
			self.world.reset(spawn_x, spawn_y)
			self.max_x_reached[:] = self.world.x
			self.ticks_stationary.fill(0)
			self.position_history.clear()
			self.last_position_check = 0.0
			tick = 0

			while tick < max_ticks and not self.episode_over():
				# One frame's worth of simulation at a time, so the render rate and the sim rate stay independent
				steps = max_ticks - tick
				if self.renderer is not None:
					self.render(checkpoint_index, len(points), tick)
					self._tick_budget += self.ticks_per_frame
					steps = min(int(self._tick_budget), steps)
					self._tick_budget -= steps
					self._adapt_ticks_per_frame()

				for _ in range(steps):
					self.simulate_tick(tick)
					tick += 1
					ticks_done += 1
					if self.episode_over():
						break

			self.rewards += self.final_rewards()
			self.should_skip_checkpoint = False

		self.best_fitness_ever = max(self.best_fitness_ever, float(self.rewards.max()))
		elapsed = time.perf_counter() - started
		self.last_speed = ticks_done / elapsed if elapsed else 0.0

	def episode_over(self) -> bool:
		return self.should_skip_checkpoint or self.manual_stop or bool(self.world.win.any()) or not self.world.alive().any()

	def simulate_tick(self, tick: int) -> None:
		"""
		One simulation tick: observe and decide on the first tick of a window, then step, reward and record.

		Holding an action for `action_repeat` ticks is both cheaper and easier to learn from: a jump arc lasts
		about 42 ticks, so at one decision per tick no reachable discount factor reaches the far side of a gap.
		A window cut short by the end of an episode is simply dropped, its observation slot gets reused.
		"""
		alive = self.world.alive()
		if tick % self.action_repeat == 0:
			self._actions = self.population.act(self.world.observe(), self.deterministic_actions)

		np.copyto(self._previous_x, self.world.x)
		np.copyto(self._previous_y, self.world.y)
		self.world.step(self._actions, tick)

		self.rewards += self.continuous_rewards(alive, self._previous_x, self._previous_y)

		self.check_agent_positions(tick)
		self.total_ticks += 1

	def _adapt_ticks_per_frame(self) -> None:
		"""
		In `--speed max`, walks the tick budget up while the framerate holds and backs off when it slips.

		The controller watches the measured framerate rather than the time a frame spends working: with
		vsync on, waiting for the next refresh is indistinguishable from working, so only the framerate
		itself says whether there is headroom left.
		"""
		if not self.auto_speed or self.renderer is None:
			return
		self._frames_since_adapt += 1
		if self._frames_since_adapt < SPEED_ADAPT_FRAMES:
			return

		self._frames_since_adapt = 0
		ratio = self.renderer.measured_fps() / max(1, self.renderer.target_fps)
		if ratio >= SPEED_HOLD_RATIO:
			self.ticks_per_frame = min(MAX_TICKS_PER_FRAME, self.ticks_per_frame * 1.3)
		elif ratio < SPEED_DROP_RATIO:
			self.ticks_per_frame = max(1.0, self.ticks_per_frame * 0.7)

	def _sample_live_speed(self) -> None:
		"""Rolling ticks-per-real-second, so the HUD number moves with the simulation instead of the generation."""
		now = time.perf_counter()
		elapsed = now - self._speed_time
		if elapsed < SPEED_SAMPLE_SECONDS:
			return
		instant = (self.total_ticks - self._speed_ticks) / elapsed
		self.live_speed = instant if self.live_speed == 0.0 else self.live_speed * 0.6 + instant * 0.4
		self._speed_time, self._speed_ticks = now, self.total_ticks

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

	def training_rows(self) -> list[tuple[str, str]]:
		"""The knobs that shape the run, laid out for the HUD's training panel."""
		rows = [
			('Population', f'{self.population_size}'),
			('Mutation Rate', f'{self.mutation_rate:.3f}'),
			('Mutation Str', f'{self.mutation_strength:.4f}'),
			('Elites', f'{self.elite_count}'),
			('Network', 'x'.join(str(size) for size in self.population.hidden_sizes)),
			('Device', str(self.population.device)),
		]
		return [
			*rows,
			('Actions', 'argmax' if self.deterministic_actions else 'sampled'),
			('Act Repeat', f'{self.action_repeat}'),
		]

	def render(self, checkpoint_index: int, checkpoint_count: int, tick: int) -> None:
		assert self.renderer is not None
		from game.render import Hud

		self.renderer.poll_events()
		self._sample_live_speed()
		best = int(np.argmax(np.where(self.world.alive(), self.rewards, -np.inf)))
		self.renderer.draw(best, self.rewards, Hud(
			generation=self.generation,
			tick=tick,
			tick_rate=self.tick_rate,
			checkpoint=(checkpoint_index + 1, checkpoint_count),
			best_ever=self.best_fitness_ever,
			elite_count=self.elite_count if self.generation > 1 else 0,
			random_count=min(RANDOM_AGENTS_COUNT, max(0, self.population_size - self.elite_count)) if self.generation > 1 else 0,
			speed=self.live_speed,
			sim_speed=self.live_speed / self.tick_rate,
			training=self.training_rows(),
			history=self.fitness_history,
		))

	def evolve_generation(self) -> None:
		"""Selects the elites, breeds the next generation and saves the best weights."""
		self.fitness_history.append(float(self.rewards.max()))
		elites = self.population.evolve(
			self.rewards.astype(np.float32),
			self.elite_count,
			min(RANDOM_AGENTS_COUNT, max(0, self.population_size - self.elite_count)),
			self.mutation_rate,
			self.mutation_strength,
		)
		print(f'Selected {len(elites)} elites: {[f"{self.rewards[i]:.2f}" for i in elites]}')

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

import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import torch
from numpy.typing import NDArray

from ai.device_runner import DeviceRunner
from ai.population import DEFAULT_HIDDEN_SIZES, Population, pick_device, seed_everything
from ai.ppo import (
	DEFAULT_CLIP_RANGE, DEFAULT_ENTROPY_COEF, DEFAULT_EPOCHS, DEFAULT_GAE_LAMBDA, DEFAULT_GAMMA,
	DEFAULT_LEARNING_RATE, DEFAULT_MINIBATCHES, DEFAULT_ROLLOUT_STEPS, DEFAULT_TARGET_KL, PPOTrainer,
	RolloutBuffer,
)
from ai.ppo_runner import PPORunner, SpawnCurriculum
from ai.rewards import (
	BACKWARD_MOVEMENT_PENALTY, COIN_REWARD, DEATH_PENALTY, DISTANCE_REWARD_DIVISOR, FALLING_PENALTY,
	FALLING_THRESHOLD, FORWARD_MOVEMENT_REWARD, MIN_REWARD, NEW_MAX_POSITION_BONUS, PROGRESS_REWARD_DIVISOR,
	PROGRESS_SPEED_BONUS, STATIONARY_PENALTY, STATIONARY_THRESHOLD, WIN_BASE_BONUS, WIN_SPEED_BONUS,
	WIN_SPEED_EXPONENT,
)
from game.art import COIN_COLOR
from game.menu import MENU_FOOTER, SETTINGS_FOOTER, Item, MenuStack, Page, level_name
from game.settings import TILE_SIZE
from game.world import World
from game.world_cuda import CudaWorld

if TYPE_CHECKING:
	from game.render import Renderer, Row

# Evolution
DEFAULT_POPULATION_SIZE: Final[int] = 300
DEFAULT_ELITE_COUNT: Final[int] = 6
DEFAULT_MUTATION_RATE: Final[float] = 0.8
DEFAULT_MUTATION_STRENGTH: Final[float] = 0.02  # Swept: the level gets furthest here, and a plateau outlives a run either side
RANDOM_AGENTS_COUNT: Final[int] = 5  # Number of random agents to add for diversity

# Simulation
DEFAULT_TICK_RATE: Final[int] = 90
DEFAULT_EPISODE_SECONDS: Final[float] = 60.0  # The level's fastest route is about 44 seconds, the rest is room to detour for coins
DEFAULT_ACTION_REPEAT: Final[int] = 2  # Physics ticks a chosen action is held for, measurably better than 1 or 4 here
POSITION_CHECK_INTERVAL: Final[float] = 2.0  # Seconds between position checks
STUCK_CHECK_WINDOW: Final[float] = 6.0  # Seconds to check if agent is stuck

# Display pacing
MAX_SPEED: Final[str] = 'max'  # `speed='max'` tunes itself to the fastest rate the target framerate survives
MAX_TICKS_PER_FRAME: Final[float] = 4096.0
SPEED_HOLD_RATIO: Final[float] = 0.95  # Share of the target framerate above which the simulation asks for more work
SPEED_DROP_RATIO: Final[float] = 0.88  # Share under which it backs off
SPEED_ADAPT_FRAMES: Final[int] = 12  # Frames between two adjustments, so the framerate average settles first
SPEED_SAMPLE_SECONDS: Final[float] = 0.25
SPEED_STEP: Final[float] = 2.0  # Factor the slower and faster keys apply
MIN_SPEED: Final[float] = 0.1
MAX_SPEED_MULTIPLIER: Final[float] = 1024.0

TRAINERS: Final[tuple[str, ...]] = ('ppo', 'ga')
DEFAULT_TRAINER: Final[str] = 'ppo'
DEFAULT_ENV_COUNT: Final[int] = 1024  # Environments PPO runs by default: one policy, so the action pass barely notices
SPAWN_MODES: Final[tuple[str, ...]] = ('curriculum', 'uniform', 'start')
DEFAULT_SPAWN_MODE: Final[str] = 'curriculum'
DEFAULT_SPAWN_SPACING: Final[int] = 20  # Tiles between two rungs of the curriculum's ladder

ALIVE_CHECK_TICKS: Final[int] = 90  # Ticks between two host reads of the alive mask, on the device path

TRAINING_CONTROLS: Final[tuple[tuple[str, str], ...]] = (
	('Esc', 'Open this menu, or step back out of it'),
	('Space', 'Pause and resume, the window stays live'),
	('Tab', 'Hide the panels, leaving the level and the agents'),
	('F, 1, -, =', 'Speed max, x1, halve, double, also on the numpad'),
	('S', 'End the rollout or generation now and learn from it'),
	('G', 'Move the curriculum on a rung, or skip the checkpoint'),
	('R', 'Start the whole run over with random weights'),
	('C', 'Camera lock, which only advances toward the furthest agent'),
	('V, drag', 'Free the camera and pan it'),
	('Wheel, PgUp, PgDn', 'Zoom, x0.2 out to x4 in'),
	('0', 'Back to x1 on the followed agent'),
)

WEIGHTS_FOLDER: Final[Path] = Path('weights')


class Generation:
	"""Training loop: one batched World and one batched Population, stepped together and bred every generation."""

	def __init__(
		self,
		population_size: int = DEFAULT_POPULATION_SIZE,
		trainer: str = DEFAULT_TRAINER,
		# Evolution
		elite_count: int = DEFAULT_ELITE_COUNT,
		mutation_rate: float = DEFAULT_MUTATION_RATE,
		mutation_strength: float = DEFAULT_MUTATION_STRENGTH,
		# Network
		hidden_sizes: tuple[int, int, int] = DEFAULT_HIDDEN_SIZES,
		device: str = 'auto',
		# Simulation
		map_path: str = 'maps/level_1.txt',
		episode_seconds: float = DEFAULT_EPISODE_SECONDS,
		tick_rate: int = DEFAULT_TICK_RATE,
		action_repeat: int = DEFAULT_ACTION_REPEAT,
		use_checkpoints: bool = False,
		# PPO
		spawn_mode: str = DEFAULT_SPAWN_MODE,
		spawn_spacing: int = DEFAULT_SPAWN_SPACING,
		rollout_steps: int = DEFAULT_ROLLOUT_STEPS,
		learning_rate: float = DEFAULT_LEARNING_RATE,
		gamma: float = DEFAULT_GAMMA,
		gae_lambda: float = DEFAULT_GAE_LAMBDA,
		clip_range: float = DEFAULT_CLIP_RANGE,
		epochs: int = DEFAULT_EPOCHS,
		minibatches: int = DEFAULT_MINIBATCHES,
		entropy_coef: float = DEFAULT_ENTROPY_COEF,
		target_kl: float = DEFAULT_TARGET_KL,
		# Run
		seed: int | None = None,
		load_latest_generation_weights: bool = False,
		# Display
		show_window: bool = True,
		speed: float | str = 1.0,
		fps: int = 0,
		renderer: 'Renderer | None' = None,
	) -> None:
		self.population_size = population_size
		self.trainer = trainer
		self.elite_count = min(elite_count, population_size)
		self.mutation_rate = mutation_rate
		self.mutation_strength = mutation_strength

		self.map_path = Path(map_path).as_posix()
		self.episode_seconds = episode_seconds
		self.tick_rate = tick_rate
		self.action_repeat = max(1, action_repeat)
		self.use_checkpoints = use_checkpoints

		self.seed = seed
		if seed is not None:
			seed_everything(seed)

		self.show_window = show_window
		self.auto_speed = speed == MAX_SPEED
		self.speed = 1.0 if self.auto_speed else float(speed)

		self.generation = 1
		# A policy gradient's episodes can be worth less than nothing, so the record starts below every score
		self.best_fitness_ever = -float('inf') if trainer == 'ppo' else 0.0
		self.best_time_ever = 0.0  # Fastest run to the flag in seconds, 0 while nobody has reached it
		self.should_skip_checkpoint = False
		self.manual_stop = False
		self.paused = False
		self.to_title = False  # Whether the run ended by asking for the title screen rather than by closing the window
		self.restart_requested = False
		self.last_speed = 0.0
		self.live_speed = 0.0
		self.total_ticks = 0
		self.fitness_history: list[float] = []

		self.world = World(map_path, population_size)
		resolved = pick_device(device)
		# PPO trains one network on every environment at once, so the batched tensors hold a single agent
		# asked about `population_size` observations rather than `population_size` agents asked about one
		self.population = (
			Population(1, hidden_sizes, resolved, batch=population_size, critic=True) if trainer == 'ppo'
			else Population(population_size, hidden_sizes, resolved)
		)
		self.rewards = np.zeros(population_size, dtype=np.float64)
		self._rewards_banked = np.zeros(population_size, dtype=np.float64)  # What the checkpoints already played are worth

		# On CUDA the whole tick runs on the device, physics and rewards both, and the host only reads between
		# windows. PPO needs that path whatever the device, since an episode restarts one environment at a time.
		self.cuda_world = CudaWorld(self.world, resolved) if resolved.type == 'cuda' or trainer == 'ppo' else None
		self.runner = DeviceRunner(self.cuda_world, self.population, self.action_repeat) if self.cuda_world and trainer == 'ga' else None
		self._alive_checked_tick = 0
		self._anyone_alive = True
		self._stuck_owed = 0
		self._collected = 0  # Steps of the last rollout that were actually played, which a manual stop cuts short
		self._updating = False  # True while the policy is being updated, which the HUD says and the window survives
		self._drawn = False  # Whether this update's one frame has gone out, since nothing moves for the rest of it

		self.spawn_mode = spawn_mode
		self.spawn_spacing = spawn_spacing
		self.menu = MenuStack()  # Empty while training, and anything on it holds the simulation behind its page
		self._map_requested: str | None = None  # Level picked in it, swapped in between two rollouts
		spawns = self.world.ground_spawns(spawn_spacing)
		self.curriculum = SpawnCurriculum(spawns, spawn_mode if len(spawns) > 1 else 'start')
		self.ppo: PPOTrainer | None = None
		self.ppo_runner: PPORunner | None = None
		self.buffer: RolloutBuffer | None = None
		self.last_stats: dict[str, float] = {}
		if trainer == 'ppo':
			assert self.cuda_world is not None
			self.buffer = RolloutBuffer(rollout_steps, population_size, resolved)
			self.ppo_runner = PPORunner(self.cuda_world, self.population, self.buffer, self.action_repeat, self.max_ticks, self.curriculum)
			self.ppo = PPOTrainer(
				self.population, population_size, learning_rate=learning_rate, gamma=gamma, gae_lambda=gae_lambda,
				clip_range=clip_range, epochs=epochs, minibatches=minibatches, entropy_coef=entropy_coef, target_kl=target_kl,
			)

		self.max_x_reached = np.zeros(population_size, dtype=np.float64)
		self.max_x_tick = np.zeros(population_size, dtype=np.float64)  # Tick each record was set on, so progress is scored on time too
		self.ticks_stationary = np.zeros(population_size, dtype=np.int32)
		self.position_history: list[NDArray[np.float64]] = []
		self._previous_x = np.zeros(population_size, dtype=np.float64)
		self._previous_y = np.zeros(population_size, dtype=np.float64)
		self._actions = np.zeros(population_size, dtype=np.int64)
		self._tick_rewards = np.zeros(population_size, dtype=np.float64)  # Scratch the per-tick terms land in

		self.renderer: Renderer | None = None
		self.ticks_per_frame = 1.0
		self._tick_budget = 0.0
		self._speed_time = 0.0
		self._speed_ticks = 0
		self._frames_since_adapt = 0
		if show_window:
			import pygame

			from game.render import Renderer

			# A shell hands its window on rather than building a second one, which would re-initialise the display
			if renderer is None:
				renderer = Renderer(self.world, fps)
			else:
				renderer.reset_key_actions()
				renderer.set_world(self.world)
			self.renderer = renderer
			# Only what is worth a key while training: everything else is a row on the pause menu
			bindings: list[tuple[int, Callable[[], None], str]] = [
				(pygame.K_ESCAPE, self.open_menu, 'Menu'),
				(pygame.K_SPACE, self.toggle_pause, 'Pause'),
				(pygame.K_TAB, renderer.toggle_hud, 'Panels'),
				(pygame.K_f, lambda: self.set_speed(MAX_SPEED), 'Speed max'),
				(pygame.K_1, lambda: self.set_speed(1.0), ''),
				(pygame.K_c, renderer.toggle_camera_lock, ''),
				(pygame.K_MINUS, lambda: self.scale_speed(1 / SPEED_STEP), ''),
				(pygame.K_EQUALS, lambda: self.scale_speed(SPEED_STEP), ''),
				(pygame.K_KP_MINUS, lambda: self.scale_speed(1 / SPEED_STEP), ''),
				(pygame.K_KP_PLUS, lambda: self.scale_speed(SPEED_STEP), ''),
				(pygame.K_g, self.skip_checkpoint, ''),
				(pygame.K_s, self.stop_generation, ''),
				(pygame.K_r, self.restart_run, ''),
			]
			for key, action, description in bindings:
				renderer.add_key_action(key, action, description)
			self._apply_speed()

		if load_latest_generation_weights:
			self.load_latest_generation_weights()

	@property
	def max_ticks(self) -> int:
		return int(self.episode_seconds * self.tick_rate)

	@property
	def stuck_check_ticks(self) -> int:
		return max(1, int(POSITION_CHECK_INTERVAL * self.tick_rate))

	def toggle_pause(self) -> None:
		self.paused = not self.paused

	def set_speed(self, speed: float | str) -> None:
		"""Switches between a fixed multiplier and `max`, the mode that tunes itself to the framerate."""
		self.auto_speed = speed == MAX_SPEED
		if not self.auto_speed:
			self.speed = min(MAX_SPEED_MULTIPLIER, max(MIN_SPEED, float(speed)))
		self._apply_speed()

	def scale_speed(self, factor: float) -> None:
		"""Multiplies the speed, leaving `max` at whatever multiplier it had reached rather than at 1."""
		self.set_speed(self.current_speed * factor)

	@property
	def current_speed(self) -> float:
		"""The multiplier actually being played, which in `max` is whatever the controller has walked up to."""
		if not self.auto_speed or self.renderer is None:
			return self.speed
		return self.ticks_per_frame * self.renderer.target_fps / self.tick_rate

	def _apply_speed(self) -> None:
		"""Rebuilds the per frame tick budget, dropping whatever the previous speed had left in it."""
		if self.renderer is None:
			return
		self.ticks_per_frame = 8.0 if self.auto_speed else self.tick_rate * self.speed / self.renderer.target_fps
		self._tick_budget = 0.0
		self._frames_since_adapt = 0

	def restart_run(self) -> None:
		"""Asks for a fresh run: the current generation is dropped rather than bred from, so nothing is saved."""
		self.restart_requested = True

	def _restart(self) -> None:
		"""Throws the run away: random weights again, generation back to 1, every record cleared."""
		self.population.randomize(torch.arange(self.population.size, device=self.population.device))
		self.generation = 1
		self.best_fitness_ever = -float('inf') if self.ppo is not None else 0.0
		self.best_time_ever = 0.0
		self.fitness_history.clear()
		self.rewards.fill(0.0)
		self._rewards_banked.fill(0.0)
		self.total_ticks = 0
		self.live_speed = 0.0
		self.restart_requested = False
		self.manual_stop = False
		self.should_skip_checkpoint = False
		if self.ppo is not None and self.ppo_runner is not None:
			self.ppo = PPOTrainer(
				self.population, self.population_size, learning_rate=self.ppo.optimizer.param_groups[0]['lr'],
				gamma=self.ppo.gamma, gae_lambda=self.ppo.gae_lambda, clip_range=self.ppo.clip_range,
				epochs=self.ppo.epochs, minibatches=self.ppo.minibatches, entropy_coef=self.ppo.entropy_start,
				target_kl=self.ppo.target_kl,
			)
			self.ppo_runner.reset_all()
		print('Restarted from scratch, generation 1')

	def skip_checkpoint(self) -> None:
		"""Ends the current checkpoint's episode, or under PPO moves the curriculum on a rung by hand."""
		if self.ppo_runner is not None and self.ppo is not None:
			self.curriculum.promote()
			self.ppo_runner.apply_curriculum()
			self.ppo.restart_exploration()
			print(f'Curriculum: episodes now start from rung {self.curriculum.front} of {len(self.curriculum.spawns) - 1}')
			return
		self.should_skip_checkpoint = True

	def stop_generation(self) -> None:
		self.manual_stop = True

	# --- Menus ---------------------------------------------------------------------------------------------

	def open_menu(self) -> None:
		self.menu.push(self.pause_page())

	def pause_page(self) -> Page:
		ppo = self.ppo is not None
		menu = self.menu
		return Page('Paused', [
			Item('Resume', menu.close),
			Item('End the rollout now' if ppo else 'Stop the generation now', menu.resuming(self.stop_generation)),
			Item('Next curriculum rung' if ppo else 'Skip this checkpoint', menu.resuming(self.skip_checkpoint)),
			Item('Restart from scratch', menu.resuming(self.restart_run)),
			Item('Settings', lambda: menu.push(self.settings_page())),
			Item('Select level', lambda: menu.push(menu.maps_page(self.map_path, self.request_map, 'The policy carries over, everything the map is baked into is rebuilt'))),
			Item('Controls', lambda: menu.push(menu.controls_page(TRAINING_CONTROLS))),
			Item('Quit to title', self.quit_to_title),
		], subtitle=f'{self.trainer.upper()} on {level_name(self.map_path)}, update {self.generation}', footer=MENU_FOOTER)

	def settings_page(self) -> Page:
		from game.render import ZOOM_STEP

		renderer = self.renderer
		assert renderer is not None
		return Page('Settings', [
			Item('Simulation speed', adjust=lambda step: self.scale_speed(SPEED_STEP ** step),
				value=lambda: MAX_SPEED if self.auto_speed else f'x{self.speed:g}'),
			Item('Tune the speed to the framerate', lambda: self.set_speed(MAX_SPEED)),
			Item('Zoom', adjust=lambda step: renderer.set_zoom(renderer.zoom * ZOOM_STEP ** step), value=lambda: f'x{renderer.zoom:.2f}'),
			Item('Free camera', renderer.toggle_freecam, value=lambda: 'on' if renderer.freecam else 'off'),
			Item('Camera lock', renderer.toggle_camera_lock, value=lambda: 'on' if renderer.camera_lock else 'off'),
			Item('Panels', renderer.toggle_hud, value=lambda: 'shown' if renderer.show_hud else 'hidden'),
			Item('Reset the view', renderer.reset_view),
			Item('Back', self.menu.back),
		], subtitle='The camera lock only ever advances toward the furthest agent', footer=SETTINGS_FOOTER)

	def request_map(self, map_path: str) -> None:
		"""Asks for a level: what is in flight is dropped and the swap happens between two rollouts."""
		self._map_requested = map_path
		self.menu.close()

	def quit_to_title(self) -> None:
		"""Ends the run and hands the window back, which the shell turns into the title screen again."""
		self.menu.close()
		self.to_title = True
		self.manual_stop = True

	def _load_map(self, map_path: str) -> None:
		"""
		Swaps the level under a running trainer, keeping the policy that was learned on the last one.

		Everything the map is baked into is built again: both worlds, the runner holding the compiled window and
		its captured graph, and the curriculum ladder read off the new floor. That costs a compile and a capture,
		which is why it happens between two rollouts rather than inside one. The network and its optimiser carry
		over, which is the point of switching at all, while every record and tracker starts from scratch since
		a score on one level says nothing about another.
		"""
		self.map_path = Path(map_path).as_posix()
		self._map_requested = None
		print(f'Loading {self.map_path}, which compiles and captures the simulation again')
		if self.renderer is not None:
			self.renderer.draw_message(f'Loading {level_name(self.map_path)}','compiling and capturing, a few seconds')

		self.world = World(map_path, self.population_size)
		device = self.population.device
		self.cuda_world = CudaWorld(self.world, device) if device.type == 'cuda' or self.trainer == 'ppo' else None
		self.runner = DeviceRunner(self.cuda_world, self.population, self.action_repeat) if self.cuda_world and self.trainer == 'ga' else None

		spawns = self.world.ground_spawns(self.spawn_spacing)
		self.curriculum = SpawnCurriculum(spawns, self.spawn_mode if len(spawns) > 1 else 'start')
		if self.ppo is not None and self.buffer is not None:
			assert self.cuda_world is not None
			self.ppo_runner = PPORunner(self.cuda_world, self.population, self.buffer, self.action_repeat, self.max_ticks, self.curriculum)
			self.ppo.restart_exploration()

		self.best_fitness_ever = -float('inf') if self.ppo is not None else 0.0
		self.best_time_ever = 0.0
		self.fitness_history.clear()
		self.position_history.clear()
		self.last_stats = {}
		self.rewards.fill(0.0)
		self._rewards_banked.fill(0.0)
		self.max_x_reached.fill(0.0)
		self.max_x_tick.fill(0.0)
		self.ticks_stationary.fill(0)
		self._collected = 0
		self._stuck_owed = 0
		self._anyone_alive = True
		if self.renderer is not None:
			self.renderer.set_world(self.world)

	def spawn_points(self) -> list[tuple[int, int]]:
		points = [self.world.spawn_point]
		if self.use_checkpoints:
			points.extend(self.world.checkpoints)
		return points

	def play_agents(self) -> None:
		"""Plays what the trainer scores on: a rollout of transitions for PPO, a full episode for evolution."""
		if self._map_requested is not None:
			self._load_map(self._map_requested)
		if self.ppo_runner is not None:
			self.collect_rollout()
			return
		self._play_episodes()

	def collect_rollout(self) -> None:
		"""
		Fills one rollout buffer, paced and rendered exactly like an episode is.

		Nothing here waits for an episode to finish: environments start and end their own, so the loop counts
		decisions and hands the buffer over the moment it is full, whatever each environment is in the middle of.
		"""
		assert self.ppo_runner is not None and self.buffer is not None
		started = time.perf_counter()
		self._speed_time = started
		self._speed_ticks = self.total_ticks
		self.ppo_runner.begin_rollout()

		wanted = self.buffer.steps
		played = 0
		while played < wanted and not (self.manual_stop or self.restart_requested or self._map_requested is not None):
			steps = wanted - played
			if self.renderer is not None:
				self.render(0, 1, played * self.action_repeat)
				if self.paused or self.menu.open:
					self._tick_budget = 0.0
					continue
				self._tick_budget += self.ticks_per_frame
				steps = min(int(self._tick_budget) // self.action_repeat, steps)
				self._tick_budget -= steps * self.action_repeat
				self._adapt_ticks_per_frame()

			for _ in range(steps):
				self.ppo_runner.collect_step()
				self.total_ticks += self.action_repeat
				self._stuck_owed += self.action_repeat
				if self._stuck_owed >= self.stuck_check_ticks:
					self._stuck_owed = 0
					self.ppo_runner.check_positions(self.stuck_check_ticks)
			played += steps

		self._collected = played
		self._sync_from_device()
		best_return, best_tick = self.ppo_runner.records()
		self.best_fitness_ever = max(self.best_fitness_ever, best_return)
		if best_tick >= 0:
			self.best_time_ever = best_tick / self.tick_rate
		elapsed = time.perf_counter() - started
		self.last_speed = played * self.action_repeat / elapsed if elapsed else 0.0
		self.manual_stop = False

	def _play_episodes(self) -> None:
		"""Runs the whole population through every spawn point and fills `self.rewards`."""
		points = self.spawn_points()
		self.rewards.fill(0.0)
		self._rewards_banked.fill(0.0)
		max_ticks = self.max_ticks
		started = time.perf_counter()
		self._speed_time = started
		self._speed_ticks = self.total_ticks
		ticks_done = 0

		for checkpoint_index, (spawn_x, spawn_y) in enumerate(points):
			self._start_episode(spawn_x, spawn_y)
			tick = 0

			while tick < max_ticks and not self.episode_over():
				# One frame's worth of simulation at a time, so the render rate and the sim rate stay independent
				steps = max_ticks - tick
				if self.renderer is not None:
					self.render(checkpoint_index, len(points), tick)
					if self.paused or self.menu.open:
						# Nothing accumulates while paused, so unpausing does not fire a burst of catch up ticks
						self._tick_budget = 0.0
						continue
					self._tick_budget += self.ticks_per_frame
					steps = min(int(self._tick_budget), steps)
					self._tick_budget -= steps
					self._adapt_ticks_per_frame()

				played = self._simulate(tick, steps)
				tick += played
				ticks_done += played
				if self.renderer is not None:
					self._tick_budget += steps - played  # Ticks a whole window could not be paid for are owed back

			self._end_episode()
			self.record_best_time()
			self.should_skip_checkpoint = False

		self.best_fitness_ever = max(self.best_fitness_ever, float(self.rewards.max()))
		elapsed = time.perf_counter() - started
		self.last_speed = ticks_done / elapsed if elapsed else 0.0

	def record_best_time(self) -> None:
		"""Keeps the fastest run to the flag seen so far, in seconds."""
		times = self.world.win_tick[self.world.win & (self.world.win_tick >= 0)]
		if times.size == 0:
			return
		best = float(times.min()) / self.tick_rate
		self.best_time_ever = best if self.best_time_ever == 0.0 else min(self.best_time_ever, best)

	def episode_over(self) -> bool:
		if self.should_skip_checkpoint or self.manual_stop or self.restart_requested or self._map_requested is not None:
			return True
		# The device path answers from the last sampled alive mask, because reading one costs a full sync
		return not (self._anyone_alive if self.runner is not None else self.world.alive().any())

	def _start_episode(self, spawn_x: int, spawn_y: int) -> None:
		"""Puts the population back on a spawn point and clears everything an episode accumulates."""
		self.world.reset(spawn_x, spawn_y)
		self.position_history.clear()
		self._alive_checked_tick = 0
		self._anyone_alive = True

		if self.runner is not None and self.cuda_world is not None:
			self.cuda_world.reset(spawn_x, spawn_y)
			self.runner.start()
			return

		self.max_x_reached[:] = self.world.x
		self.max_x_tick.fill(0.0)
		self.ticks_stationary.fill(0)

	def _end_episode(self) -> None:
		"""Banks what the checkpoint paid: the per-tick rewards it accumulated plus its end of episode payout."""
		if self.runner is not None:
			self._sync_from_device()
		self.rewards += self.final_rewards()
		self._rewards_banked[:] = self.rewards

	def _sync_from_device(self) -> None:
		"""Reads the device state back into the numpy world and the trackers the fitness and the HUD use."""
		assert self.cuda_world is not None
		self.cuda_world.sync()
		if self.ppo_runner is not None:
			# What the HUD ranks agents on under PPO is the return of the episode each one is still playing
			np.copyto(self.rewards, self.ppo_runner.episode_return.cpu().numpy())
			np.copyto(self.max_x_reached, self.ppo_runner.max_x_reached.cpu().numpy())
			return

		assert self.runner is not None
		self.rewards[:] = self._rewards_banked + self.runner.rewards.cpu().numpy()
		np.copyto(self.max_x_reached, self.runner.max_x_reached.cpu().numpy())
		np.copyto(self.max_x_tick, self.runner.max_x_tick.cpu().numpy())

	def _simulate(self, tick: int, steps: int) -> int:
		"""Advances the simulation by up to `steps` ticks, returning how many were actually played."""
		if self.runner is not None:
			return self._simulate_windows(tick, steps)

		played = 0
		for _ in range(steps):
			self.simulate_tick(tick + played)
			played += 1
			if self.episode_over():
				break
		return played

	def _simulate_windows(self, tick: int, steps: int) -> int:
		"""
		The device path: whole action windows, one graph replay each, with nothing read back between them.

		Both periodic checks land on a window boundary rather than on the exact tick the numpy loop would
		have used, so a stuck agent can outlive its sentence by a tick and an episode where everyone is
		already dead runs to the end of the current second.
		"""
		assert self.runner is not None
		played = 0
		while played + self.action_repeat <= steps:
			self.runner.play_window()
			played += self.action_repeat
			self.total_ticks += self.action_repeat

			done = tick + played
			if done % self.stuck_check_ticks == 0:
				self.runner.check_positions()
			if done - self._alive_checked_tick >= ALIVE_CHECK_TICKS:
				self._alive_checked_tick = done
				self._anyone_alive = self.runner.check_alive()
				if self.episode_over():
					break
		return played

	def simulate_tick(self, tick: int) -> None:
		"""
		One simulation tick: observe and decide on the first tick of a window, then step, reward and record.

		Holding an action for `action_repeat` ticks is both cheaper and easier to learn from: a jump arc lasts
		about 42 ticks, so at one decision per tick no reachable discount factor reaches the far side of a gap.
		"""
		alive = self.world.alive()
		if tick % self.action_repeat == 0:
			self._actions = self.population.act(self.world.observe())

		np.copyto(self._previous_x, self.world.x)
		np.copyto(self._previous_y, self.world.y)
		self.world.step(self._actions, tick)
		self.add_continuous_rewards(alive, self._previous_x, self._previous_y, tick)

		self.check_agent_positions(tick)
		self.total_ticks += 1

	def _adapt_ticks_per_frame(self) -> None:
		"""
		In `--speed max`, walks the tick budget up while the framerate holds and backs off when it slips.

		The controller watches the measured framerate rather than the time a frame spends working: with
		vsync on, waiting for the next refresh is indistinguishable from working, so only the framerate
		itself says whether there is headroom left.
		"""
		if not self.auto_speed or self.renderer is None or self.paused or self.menu.open:
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

	def add_continuous_rewards(self, alive: NDArray[np.bool_], previous_x: NDArray[np.float64], previous_y: NDArray[np.float64], tick: int) -> None:
		"""Per-tick micro rewards, accumulated into `self.rewards` for the whole population at once."""
		# Every term is a masked add rather than a mask multiplied by its value, which halves the numpy calls
		world = self.world
		x_delta = world.x - previous_x
		rewards = self._tick_rewards
		rewards.fill(0.0)

		forward = x_delta > 0
		np.add(rewards, FORWARD_MOVEMENT_REWARD, out=rewards, where=forward)
		# Only a living player ever moves, so a step forward is already proof the agent was alive for it
		record = forward & (world.x > self.max_x_reached)
		np.add(rewards, NEW_MAX_POSITION_BONUS, out=rewards, where=record)
		np.copyto(self.max_x_tick, float(tick), where=record)
		np.copyto(self.max_x_reached, world.x, where=record)

		np.add(rewards, BACKWARD_MOVEMENT_PENALTY, out=rewards, where=x_delta < 0)

		# Counting up then zeroing by multiplication: two whole-array calls instead of two masked ones and an invert
		still = x_delta == 0
		np.add(self.ticks_stationary, still, out=self.ticks_stationary)
		np.multiply(self.ticks_stationary, still, out=self.ticks_stationary)
		# The counter is back to zero for anyone who moved, so being over the threshold already means standing still
		np.add(rewards, STATIONARY_PENALTY, out=rewards, where=self.ticks_stationary > STATIONARY_THRESHOLD)

		np.add(rewards, FALLING_PENALTY, out=rewards, where=(world.y - previous_y) > FALLING_THRESHOLD)
		# The dead score nothing, which is one masked accumulate instead of a multiply and an add
		np.add(self.rewards, rewards, out=self.rewards, where=alive)

	def speed_ratio(self, ticks: NDArray[np.float64]) -> NDArray[np.float64]:
		"""How much of the episode was still left after `ticks`, bent by `WIN_SPEED_EXPONENT`, in [0, 1]."""
		return np.clip(1.0 - ticks / self.max_ticks, 0.0, 1.0) ** WIN_SPEED_EXPONENT

	def final_rewards(self) -> NDArray[np.floating]:
		"""End of episode reward: win bonus, reward tile value, or distance travelled minus the death penalty."""
		world = self.world
		# Winners are ranked by the tick the flag was touched on: the whole episode is the scale, so there is a
		# gradient the entire way instead of a cliff, and the exponent makes a fast run worth beating further
		win_ticks = np.where(world.win_tick >= 0, world.win_tick.astype(np.float64), float(self.max_ticks))
		win_reward = world.x / DISTANCE_REWARD_DIVISOR + WIN_BASE_BONUS + WIN_SPEED_BONUS * self.speed_ratio(win_ticks)

		# Same shape for the rest, scaled by how far they got, so getting nowhere fast is worth nothing
		reached = np.clip(self.max_x_reached / (world.width * TILE_SIZE), 0.0, 1.0)
		progress = np.maximum(
			MIN_REWARD,
			self.max_x_reached / PROGRESS_REWARD_DIVISOR
			+ PROGRESS_SPEED_BONUS * reached * self.speed_ratio(self.max_x_tick)
			+ world.dead * DEATH_PENALTY,
		)

		# Coins are paid whatever the run ended as, so a detour that banks one is always worth something
		return np.where(world.win, win_reward, progress) + COIN_REWARD * world.coins

	def check_agent_positions(self, tick: int) -> None:
		"""
		Kills agents that are stuck in place or crawling backwards, once every `POSITION_CHECK_INTERVAL`.

		The schedule counts ticks rather than in-game seconds so that it lands on the same tick whether the
		episode is played one tick at a time or one action window at a time.
		"""
		if (tick + 1) % self.stuck_check_ticks:
			return

		self.position_history.append(self.world.x.copy())
		window = int(STUCK_CHECK_WINDOW / POSITION_CHECK_INTERVAL) + 1
		del self.position_history[:-window]

		alive = self.world.alive()
		if len(self.position_history) >= 2:
			self.world.kill(alive & (self.position_history[-1] == self.position_history[-2]))
		if len(self.position_history) >= 4:
			self.world.kill(alive & (self.world.x < self.position_history[0]))

	def training_rows(self) -> 'list[Row]':
		"""The knobs that shape the run, laid out for the HUD's training panel."""
		if self.ppo is not None:
			assert self.buffer is not None
			rows: 'list[Row]' = [
				('Envs', f'{self.population_size}'),
				('Rollout', f'{self.buffer.steps} x {self.population_size}'),
				('Entropy', f'{self.ppo.entropy_coef:.4f}'),
				('KL', f'{self.last_stats.get("approx_kl", 0.0):.4f}'),
				('Value Loss', f'{self.last_stats.get("value_loss", 0.0):.3f}'),
				('Spawn', f'{self.curriculum.mode} {self.curriculum.front}/{len(self.curriculum.spawns) - 1}'),
				('Update', f'{self.generation}'),
				('Network', 'x'.join(str(size) for size in self.population.hidden_sizes)),
				('Device', str(self.population.device)),
			]
		else:
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
			('Map', level_name(self.map_path)),
			('Coin', f'+{COIN_REWARD:g} x{self.world.coin_count}'),
			('Act Repeat', f'{self.action_repeat}'),
			('Speed', MAX_SPEED if self.auto_speed else f'x{self.speed:g}'),
		]

	def render(self, checkpoint_index: int, checkpoint_count: int, tick: int) -> None:
		"""Builds the overlay for the current tick and hands it to the renderer, which only lays it out."""
		assert self.renderer is not None
		from game.render import TRAINING_LEGEND, Fitness, Gauge, Hud, Legend, Panel

		self.renderer.poll_events(self.menu)
		self._sample_live_speed()
		if self.cuda_world is not None and not self._updating:
			self._sync_from_device()  # A frame is the one place the whole device state is worth reading back
		world = self.world
		best = int(np.argmax(np.where(world.alive(), self.rewards, -np.inf)))
		alive = int(world.alive().sum())
		fps = self.renderer.measured_fps()
		best_coins = int(world.coins.max())
		elite_count = self.elite_count if self.generation > 1 and self.ppo is None else 0
		random_count = min(RANDOM_AGENTS_COUNT, max(0, self.population_size - self.elite_count)) if elite_count else 0
		progress = (f'  {tick // max(1, self.action_repeat)}/{self.buffer.steps}' + ('  LEARNING' if self._updating else '')
			if self.buffer is not None
			else f'  ckpt {checkpoint_index + 1}/{checkpoint_count}' if checkpoint_count > 1 else '')

		run = Panel('Run', [
			('Generation' if self.ppo is None else 'Rollout', f'{self.generation}'),
			('Time', f'{tick / max(1, self.tick_rate):.1f}s' + progress + ('  PAUSED' if self.paused else '')),
			Gauge('Alive', f'{alive}/{world.count}', alive / max(1, world.count)),
			('Best', f'{float(self.rewards.max()):.1f}'),
			('Record', (f'{self.best_fitness_ever:.1f}' if self.best_fitness_ever > -float('inf') else '-') + (f'  {self.best_time_ever:.2f}s' if self.best_time_ever > 0 else '')),
			Gauge('Coins', f'{best_coins}/{world.coin_count}', best_coins / max(1, world.coin_count), COIN_COLOR),
			('Ticks/s', f'{self.live_speed:,.0f}  x{self.live_speed / self.tick_rate:.0f}'),
			Gauge('FPS', f'{fps:.0f}/{self.renderer.target_fps}', fps / max(1, self.renderer.target_fps)),
		])
		focus = Panel('Focus', [
			('Agent', f'#{best}' + (' elite' if best < elite_count else '')),
			('Fitness', f'{float(self.rewards[best]):.1f}'),
			('Position', f'{int(world.x[best])}, {int(world.y[best])}'),
			('Coins', f'{int(world.coins[best])}/{world.coin_count}'),
			('View', self.renderer.view_label()),
		])

		self.renderer.draw(best, self.rewards, Hud(
			left=[run, focus],
			right=[Panel('Training', self.training_rows())],
			legend=Legend(TRAINING_LEGEND, self.renderer.key_hints(), ramp=True),
			fitness=Fitness(self.rewards, self.fitness_history),
			menu=self.menu.page,
			elite_count=elite_count,
			random_count=random_count,
		))

	def evolve_generation(self) -> None:
		"""Improves the policy on what was just played: a PPO update, or selection and breeding."""
		if self.restart_requested:
			self._restart()
			return
		# The level is about to change, so what was just played belongs to the old one and is dropped
		if self._map_requested is not None:
			return
		if self.ppo is not None:
			self._update_policy()
			return

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
		self._save_weights()

	def _update_policy(self) -> None:
		"""One PPO update over the rollout just collected, then the same weight file every generation writes."""
		assert self.ppo is not None and self.ppo_runner is not None and self.buffer is not None
		if self._collected == 0:
			return
		statistics = self.ppo_runner.statistics()
		self._updating, self._drawn = True, False
		try:
			result = self.ppo.update(self.buffer, self.ppo_runner.last_value(), self._collected, self._draw_frame)
		finally:
			self._updating = False
		self.last_stats = {**statistics, **result}
		self.fitness_history.append(statistics['mean_return'])

		if self.curriculum.observe(int(statistics['front_episodes']), int(statistics['front_cleared'])):
			self.ppo_runner.apply_curriculum()
			self.ppo.restart_exploration()
			print(f'Curriculum: episodes now start from rung {self.curriculum.front} of {len(self.curriculum.spawns) - 1}, tile {self.curriculum.spawns[self.curriculum.front][0] // TILE_SIZE}')
		elif self.curriculum.stalled():
			self.ppo.restart_exploration(stalled=True)
			print(f'Stuck on rung {self.curriculum.front}, entropy bonus back up to {self.ppo.entropy_coef:.4f}')

		self.generation += 1
		self.manual_stop = False
		self._save_weights()

	def _draw_frame(self) -> None:
		"""
		Keeps the window alive through an update, which the simulation spends standing still.

		The world does not move during an update, so there is exactly one frame worth drawing and the rest of
		the epochs only pump the event queue. Drawing every one of them would be far worse than wasted: a frame
		reads the device state back, and every one of those reads blocks on the gradient kernels already queued
		in front of it, which serialises the whole update behind the window.
		"""
		if self.renderer is None:
			return
		if self._drawn:
			self.renderer.poll_events(self.menu)
			return
		self._drawn = True
		self.render(0, 1, self._collected * self.action_repeat)

	def _save_weights(self) -> None:
		WEIGHTS_FOLDER.mkdir(exist_ok=True)
		torch.save({
			'weights': self.population.state_dict(0),
			'hidden_sizes': self.population.hidden_sizes,
			'best_fitness': self.best_fitness_ever,
			'best_time': self.best_time_ever,
			'mutation_rate': self.mutation_rate,
			'mutation_strength': self.mutation_strength,
		}, WEIGHTS_FOLDER / f'generation_{self.generation}.pth')

	def load_latest_generation_weights(self) -> None:
		"""Loads the most recent weight file into every agent, then immediately breeds from it."""
		try:
			latest = max(
				int(path.stem.split('_')[1])
				for path in WEIGHTS_FOLDER.iterdir()
				if re.match(r'generation_\d+\.pth', path.name)
			)
			data = torch.load(WEIGHTS_FOLDER / f'generation_{latest}.pth', weights_only=True, map_location='cpu')
			weights = data['weights'] if isinstance(data, dict) and 'weights' in data else data
			if not isinstance(weights, dict):
				raise ValueError('invalid weights format')

			self.population.load_state_dict(weights)
			if isinstance(data, dict):
				self.best_fitness_ever = float(data.get('best_fitness', 0.0))
				self.best_time_ever = float(data.get('best_time', 0.0))
				self.mutation_rate = float(data.get('mutation_rate', self.mutation_rate))
				self.mutation_strength = float(data.get('mutation_strength', self.mutation_strength))

			self.generation = latest
			print(f'Loaded weights for generation {latest}')
			if self.ppo is not None:
				self.generation += 1
				return
			print(f'Mutation parameters: rate={self.mutation_rate:.4f}, strength={self.mutation_strength:.4f}')
			self.evolve_generation()
		# A file saved against another observation size fails the shape check as a RuntimeError
		except (FileNotFoundError, ValueError, KeyError, RuntimeError) as error:
			print(f'No usable weights found, starting with random weights: {error}')

	def run(self, generations: int = 0) -> bool:
		"""
		Trains until asked to stop, printing what every update scored.

		Returns whether the window is still open, so a shell can put its title screen back up rather than tear
		the process down. A headless run has no window to hand back and always answers False.
		"""
		ppo = self.ppo is not None
		if ppo and self.buffer is not None:
			print(f'--- PPO, {self.population_size} environments on {self.population.device}, '
				  f'{self.buffer.steps * self.population_size:,} transitions per update ---')
		else:
			print(f'--- Generation {self.generation}, {self.population_size} agents on {self.population.device}, '
				  f'mutation rate: {self.mutation_rate:.3f} - strength: {self.mutation_strength:.3f} ---')

		ticks = self.total_ticks
		started = time.perf_counter()
		try:
			while not self.to_title:
				self.play_agents()
				if self.to_title:
					break
				print(f"--- {'Update' if ppo else 'Generation'} {self.generation} ---")
				if ppo:
					self.evolve_generation()
					self._print_update()
				else:
					print(f'  Best: {self.rewards.max():.2f} | Avg: {self.rewards.mean():.2f} | Worst: {self.rewards.min():.2f}')
					self.evolve_generation()
				# Two numbers under PPO: what the simulation runs at, and what the wall clock sees once the
				# update in front of it is paid for, which is most of a cycle at these batch sizes
				elapsed = time.perf_counter() - started
				overall = (self.total_ticks - ticks) / elapsed if elapsed else 0.0
				ticks, started = self.total_ticks, time.perf_counter()
				print(f'  Speed: {self.last_speed:,.0f} ticks/s ({self.last_speed * self.population_size:,.0f} agent-steps/s)'
					  + (f' | {overall:,.0f} ticks/s counting the update' if ppo else ''))
				if generations and self.generation > generations:
					break
		except (KeyboardInterrupt, SystemExit):
			print('Stopped.')
			self.to_title = False
		return self.to_title

	def _print_update(self) -> None:
		"""What one PPO update scored, which is the only place the run's numbers are written down."""
		stats = self.last_stats
		if not stats:
			return
		record = f'{self.best_fitness_ever:.2f}' if self.best_fitness_ever > -float('inf') else '-'
		print(f"  Episodes: {stats['episodes']:,.0f} | Wins: {stats['wins']:,.0f} | Return: {stats['mean_return']:.2f} | Best: {record}"
			  + (f" | Fastest: {self.best_time_ever:.2f}s" if self.best_time_ever else ''))
		front = f"{stats['front_cleared']:,.0f}/{stats['front_episodes']:,.0f}"
		print(f"  Front: rung {self.curriculum.front} of {len(self.curriculum.spawns) - 1}, cleared {front} | Entropy: {stats['entropy']:.3f}")
		print(f"  Policy: {stats['policy_loss']:+.4f} | Value: {stats['value_loss']:.4f} | KL: {stats['approx_kl']:.4f} "
			  f"| Clipped: {stats['clip_fraction']:.1%} | Scale: {stats['return_scale']:.1f}")

	def quit(self) -> None:
		if self.renderer is not None:
			self.renderer.quit()

from collections.abc import Callable
from typing import Final

import torch
from torch import Tensor

from ai.population import Population
from ai.rewards import (
	BACKWARD_MOVEMENT_PENALTY, FALLING_PENALTY, FALLING_THRESHOLD, FORWARD_MOVEMENT_REWARD,
	NEW_MAX_POSITION_BONUS, STATIONARY_PENALTY, STATIONARY_THRESHOLD,
)
from game.world import OBSERVATION_SIZE
from game.world_cuda import POSITION_DTYPE, CudaWorld

STUCK_SNAPSHOTS: Final[int] = 4  # Position samples kept, so the oldest is the far end of the stuck window


def compiled(function: Callable[[], None]) -> Callable[[], None]:
	"""
	Returns `function` fused by inductor, or unchanged if this machine cannot compile.

	A tick is a long chain of tiny elementwise kernels, each one paying a fixed cost whatever the population
	size, so fusing them is worth an order of magnitude. Torch ships the compiler backend on linux and the
	`triton-windows` wheel provides it on windows; without either, eager still runs, only slower.
	"""
	candidate = torch.compile(function, dynamic=False)
	try:
		candidate()
	except Exception as error:  # A missing backend surfaces as whatever it failed on, so nothing narrower catches it
		print(f'Compiling the simulation is unavailable, falling back to eager mode: {error}')
		return function
	return candidate


class DeviceRunner:
	"""
	Plays whole action windows on the device: physics, per-tick rewards and the next decision, in one graph.

	Nothing crosses back to the host inside a window, which is what makes the graph capturable, so the
	trackers a reward needs live here as tensors rather than in the training loop. The loop reads them back
	between windows, when it renders or when a checkpoint ends.
	"""

	def __init__(self, world: CudaWorld, population: Population, action_repeat: int) -> None:
		self.world = world
		self.population = population
		self.action_repeat = action_repeat

		count, device = world.count, world.device
		self.actions = torch.zeros(count, device=device, dtype=torch.int64)
		self.observations = torch.zeros(count, 1, OBSERVATION_SIZE, device=device, dtype=population.dtype)

		self.rewards = torch.zeros(count, device=device, dtype=POSITION_DTYPE)
		self.max_x_reached = torch.zeros(count, device=device, dtype=POSITION_DTYPE)
		self.max_x_tick = torch.zeros(count, device=device, dtype=POSITION_DTYPE)
		self.ticks_stationary = torch.zeros(count, device=device, dtype=torch.int32)
		self._tick_rewards = torch.zeros(count, device=device, dtype=POSITION_DTYPE)
		self._previous_x = torch.zeros(count, device=device, dtype=POSITION_DTYPE)
		self._previous_y = torch.zeros(count, device=device, dtype=POSITION_DTYPE)
		self._history: list[Tensor] = []

		# Compiling and capturing both play real windows, so they happen here, on the state a reset throws away
		self._window = compiled(self._play_window)
		self._graph: torch.cuda.CUDAGraph | None = None
		self._capture()

	def start(self) -> None:
		"""Seeds the trackers from a freshly reset world and takes the decision its first window plays."""
		self.rewards.zero_()
		self.max_x_reached.copy_(self.world.x)
		self.max_x_tick.zero_()
		self.ticks_stationary.zero_()
		self._history.clear()
		self._decide()

	def _decide(self) -> None:
		self.world.observe(self.observations)
		self.actions.copy_(self.population.decide(self.observations))

	def _play_window(self) -> None:
		"""One action window: the held action is played for every tick of it, then the next one is chosen."""
		world = self.world
		for _ in range(self.action_repeat):
			alive = world.alive()
			self._previous_x.copy_(world.x)
			self._previous_y.copy_(world.y)
			world.step(self.actions)
			self._add_tick_rewards(alive)
		world.observe(self.observations)

	def _add_tick_rewards(self, alive: Tensor) -> None:
		"""Per-tick micro rewards, in the order and the precision the numpy loop pays them in."""
		world = self.world
		x_delta = world.x - self._previous_x
		rewards = self._tick_rewards
		rewards.zero_()

		forward = x_delta > 0
		rewards += forward.to(POSITION_DTYPE) * FORWARD_MOVEMENT_REWARD
		# Only a living player ever moves, so a step forward is already proof the agent was alive for it
		record = forward & (world.x > self.max_x_reached)
		rewards += record.to(POSITION_DTYPE) * NEW_MAX_POSITION_BONUS
		# The tick counter has already been stepped by the physics, so the record is stamped with the tick just played
		torch.where(record, (world.tick - 1).to(POSITION_DTYPE), self.max_x_tick, out=self.max_x_tick)
		torch.where(record, world.x, self.max_x_reached, out=self.max_x_reached)

		rewards += (x_delta < 0).to(POSITION_DTYPE) * BACKWARD_MOVEMENT_PENALTY

		still = x_delta == 0
		self.ticks_stationary += still
		self.ticks_stationary *= still
		rewards += (self.ticks_stationary > STATIONARY_THRESHOLD).to(POSITION_DTYPE) * STATIONARY_PENALTY

		rewards += ((world.y - self._previous_y) > FALLING_THRESHOLD).to(POSITION_DTYPE) * FALLING_PENALTY
		self.rewards += rewards * alive

	def _capture(self) -> None:
		"""Captures a whole window, the decision included, so playing one is a single launch."""
		warmup = torch.cuda.Stream()
		warmup.wait_stream(torch.cuda.current_stream())
		with torch.cuda.stream(warmup):
			for _ in range(3):
				self._window()
				self.actions.copy_(self.population.decide(self.observations))
		torch.cuda.current_stream().wait_stream(warmup)

		graph = torch.cuda.CUDAGraph()
		try:
			with torch.cuda.graph(graph):
				self._window()
				self.actions.copy_(self.population.decide(self.observations))
			self._graph = graph
		except RuntimeError as error:
			print(f'CUDA graph capture unavailable, falling back to eager mode: {error}')
			self._graph = None

	def play_window(self) -> None:
		"""Plays `action_repeat` ticks and decides the action the next window holds."""
		if self._graph is None:
			self._window()
			self._decide()
			return
		self._graph.replay()

	def check_positions(self) -> None:
		"""Kills agents that are stuck in place or crawling backwards, on the same schedule the loop asks for."""
		self._history.append(self.world.x.clone())
		del self._history[:-STUCK_SNAPSHOTS]

		alive = self.world.alive()
		if len(self._history) >= 2:
			self.world.kill(alive & (self._history[-1] == self._history[-2]))
		if len(self._history) >= STUCK_SNAPSHOTS:
			self.world.kill(alive & (self.world.x < self._history[0]))

	def anyone_alive(self) -> bool:
		"""The one host read a running episode needs, which is why the loop only takes it now and then."""
		return bool(self.world.alive().any())

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
BUCKET_FLOOR: Final[int] = 16  # Smallest action pass captured, under which a halving saves less than the extra graph costs


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


def captured(function: Callable[[], None], warmups: int) -> torch.cuda.CUDAGraph | None:
	"""Plays `function` `warmups` times on a side stream then captures it, or returns None if the capture fails."""
	warmup = torch.cuda.Stream()
	warmup.wait_stream(torch.cuda.current_stream())
	with torch.cuda.stream(warmup):
		for _ in range(warmups):
			function()
	torch.cuda.current_stream().wait_stream(warmup)

	graph = torch.cuda.CUDAGraph()
	try:
		with torch.cuda.graph(graph):
			function()
	except RuntimeError as error:
		print(f'CUDA graph capture unavailable, falling back to eager mode: {error}')
		return None
	return graph


class Shaping:
	"""The per-tick shaping terms and the trackers they read, paid in the order and the precision the numpy loop pays them in."""

	def __init__(self, world: CudaWorld) -> None:
		self.world = world
		count, device = world.count, world.device
		self.max_x_reached = torch.zeros(count, device=device, dtype=POSITION_DTYPE)
		self.max_x_tick = torch.zeros(count, device=device, dtype=POSITION_DTYPE)
		self.ticks_stationary = torch.zeros(count, device=device, dtype=torch.int32)
		self._previous_x = torch.zeros(count, device=device, dtype=POSITION_DTYPE)
		self._previous_y = torch.zeros(count, device=device, dtype=POSITION_DTYPE)

	def reset(self) -> None:
		"""Seeds the trackers from a freshly reset world."""
		self.max_x_reached.copy_(self.world.x)
		self.max_x_tick.zero_()
		self.ticks_stationary.zero_()

	def restart(self, mask: Tensor) -> None:
		"""Seeds the trackers of the players `mask` selects, once the world has put them back on a spawn point."""
		torch.where(mask, self.world.x, self.max_x_reached, out=self.max_x_reached)
		self.max_x_tick *= ~mask
		self.ticks_stationary *= ~mask

	def step(self, actions: Tensor) -> Tensor:
		"""Steps the world one tick and returns what it paid every player, the dead scoring nothing."""
		world = self.world
		alive = world.alive()
		self._previous_x.copy_(world.x)
		self._previous_y.copy_(world.y)
		world.step(actions)

		x_delta = world.x - self._previous_x
		forward = x_delta > 0
		rewards = forward.to(POSITION_DTYPE) * FORWARD_MOVEMENT_REWARD
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
		return rewards * alive


class DeviceRunner:
	"""
	Plays whole action windows on the device: physics, per-tick rewards and the next decision, in one graph.

	Nothing crosses back to the host inside a window, which is what makes the graph capturable, so the
	trackers a reward needs live here as tensors rather than in the training loop. The loop reads them back
	between windows, when it renders or when a checkpoint ends.

	The action pass is bound by reading every agent's weights, and most of a long episode is played by a
	handful of survivors. Death is final within an episode, so each alive check moves the pass to the smallest
	bucket of a halving ladder that still holds the living: their weights are copied once into `_weights`,
	living agents first, and every window after that reads only that prefix. `_slots` maps a slot of it back
	to its agent. Each bucket is its own captured graph, and the full one reads the population directly.
	"""

	def __init__(self, world: CudaWorld, population: Population, action_repeat: int) -> None:
		self.world = world
		self.population = population
		self.action_repeat = action_repeat

		count, device = world.count, world.device
		self.actions = torch.zeros(count, device=device, dtype=torch.int64)
		self.observations = torch.zeros(count, 1, OBSERVATION_SIZE, device=device, dtype=population.dtype)

		self.rewards = torch.zeros(count, device=device, dtype=POSITION_DTYPE)
		self.shaping = Shaping(world)
		self._history: list[Tensor] = []

		self.buckets = [count]
		while self.buckets[-1] // 2 >= BUCKET_FLOOR:
			self.buckets.append(self.buckets[-1] // 2)
		self.bucket = count
		self._slots = torch.arange(count, device=device)
		half = self.buckets[1] if len(self.buckets) > 1 else 0
		self._weights = {name: torch.zeros(half, *tensor.shape[1:], device=device, dtype=tensor.dtype) for name, tensor in population.weights.items()}
		self._biases = {name: torch.zeros(half, *tensor.shape[1:], device=device, dtype=tensor.dtype) for name, tensor in population.biases.items()}

		# Compiling and capturing both play real windows, so they happen here, on the state a reset throws away
		self._window = compiled(self._play_window)
		self._graphs = {bucket: captured(lambda bucket=bucket: self._play_and_decide(bucket), 3) for bucket in self.buckets}

	def start(self) -> None:
		"""Seeds the trackers from a freshly reset world and takes the decision its first window plays."""
		self.rewards.zero_()
		self.shaping.reset()
		self._history.clear()
		self.bucket = self.buckets[0]
		torch.arange(self.world.count, device=self.world.device, out=self._slots)
		self.world.observe(self.observations)
		self._decide(self.bucket)

	@torch.no_grad()
	def _decide(self, bucket: int) -> None:
		"""Picks the actions of the agents in the first `bucket` slots, leaving the rest, all dead, as they were."""
		if bucket == self.buckets[0]:
			actions = self.population.forward(self.observations).squeeze(1).argmax(dim=-1)
			self.actions.copy_(actions)
			return
		slots = self._slots[:bucket]
		weights = {name: tensor[:bucket] for name, tensor in self._weights.items()}
		biases = {name: tensor[:bucket] for name, tensor in self._biases.items()}
		logits = Population.forward_with(self.observations.index_select(0, slots), weights, biases)
		self.actions.index_copy_(0, slots, logits.squeeze(1).argmax(dim=-1))

	def _play_window(self) -> None:
		"""One action window: the held action is played for every tick of it, then the next one is chosen."""
		for _ in range(self.action_repeat):
			self.rewards += self.shaping.step(self.actions)
		self.world.observe(self.observations)

	def _play_and_decide(self, bucket: int) -> None:
		"""A whole window and the decision over `bucket` slots, which is what one captured graph covers."""
		self._window()
		self._decide(bucket)

	def play_window(self) -> None:
		"""Plays `action_repeat` ticks and decides the action the next window holds."""
		graph = self._graphs[self.bucket]
		if graph is None:
			self._play_and_decide(self.bucket)
			return
		graph.replay()

	def check_positions(self) -> None:
		"""Kills agents that are stuck in place or crawling backwards, on the same schedule the loop asks for."""
		self._history.append(self.world.x.clone())
		del self._history[:-STUCK_SNAPSHOTS]

		alive = self.world.alive()
		if len(self._history) >= 2:
			self.world.kill(alive & (self._history[-1] == self._history[-2]))
		if len(self._history) >= STUCK_SNAPSHOTS:
			self.world.kill(alive & (self.world.x < self._history[0]))

	def check_alive(self) -> bool:
		"""
		Whether anybody is still playing, moving the action pass down to the smallest bucket that holds them.

		This is the one host read a running episode needs, which is why the loop only takes it now and then,
		and the only point where the bucket can change.
		"""
		alive = self.world.alive()
		living = int(alive.sum())
		bucket = min((size for size in self.buckets if size >= living), default=self.bucket)
		if bucket < self.bucket:
			# A stable sort puts the living first in agent order, and the dead behind them only pad the bucket
			self._slots.copy_(torch.argsort(~alive, stable=True))
			slots = self._slots[:bucket]
			with torch.no_grad():
				for name, tensor in self.population.weights.items():
					torch.index_select(tensor, 0, slots, out=self._weights[name][:bucket])
				for name, tensor in self.population.biases.items():
					torch.index_select(tensor, 0, slots, out=self._biases[name][:bucket])
			self.bucket = bucket
		return living > 0

"""Rollout collection for PPO: one shared policy, one environment per agent, all of it on the device."""
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ai.device_runner import STUCK_SNAPSHOTS, compiled
from ai.population import HOST_DTYPE, Population
from ai.ppo import RolloutBuffer
from ai.rewards import (
	BACKWARD_MOVEMENT_PENALTY, COIN_REWARD, DEATH_PENALTY, FALLING_PENALTY, FALLING_THRESHOLD,
	FORWARD_MOVEMENT_REWARD, NEW_MAX_POSITION_BONUS, STATIONARY_PENALTY, STATIONARY_THRESHOLD, WIN_BASE_BONUS,
	WIN_SPEED_BONUS, WIN_SPEED_EXPONENT,
)
from game.world import OBSERVATION_SIZE
from game.world_cuda import POSITION_DTYPE, CudaWorld

NO_WIN: Final[int] = 1 << 30  # Stands in for "no run to the flag yet" in the record, above any reachable tick


class SpawnCurriculum:
	"""
	Which spawn point an environment restarts on, and how that choice moves over a run.

	Dropping an agent at the start of a level it has never finished means every episode ends in the same
	first few seconds, so nothing downstream is ever visited and nothing about it is ever learned. The
	curriculum runs the other way: episodes begin at the checkpoint nearest the flag, and the front moves
	one checkpoint back down the level each time the policy wins often enough from where it currently is.
	Spawns already behind the front keep a share of the episodes, so what was learned there is not forgotten.
	"""

	FRONT_SHARE: Final[float] = 0.6  # Episodes spent on the checkpoint being learned, the rest replay the easier ones
	PROMOTE_RATE: Final[float] = 0.6  # Win rate from the front that moves it one checkpoint back
	MIN_EPISODES: Final[int] = 200  # Episodes the front is judged on before it can move
	PATIENCE: Final[int] = 25  # Rollouts under `STALL_RATE` in a row before the front is called stuck
	STALL_RATE: Final[float] = 0.1  # Win rate from the front under which a rollout counts toward a stall

	def __init__(self, spawns: list[tuple[int, int]], mode: str) -> None:
		self.spawns = spawns
		self.mode = mode
		self.front = len(spawns) - 1 if mode == 'curriculum' else 0
		self.rate = 0.0
		self.seen = 0
		self.idle = 0

	def weights(self) -> NDArray[np.float64]:
		"""The probability of each spawn point, in map order."""
		weights = np.zeros(len(self.spawns), dtype=np.float64)
		if self.mode == 'start':
			weights[0] = 1.0
			return weights
		if self.mode == 'uniform':
			weights[:] = 1.0 / len(self.spawns)
			return weights

		behind = len(self.spawns) - 1 - self.front
		weights[self.front] = 1.0 if behind == 0 else self.FRONT_SHARE
		if behind:
			weights[self.front + 1:] = (1.0 - self.FRONT_SHARE) / behind
		return weights

	def observe(self, episodes: int, cleared: int) -> bool:
		"""
		Folds a rollout's episodes at the front in and returns whether the front moved down the level.

		The rate is a moving estimate rather than a total: the rollouts spent learning a rung all score zero,
		and averaging those in with the ones that follow would hold the front where it is long after the
		policy has outgrown it. A rollout weighs on the estimate in proportion to what it played.

		A rollout counts toward a stall while the rate sits under `STALL_RATE`, not only when nothing clears.
		A rung the policy has settled away from still lets the odd lucky episode through, and a count that any
		single clear resets never fires on it while the entropy bonus anneals the policy into the failure.
		"""
		if self.mode != 'curriculum' or self.front == 0 or episodes == 0:
			return False
		self.rate += (cleared / episodes - self.rate) * min(1.0, episodes / self.MIN_EPISODES)
		self.seen += episodes
		self.idle = 0 if self.rate >= self.STALL_RATE else self.idle + 1
		if self.seen < self.MIN_EPISODES or self.rate < self.PROMOTE_RATE:
			return False
		self.promote()
		return True

	def stalled(self) -> bool:
		"""Whether the front has gone `PATIENCE` rollouts without a single episode clearing it, and resets the count."""
		if self.idle < self.PATIENCE:
			return False
		self.idle = 0
		return True

	def promote(self) -> None:
		"""Moves the front one rung back down the level and starts judging it from scratch."""
		self.front = max(0, self.front - 1)
		self.rate, self.seen, self.idle = 0.0, 0, 0


class PPORunner:
	"""
	Plays one rollout step per replay: an action window, its reward, the reset it may trigger, the next decision.

	Everything an episode boundary needs is a tensor here, resets included, so a whole step stays inside one
	captured graph and the host reads nothing while a rollout fills. Environments run their own episodes:
	one that dies restarts on its own spawn point on the next step while the rest carry on with theirs.
	"""

	def __init__(
		self,
		world: CudaWorld,
		population: Population,
		buffer: RolloutBuffer,
		action_repeat: int,
		max_ticks: int,
		curriculum: SpawnCurriculum,
	) -> None:
		self.world = world
		self.population = population
		self.buffer = buffer
		self.action_repeat = action_repeat
		self.max_ticks = max_ticks
		self.curriculum = curriculum

		count, device = world.count, world.device
		spawns = curriculum.spawns
		self.spawn_x = torch.tensor([x for x, _ in spawns], device=device, dtype=POSITION_DTYPE)
		self.spawn_y = torch.tensor([y for _, y in spawns], device=device, dtype=POSITION_DTYPE)
		# Where an episode from each rung has to get to count as having cleared it: the next rung, or the flag
		cleared_x = [point[0] for point in spawns[1:]] + [world.world.goal_x]
		self.cleared_x = torch.tensor(cleared_x, device=device, dtype=POSITION_DTYPE)
		self.spawn_cdf = torch.zeros(len(spawns), device=device)
		self.spawn_index = torch.zeros(count, device=device, dtype=torch.int64)
		self.apply_curriculum()

		self.observations = torch.zeros(1, count, OBSERVATION_SIZE, device=device, dtype=HOST_DTYPE)
		self.actions = torch.zeros(1, count, device=device, dtype=torch.int64)
		self.log_probs = torch.zeros(1, count, device=device)
		self.values = torch.zeros(1, count, device=device)

		self.episode_return = torch.zeros(count, device=device)
		self.max_x_reached = torch.zeros(count, device=device, dtype=POSITION_DTYPE)
		self.max_x_tick = torch.zeros(count, device=device, dtype=POSITION_DTYPE)
		self.ticks_stationary = torch.zeros(count, device=device, dtype=torch.int32)
		self._previous_x = torch.zeros(count, device=device, dtype=POSITION_DTYPE)
		self._previous_y = torch.zeros(count, device=device, dtype=POSITION_DTYPE)
		self._previous_coins = torch.zeros(count, device=device, dtype=torch.int32)
		self._window_reward = torch.zeros(count, device=device)
		self._history: list[Tensor] = []

		# Every episode statistic is counted on the device and read back once a rollout, for the same reason
		self.finished = torch.zeros((), device=device, dtype=torch.int64)
		self.finished_wins = torch.zeros((), device=device, dtype=torch.int64)
		self.finished_return = torch.zeros((), device=device)
		self.front_episodes = torch.zeros(len(spawns), device=device, dtype=torch.int64)
		self.front_cleared = torch.zeros(len(spawns), device=device, dtype=torch.int64)
		self.judged = torch.zeros(count, device=device, dtype=torch.bool)  # Whether the running episode is already counted
		# Records, which outlive a rollout: the best episode ever returned and the fastest run to the flag
		self.best_return = torch.full((), -float('inf'), device=device)
		self.best_win_tick = torch.full((), NO_WIN, device=device, dtype=torch.int64)

		self.reset_all()
		self._compiled_step = compiled(self._step)
		self._graph: torch.cuda.CUDAGraph | None = None
		self._capture()
		self.reset_all()

	def apply_curriculum(self) -> None:
		"""Pushes the current spawn distribution to the device, as the cumulative table a step samples from."""
		weights = self.curriculum.weights()
		self.spawn_cdf.copy_(torch.from_numpy(np.cumsum(weights) / weights.sum()))

	def reset_all(self) -> None:
		"""Starts every environment on a freshly drawn spawn point and takes the decision its first step plays."""
		world = self.world
		self._draw_spawns(torch.ones(world.count, device=world.device, dtype=torch.bool))
		world.reset(0, 0)
		self._place(torch.ones(world.count, device=world.device, dtype=torch.bool))
		self.episode_return.zero_()
		self.max_x_reached.copy_(world.x)
		self.max_x_tick.zero_()
		self.ticks_stationary.zero_()
		self._previous_coins.zero_()
		self.judged.zero_()
		self._history.clear()
		self._decide()

	def _draw_spawns(self, mask: Tensor) -> None:
		"""Redraws a spawn point for the selected environments, from the curriculum's distribution."""
		draw = torch.rand(self.world.count, device=self.world.device)
		choice = torch.searchsorted(self.spawn_cdf, draw).clamp_max(len(self.curriculum.spawns) - 1)
		torch.where(mask, choice, self.spawn_index, out=self.spawn_index)

	def _place(self, mask: Tensor) -> None:
		self.world.reset_where(mask, self.spawn_x[self.spawn_index], self.spawn_y[self.spawn_index])

	def _decide(self) -> None:
		self.world.observe(self.observations)
		actions, log_probs, values = self.population.sample(self.observations.to(self.population.dtype))
		self.actions.copy_(actions)
		self.log_probs.copy_(log_probs)
		self.values.copy_(values)

	def _play_window(self) -> None:
		"""One action window: the held action played for every tick of it, and what those ticks are worth."""
		world = self.world
		self._window_reward.zero_()
		for _ in range(self.action_repeat):
			alive = world.alive()
			self._previous_x.copy_(world.x)
			self._previous_y.copy_(world.y)
			world.step(self.actions.squeeze(0))
			self._add_tick_rewards(alive)

	def _add_tick_rewards(self, alive: Tensor) -> None:
		"""
		Per-tick shaping, plus the coins banked on this tick.

		Evolution pays the coins at the end of the episode, because all it needs is a number to rank on.
		A policy gradient needs to know which decision earned them, so they are paid the tick they are taken.
		"""
		world = self.world
		x_delta = world.x - self._previous_x
		rewards = torch.zeros_like(self._window_reward)

		forward = x_delta > 0
		rewards += forward * FORWARD_MOVEMENT_REWARD
		record = forward & (world.x > self.max_x_reached)
		rewards += record * NEW_MAX_POSITION_BONUS
		torch.where(record, (world.tick - 1).to(POSITION_DTYPE), self.max_x_tick, out=self.max_x_tick)
		torch.where(record, world.x, self.max_x_reached, out=self.max_x_reached)

		rewards += (x_delta < 0) * BACKWARD_MOVEMENT_PENALTY

		still = x_delta == 0
		self.ticks_stationary += still
		self.ticks_stationary *= still
		rewards += (self.ticks_stationary > STATIONARY_THRESHOLD) * STATIONARY_PENALTY
		rewards += ((world.y - self._previous_y) > FALLING_THRESHOLD) * FALLING_PENALTY

		rewards += (world.coins - self._previous_coins) * COIN_REWARD
		self._previous_coins.copy_(world.coins)
		self._window_reward += rewards * alive

	def _finish_step(self) -> None:
		"""
		Closes the step: the terminal payout, the buffer write, the statistics and the restart of what ended.

		The end of episode reward is the flag bonus or the death penalty and nothing else. Distance is already
		paid tick by tick as the agent covers it, and paying it twice would put most of the return in a single
		terminal spike that no advantage estimate can spread back over the decisions that earned it.

		An episode that runs out its tick budget ends the same way a death does, bootstrap cut and all. The
		budget is part of what a run is scored on here rather than a limit imposed on top of it, so a state
		near the end of one really is worth less than the same state at the start.

		The curriculum judges an episode the moment it reaches the next rung, and at its end only if it never
		did. A policy that clears its rung carries on toward the flag, and judged at the end its success would
		reach the curriculum up to a whole episode late, while its failures, being short, arrive first.
		"""
		world = self.world
		won = world.win
		terminal = world.dead | won
		done = terminal | (world.tick >= self.max_ticks)

		speed = (1.0 - world.win_tick.to(POSITION_DTYPE) / self.max_ticks).clamp(0.0, 1.0) ** WIN_SPEED_EXPONENT
		reward = (self._window_reward + won * (WIN_BASE_BONUS + WIN_SPEED_BONUS * speed) + world.dead * DEATH_PENALTY).float()
		self.buffer.write_outcome(reward, done)

		self.episode_return += reward
		self.finished += done.sum()
		self.finished_wins += (done & won).sum()
		self.finished_return += (self.episode_return * done).sum()
		torch.maximum(self.best_return, torch.where(done, self.episode_return, self.best_return).max(), out=self.best_return)
		# Only a run that started where the level does counts as a time: one from a rung had less to cover
		full_run = done & won & (self.spawn_index == 0)
		torch.minimum(self.best_win_tick, torch.where(full_run, world.win_tick, self.best_win_tick).min(), out=self.best_win_tick)
		# Touching the flag clears whatever rung the run started on, whether or not it walked past the tile
		cleared = ~self.judged & (won | (self.max_x_reached >= self.cleared_x[self.spawn_index]))
		judged = cleared | (done & ~self.judged)
		self.front_episodes.scatter_add_(0, self.spawn_index, judged.to(torch.int64))
		self.front_cleared.scatter_add_(0, self.spawn_index, cleared.to(torch.int64))
		self.judged |= judged
		self.judged &= ~done
		self.episode_return *= ~done

		self._draw_spawns(done)
		self._place(done)
		torch.where(done, world.x, self.max_x_reached, out=self.max_x_reached)
		torch.where(done, torch.zeros_like(self.max_x_tick), self.max_x_tick, out=self.max_x_tick)
		self.ticks_stationary *= ~done
		self._previous_coins *= ~done

	def _step(self) -> None:
		"""
		A whole rollout step, which is exactly what the compile and the capture cover.

		It is compiled whole rather than just its physics: the episode bookkeeping, the observation, the sampling
		and the buffer writes would otherwise be a hundred tiny kernels, half of what a step costs.
		"""
		self.buffer.write_decision(self.observations, self.actions, self.log_probs, self.values)
		self._play_window()
		self._finish_step()
		self._decide()

	def _capture(self) -> None:
		"""Captures a rollout step so collecting one costs a single launch instead of a few hundred."""
		if self.world.device.type != 'cuda':
			return
		warmup = torch.cuda.Stream()
		warmup.wait_stream(torch.cuda.current_stream())
		with torch.cuda.stream(warmup):
			for _ in range(3):
				self._compiled_step()
		torch.cuda.current_stream().wait_stream(warmup)

		graph = torch.cuda.CUDAGraph()
		try:
			with torch.cuda.graph(graph):
				self._compiled_step()
			self._graph = graph
		except RuntimeError as error:
			print(f'CUDA graph capture unavailable, falling back to eager mode: {error}')
			self._graph = None

	def collect_step(self) -> None:
		"""Plays one decision's worth of every environment and stores it in the rollout buffer."""
		if self._graph is None:
			self._compiled_step()
			return
		self._graph.replay()

	def begin_rollout(self) -> None:
		self.buffer.rewind()
		self.finished.zero_()
		self.finished_wins.zero_()
		self.finished_return.zero_()
		self.front_episodes.zero_()
		self.front_cleared.zero_()

	def last_value(self) -> Tensor:
		"""The value of the observation the runner is holding, which is what GAE bootstraps the tail from."""
		return self.values.squeeze(0)

	def check_positions(self, stuck_check_ticks: int) -> None:
		"""
		Kills agents stuck in place or crawling backwards, ignoring the ones whose episode is too young to judge.

		An environment that restarted since the last snapshot is behind where it was through no fault of its
		policy, so the comparison only applies once its own tick counter has covered the whole window.
		"""
		world = self.world
		self._history.append(world.x.clone())
		del self._history[:-STUCK_SNAPSHOTS]

		alive = world.alive()
		if len(self._history) >= 2:
			world.kill(alive & (world.tick >= 2 * stuck_check_ticks) & (self._history[-1] == self._history[-2]))
		if len(self._history) >= STUCK_SNAPSHOTS:
			world.kill(alive & (world.tick >= STUCK_SNAPSHOTS * stuck_check_ticks) & (world.x < self._history[0]))

	def statistics(self) -> dict[str, float]:
		"""The rollout's episode counts, read back in one go now that it is over."""
		finished = int(self.finished)
		front = self.curriculum.front
		return {
			'episodes': finished,
			'wins': int(self.finished_wins),
			'mean_return': float(self.finished_return) / max(1, finished),
			'front_episodes': int(self.front_episodes[front]),
			'front_cleared': int(self.front_cleared[front]),
		}

	def records(self) -> tuple[float, int]:
		"""The best episode return ever collected and the fastest run to the flag, in ticks, or -1 for none."""
		best_tick = int(self.best_win_tick)
		return float(self.best_return), -1 if best_tick >= NO_WIN else best_tick

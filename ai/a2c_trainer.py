from typing import Final

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from ai.population import Population
from game.world import OBSERVATION_DTYPE, OBSERVATION_SIZE

DEFAULT_LEARNING_RATE: Final[float] = 0.003
DEFAULT_GAMMA: Final[float] = 0.95
DEFAULT_GAE_LAMBDA: Final[float] = 0.95  # Bias/variance knob of the advantage estimator, 1.0 is plain Monte Carlo
RETURN_SCALE_DECAY: Final[float] = 0.9  # Smoothing of the running return scale the rewards are divided by
MIN_RETURN_SCALE: Final[float] = 1e-3
DEFAULT_VALUE_LOSS_COEF: Final[float] = 0.3
DEFAULT_ENTROPY_COEF: Final[float] = 0.005
DEFAULT_MAX_GRAD_NORM: Final[float] = 0.5
DEFAULT_CHUNK_SIZE: Final[int] = 256  # Timesteps replayed per backward pass, caps the autograd graph size
ENTROPY_DECAY_START: Final[int] = 20  # Generation after which the entropy bonus starts decaying
ENTROPY_DECAY_RATE: Final[float] = 0.95
MIN_ENTROPY_COEF: Final[float] = 0.001


class RolloutBuffer:
	"""
	Stores a whole generation of transitions for every agent at once.

	The rollout itself runs without autograd; the trainer replays the stored observations in chunks to
	build the graph only when it updates. Observations share the world's half precision layout, so
	`next_slot` hands the world the exact row it should write into and the tick costs no extra copy.
	"""

	def __init__(self, capacity: int, size: int) -> None:
		self.capacity = capacity
		self.size = size
		self.observations = np.zeros((capacity, size, OBSERVATION_SIZE), dtype=OBSERVATION_DTYPE)
		self.actions = np.zeros((capacity, size), dtype=np.int64)
		self.rewards = np.zeros((capacity, size), dtype=np.float32)
		self.alive = np.zeros((capacity, size), dtype=np.bool_)
		self.done = np.zeros((capacity, size), dtype=np.bool_)
		self.length = 0
		self.episode_start = 0

	def next_slot(self) -> NDArray[np.float16] | None:
		"""The observation row this tick should be written into, or None once the buffer is full."""
		return None if self.length >= self.capacity else self.observations[self.length]

	def commit(self, actions: NDArray[np.int64], rewards: NDArray[np.float64], alive: NDArray[np.bool_]) -> None:
		"""Closes the transition whose observations were written into the row `next_slot` handed out."""
		if self.length >= self.capacity:
			return
		index = self.length
		self.actions[index] = actions
		self.rewards[index] = rewards
		self.alive[index] = alive
		self.done[index] = False
		self.length += 1

	def finish_episode(self, final_rewards: NDArray[np.float64]) -> None:
		"""
		Folds the end of episode reward into each agent's last recorded transition and marks it terminal.

		Without this the update only ever sees the per tick shaping: dying, winning and how far the agent
		got never reach the gradient, so the policy optimises a different objective than the selection does.
		The terminal flag is what stops the next episode's return from leaking backwards across the reset.
		"""
		span = self.alive[self.episode_start:self.length]
		if span.shape[0] == 0:
			return
		last = self.episode_start + span.shape[0] - 1 - np.argmax(span[::-1], axis=0)
		agents = np.flatnonzero(span.any(axis=0))
		self.rewards[last[agents], agents] += final_rewards[agents]
		self.done[last[agents], agents] = True
		self.episode_start = self.length

	def clear(self) -> None:
		self.length = 0
		self.episode_start = 0

	def steps(self) -> int:
		return int(self.alive[:self.length].sum())


class A2CTrainer:
	"""
	Advantage actor-critic over a batched Population.

	One Adam covers every agent: because an agent's weights only ever touch its own outputs, the gradient
	of the summed loss is exactly each agent's own gradient, so a single step trains the whole population
	independently. Gradients are clipped per agent, like one clip per model would be.
	"""

	def __init__(
		self,
		population: Population,
		learning_rate: float = DEFAULT_LEARNING_RATE,
		gamma: float = DEFAULT_GAMMA,
		gae_lambda: float = DEFAULT_GAE_LAMBDA,
		value_loss_coef: float = DEFAULT_VALUE_LOSS_COEF,
		entropy_coef: float = DEFAULT_ENTROPY_COEF,
		max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
		chunk_size: int = DEFAULT_CHUNK_SIZE,
	) -> None:
		self.population = population
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.gae_lambda = gae_lambda
		self.return_scale = 0.0  # Running scale of the returns, set from the first rollout
		self.value_loss_coef = value_loss_coef
		self.entropy_coef = entropy_coef
		self.max_grad_norm = max_grad_norm
		self.chunk_size = chunk_size
		self.optimizer = torch.optim.Adam(population.parameters(), lr=learning_rate)

	def reset_optimizer(self) -> None:
		"""Drops the Adam moments, which no longer match the agents after the population is bred."""
		self.optimizer = torch.optim.Adam(self.population.parameters(), lr=self.learning_rate)

	@torch.no_grad()
	def _values(self, buffer: RolloutBuffer) -> torch.Tensor:
		"""State values for the whole rollout, `(length, agents)`, needed before the advantages can be built."""
		device = self.population.device
		values = torch.zeros(buffer.length, self.population.size, device=device)
		for start in range(0, buffer.length, self.chunk_size):
			stop = min(start + self.chunk_size, buffer.length)
			observations = torch.from_numpy(buffer.observations[start:stop]).to(device).float().transpose(0, 1)
			values[start:stop] = self.population.forward(observations)[1].t()
		return values

	def _advantages(self, buffer: RolloutBuffer, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		GAE(lambda) advantages and their value targets, per agent.

		Rewards are divided by a running scale first: the terminal reward is worth hundreds while a tick of
		shaping is worth hundredths, and a critic regressing raw returns would never leave that gap behind.
		Bootstrapping is cut at every terminal step, so a checkpoint's return cannot bleed into the previous one.
		"""
		device = self.population.device
		length = buffer.length
		rewards = torch.from_numpy(buffer.rewards[:length]).to(device)
		mask = torch.from_numpy(buffer.alive[:length]).to(device).float()
		keep = (1.0 - torch.from_numpy(buffer.done[:length]).to(device).float()) * mask

		scale = self._update_return_scale(rewards * mask, keep)
		rewards = rewards * mask / scale

		advantages = torch.zeros_like(rewards)
		running = torch.zeros(buffer.size, device=device)
		next_value = torch.zeros(buffer.size, device=device)
		for step in range(length - 1, -1, -1):
			delta = rewards[step] + self.gamma * next_value * keep[step] - values[step]
			running = (delta + self.gamma * self.gae_lambda * keep[step] * running) * mask[step]
			advantages[step] = running
			next_value = values[step]
		return advantages, advantages + values, mask

	def _update_return_scale(self, rewards: torch.Tensor, keep: torch.Tensor) -> float:
		"""Tracks the standard deviation of the discounted return, the scale the rewards are measured against."""
		returns = torch.zeros_like(rewards)
		running = torch.zeros(rewards.shape[1], device=rewards.device)
		for step in range(rewards.shape[0] - 1, -1, -1):
			running = rewards[step] + self.gamma * running * keep[step]
			returns[step] = running
		observed = float(returns.std()) if returns.numel() > 1 else 0.0
		observed = max(observed, MIN_RETURN_SCALE)
		self.return_scale = observed if self.return_scale == 0.0 else self.return_scale * RETURN_SCALE_DECAY + observed * (1.0 - RETURN_SCALE_DECAY)
		return self.return_scale

	def train_step(self, buffer: RolloutBuffer) -> dict[str, float]:
		"""Replays the rollout in chunks, accumulates the A2C gradients and applies one optimizer step."""
		if buffer.length == 0:
			return {'avg_total_loss': 0.0, 'avg_actor_loss': 0.0, 'avg_critic_loss': 0.0, 'num_trained_agents': 0}

		device = self.population.device
		values = self._values(buffer)
		advantages, returns, mask = self._advantages(buffer, values)
		total_steps = mask.sum().clamp(min=1.0)
		trained_agents = int((mask.sum(dim=0) > 0).sum().item())

		# Normalised over the whole batch rather than per agent: a stuck agent has near constant returns, and
		# dividing by its own vanishing spread turns pure noise into a full strength gradient
		mean = (advantages * mask).sum() / total_steps
		spread = ((((advantages - mean) * mask) ** 2).sum() / total_steps).sqrt()
		advantages = (advantages - mean) / (spread + 1e-8)

		self.optimizer.zero_grad(set_to_none=True)
		totals = torch.zeros(3, device=device)

		for start in range(0, buffer.length, self.chunk_size):
			stop = min(start + self.chunk_size, buffer.length)
			observations = torch.from_numpy(buffer.observations[start:stop]).to(device).float().transpose(0, 1)
			actions = torch.from_numpy(buffer.actions[start:stop]).to(device).t()
			chunk_mask = mask[start:stop].t()
			chunk_returns = returns[start:stop].t()
			chunk_advantages = advantages[start:stop].t()

			logits, chunk_values = self.population.forward(observations)
			log_probs = F.log_softmax(logits, dim=-1)
			probs = log_probs.exp()
			taken = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

			actor_loss = -(taken * chunk_advantages * chunk_mask).sum() / total_steps
			entropy = -((probs * log_probs).sum(dim=-1) * chunk_mask).sum() / total_steps
			critic_loss = (((chunk_values - chunk_returns) ** 2) * chunk_mask).sum() / total_steps
			loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
			loss.backward()

			totals += torch.stack([loss.detach(), actor_loss.detach(), critic_loss.detach()])

		self._clip_gradients()
		self.optimizer.step()
		buffer.clear()

		return {
			'avg_total_loss': float(totals[0]),
			'avg_actor_loss': float(totals[1]),
			'avg_critic_loss': float(totals[2]),
			'num_trained_agents': trained_agents,
		}

	def _clip_gradients(self) -> None:
		"""Clips each agent's gradient norm separately, as one clip_grad_norm_ per model would."""
		parameters = [tensor for tensor in self.population.parameters() if tensor.grad is not None]
		if not parameters:
			return
		squared = torch.zeros(self.population.size, device=self.population.device)
		for tensor in parameters:
			squared += tensor.grad.flatten(1).pow(2).sum(dim=1)
		scale = (self.max_grad_norm / (squared.sqrt() + 1e-6)).clamp(max=1.0)
		for tensor in parameters:
			tensor.grad *= scale.view(-1, *([1] * (tensor.dim() - 1)))

	def decay_entropy(self, generation: int) -> None:
		"""Anneals the entropy bonus once the population has had time to explore."""
		if generation <= ENTROPY_DECAY_START:
			return
		self.entropy_coef = max(MIN_ENTROPY_COEF, DEFAULT_ENTROPY_COEF * (ENTROPY_DECAY_RATE ** (generation - ENTROPY_DECAY_START)))

	def adjust_learning_rate(self, learning_rate: float) -> None:
		self.learning_rate = learning_rate
		for group in self.optimizer.param_groups:
			group['lr'] = learning_rate

	def get_training_summary(self) -> str:
		return (
			f"A2C Training Configuration:\n"
			f"---------------------------\n"
			f"Number of Agents: {self.population.size}\n"
			f"Learning Rate: {self.learning_rate}\n"
			f"Discount Factor (gamma): {self.gamma}\n"
			f"GAE Lambda: {self.gae_lambda}\n"
			f"Value Loss Coefficient: {self.value_loss_coef}\n"
			f"Entropy Coefficient: {self.entropy_coef}\n"
			f"Device: {self.population.device}"
		)

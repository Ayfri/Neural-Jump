from typing import Final

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from ai.population import Population
from game.world import OBSERVATION_DTYPE, OBSERVATION_SIZE

DEFAULT_LEARNING_RATE: Final[float] = 0.003
DEFAULT_GAMMA: Final[float] = 0.95
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
		self.length = 0

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
		self.length += 1

	def clear(self) -> None:
		self.length = 0

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
		value_loss_coef: float = DEFAULT_VALUE_LOSS_COEF,
		entropy_coef: float = DEFAULT_ENTROPY_COEF,
		max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
		chunk_size: int = DEFAULT_CHUNK_SIZE,
	) -> None:
		self.population = population
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.value_loss_coef = value_loss_coef
		self.entropy_coef = entropy_coef
		self.max_grad_norm = max_grad_norm
		self.chunk_size = chunk_size
		self.optimizer = torch.optim.Adam(population.parameters(), lr=learning_rate)

	def reset_optimizer(self) -> None:
		"""Drops the Adam moments, which no longer match the agents after the population is bred."""
		self.optimizer = torch.optim.Adam(self.population.parameters(), lr=self.learning_rate)

	def _returns(self, buffer: RolloutBuffer) -> tuple[torch.Tensor, torch.Tensor]:
		"""Discounted returns per agent, normalised over the steps the agent was actually alive."""
		device = self.population.device
		length = buffer.length
		rewards = torch.from_numpy(buffer.rewards[:length]).to(device)
		mask = torch.from_numpy(buffer.alive[:length]).to(device).float()
		rewards = rewards * mask

		returns = torch.zeros_like(rewards)
		running = torch.zeros(buffer.size, device=device)
		for step in range(length - 1, -1, -1):
			running = rewards[step] + self.gamma * running
			returns[step] = running

		counts = mask.sum(dim=0).clamp(min=1.0)
		mean = (returns * mask).sum(dim=0) / counts
		variance = (((returns - mean) * mask) ** 2).sum(dim=0) / counts
		return (returns - mean) / (variance.sqrt() + 1e-8), mask

	def train_step(self, buffer: RolloutBuffer) -> dict[str, float]:
		"""Replays the rollout in chunks, accumulates the A2C gradients and applies one optimizer step."""
		if buffer.length == 0:
			return {'avg_total_loss': 0.0, 'avg_actor_loss': 0.0, 'avg_critic_loss': 0.0, 'num_trained_agents': 0}

		device = self.population.device
		returns, mask = self._returns(buffer)
		total_steps = mask.sum().clamp(min=1.0)
		trained_agents = int((mask.sum(dim=0) > 0).sum().item())

		self.optimizer.zero_grad(set_to_none=True)
		totals = torch.zeros(3, device=device)

		for start in range(0, buffer.length, self.chunk_size):
			stop = min(start + self.chunk_size, buffer.length)
			observations = torch.from_numpy(buffer.observations[start:stop]).to(device).float().transpose(0, 1)
			actions = torch.from_numpy(buffer.actions[start:stop]).to(device).t()
			chunk_mask = mask[start:stop].t()
			chunk_returns = returns[start:stop].t()

			logits, values = self.population.forward(observations)
			log_probs = F.log_softmax(logits, dim=-1)
			probs = log_probs.exp()
			taken = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

			advantages = chunk_returns - values.detach()
			actor_loss = -(taken * advantages * chunk_mask).sum() / total_steps
			entropy = -((probs * log_probs).sum(dim=-1) * chunk_mask).sum() / total_steps
			critic_loss = (((values - chunk_returns) ** 2) * chunk_mask).sum() / total_steps
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
			f"Value Loss Coefficient: {self.value_loss_coef}\n"
			f"Entropy Coefficient: {self.entropy_coef}\n"
			f"Device: {self.population.device}"
		)

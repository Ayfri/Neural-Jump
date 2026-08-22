from typing import Final

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from ai.population import Population
from game.world import OBSERVATION_DTYPE, OBSERVATION_SIZE

DEFAULT_LEARNING_RATE: Final[float] = 3e-4
DEFAULT_GAMMA: Final[float] = 0.98  # Per decision, so with an action repeat of 2 at 90 ticks/s the horizon is ~1.1s
DEFAULT_GAE_LAMBDA: Final[float] = 0.95  # Bias/variance knob of the advantage estimator, 1.0 is plain Monte Carlo
MIN_RETURN_SCALE: Final[float] = 1e-3
DEFAULT_VALUE_LOSS_COEF: Final[float] = 0.3
DEFAULT_ENTROPY_COEF: Final[float] = 0.005  # Held low on purpose: a fuzzy policy makes the measured fitness a lottery
DEFAULT_CLIP_RANGE: Final[float] = 0.2  # PPO trust region, the standard value across implementations
DEFAULT_EPOCHS: Final[int] = 4  # Passes over one rollout, what the clipped objective buys over plain A2C
DEFAULT_MAX_GRAD_NORM: Final[float] = 0.5
DEFAULT_CHUNK_SIZE: Final[int] = 256  # Timesteps replayed per backward pass, caps the autograd graph size
ENTROPY_DECAY_START: Final[int] = 10  # Generation after which the entropy bonus starts decaying
ENTROPY_DECAY_RATE: Final[float] = 0.95
MIN_ENTROPY_COEF: Final[float] = 0.002


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
	Clipped policy optimisation (PPO) over a batched Population.

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
		clip_range: float = DEFAULT_CLIP_RANGE,
		epochs: int = DEFAULT_EPOCHS,
		max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
		chunk_size: int = DEFAULT_CHUNK_SIZE,
	) -> None:
		self.population = population
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.gae_lambda = gae_lambda
		self.return_scale = 1.0  # Scale the last rollout's rewards were divided by, kept for the HUD
		self.value_loss_coef = value_loss_coef
		self.entropy_coef = entropy_coef
		self.clip_range = clip_range
		self.epochs = epochs
		self.max_grad_norm = max_grad_norm
		self.chunk_size = chunk_size
		self.optimizer = torch.optim.Adam(population.parameters(), lr=learning_rate)

	def reset_optimizer(self) -> None:
		"""Drops the Adam moments, which no longer match the agents after the population is bred."""
		self.optimizer = torch.optim.Adam(self.population.parameters(), lr=self.learning_rate)

	@torch.no_grad()
	def _evaluate(self, buffer: RolloutBuffer) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		The behaviour policy's values and log probabilities over the whole rollout, `(length, agents)`.

		Recomputed here rather than stored during the rollout: the weights have not moved since, so this is
		the same policy that acted, and the sampling path stays a single CUDA graph replay per tick.
		"""
		device = self.population.device
		values = torch.zeros(buffer.length, self.population.size, device=device)
		log_probs = torch.zeros(buffer.length, self.population.size, device=device)
		for start in range(0, buffer.length, self.chunk_size):
			stop = min(start + self.chunk_size, buffer.length)
			observations = torch.from_numpy(buffer.observations[start:stop]).to(device).float().transpose(0, 1)
			actions = torch.from_numpy(buffer.actions[start:stop]).to(device).t()
			logits, chunk_values = self.population.forward(observations)
			taken = F.log_softmax(logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
			values[start:stop] = chunk_values.t()
			log_probs[start:stop] = taken.t()
		return values, log_probs

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

		rewards = self._rescale(rewards * mask, keep)

		advantages = torch.zeros_like(rewards)
		running = torch.zeros(buffer.size, device=device)
		next_value = torch.zeros(buffer.size, device=device)
		for step in range(length - 1, -1, -1):
			delta = rewards[step] + self.gamma * next_value * keep[step] - values[step]
			running = (delta + self.gamma * self.gae_lambda * keep[step] * running) * mask[step]
			advantages[step] = running
			next_value = values[step]
		return advantages, advantages + values, mask

	def _rescale(self, rewards: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
		"""
		Divides the rewards by the spread of their own discounted return.

		The scale is taken from the rollout at hand rather than carried across generations: the reward a
		generation collects grows as the population gets further, and a lagging scale leaves the critic
		chasing a target that drifts out from under it every update.
		"""
		returns = torch.zeros_like(rewards)
		running = torch.zeros(rewards.shape[1], device=rewards.device)
		for step in range(rewards.shape[0] - 1, -1, -1):
			running = rewards[step] + self.gamma * running * keep[step]
			returns[step] = running
		self.return_scale = max(float(returns.std()) if returns.numel() > 1 else 0.0, MIN_RETURN_SCALE)
		return rewards / self.return_scale

	def train_step(self, buffer: RolloutBuffer) -> dict[str, float]:
		"""
		Runs several clipped policy iterations over the rollout, one optimizer step per minibatch.

		A single step per rollout is what kept the critic pinned at its initial loss: with fresh Adam moments
		every generation, one step is `lr * sign(gradient)` on every weight and nothing converges. The PPO
		ratio is what makes the extra passes legal, since the data goes stale as soon as the policy moves.
		"""
		if buffer.length == 0:
			return {'avg_total_loss': 0.0, 'avg_actor_loss': 0.0, 'avg_critic_loss': 0.0, 'avg_entropy': 0.0, 'approx_kl': 0.0, 'return_scale': self.return_scale, 'num_trained_agents': 0}

		device = self.population.device
		values, old_log_probs = self._evaluate(buffer)
		advantages, returns, mask = self._advantages(buffer, values)
		total_steps = mask.sum().clamp(min=1.0)
		trained_agents = int((mask.sum(dim=0) > 0).sum().item())

		# Normalised over the whole batch rather than per agent: a stuck agent has near constant returns, and
		# dividing by its own vanishing spread turns pure noise into a full strength gradient
		mean = (advantages * mask).sum() / total_steps
		spread = ((((advantages - mean) * mask) ** 2).sum() / total_steps).sqrt()
		advantages = (advantages - mean) / (spread + 1e-8)

		starts = list(range(0, buffer.length, self.chunk_size))
		totals = torch.zeros(5, device=device)
		updates = 0

		for _ in range(self.epochs):
			for index in torch.randperm(len(starts)).tolist():
				start = starts[index]
				stop = min(start + self.chunk_size, buffer.length)
				observations = torch.from_numpy(buffer.observations[start:stop]).to(device).float().transpose(0, 1)
				actions = torch.from_numpy(buffer.actions[start:stop]).to(device).t()
				chunk_mask = mask[start:stop].t()
				chunk_steps = chunk_mask.sum().clamp(min=1.0)
				chunk_returns = returns[start:stop].t()
				chunk_advantages = advantages[start:stop].t()
				chunk_old = old_log_probs[start:stop].t()

				logits, chunk_values = self.population.forward(observations)
				log_probs = F.log_softmax(logits, dim=-1)
				probs = log_probs.exp()
				taken = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

				ratio = (taken - chunk_old).exp()
				clipped = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range)
				surrogate = torch.min(ratio * chunk_advantages, clipped * chunk_advantages)
				actor_loss = -(surrogate * chunk_mask).sum() / chunk_steps
				entropy = -((probs * log_probs).sum(dim=-1) * chunk_mask).sum() / chunk_steps
				critic_loss = (((chunk_values - chunk_returns) ** 2) * chunk_mask).sum() / chunk_steps
				loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

				self.optimizer.zero_grad(set_to_none=True)
				loss.backward()
				self._clip_gradients()
				self.optimizer.step()

				with torch.no_grad():
					# Mean entropy and the usual PPO staleness probe, the two numbers that say whether the
					# extra epochs are still learning or already off policy
					divergence = ((chunk_old - taken) * chunk_mask).sum() / chunk_steps
				totals += torch.stack([loss.detach(), actor_loss.detach(), critic_loss.detach(), entropy.detach(), divergence])
				updates += 1

		buffer.clear()
		totals /= max(1, updates)
		return {
			'avg_total_loss': float(totals[0]),
			'avg_actor_loss': float(totals[1]),
			'avg_critic_loss': float(totals[2]),
			'avg_entropy': float(totals[3]),
			'approx_kl': float(totals[4]),
			'return_scale': self.return_scale,
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

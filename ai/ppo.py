"""Clipped policy optimisation over the batched world: the rollout buffer, the return scaler and the update."""
from collections.abc import Callable
from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor

from ai.device_runner import captured
from ai.population import HOST_DTYPE, Population
from game.world import OBSERVATION_SIZE

DEFAULT_ROLLOUT_STEPS: Final[int] = 256  # Decisions per environment between two updates
DEFAULT_LEARNING_RATE: Final[float] = 3e-4
DEFAULT_GAMMA: Final[float] = 0.999  # Per decision: at 45 decisions a second that is a horizon of about 22 seconds
DEFAULT_GAE_LAMBDA: Final[float] = 0.95
DEFAULT_CLIP_RANGE: Final[float] = 0.2
DEFAULT_EPOCHS: Final[int] = 4  # Passes over one rollout, cut short by the KL check when the data goes stale
DEFAULT_MINIBATCHES: Final[int] = 8
DEFAULT_VALUE_COEF: Final[float] = 0.5
DEFAULT_ENTROPY_COEF: Final[float] = 0.01
DEFAULT_ENTROPY_FINAL: Final[float] = 0.001
DEFAULT_ENTROPY_UPDATES: Final[int] = 100  # Updates on one rung the entropy bonus takes to reach its floor
DEFAULT_MAX_GRAD_NORM: Final[float] = 0.5
STALL_ENTROPY_BOOST: Final[float] = 2.0  # Factor the bonus is multiplied by every time a rung refuses to fall
MAX_ENTROPY_COEF: Final[float] = 0.08  # Ceiling on that, above which the policy is noise and learns nothing
DEFAULT_TARGET_KL: Final[float] = 0.02  # Stops the extra epochs once the policy has moved far enough off the data
ADAM_EPS: Final[float] = 1e-5  # Torch defaults to 1e-8, which is small enough to make PPO's updates jumpy
RETURN_SCALE_FLOOR: Final[float] = 1e-4


class RolloutBuffer:
	"""
	One rollout of `steps` decisions for every environment, kept on the device.

	The write cursor is a device tensor rather than a Python integer, so a whole step, buffer writes
	included, sits inside the captured graph the runner replays: collecting a rollout never touches the host.
	Observations keep the world's half precision layout, which is most of the buffer's bytes.
	"""

	def __init__(self, steps: int, envs: int, device: torch.device) -> None:
		self.steps = steps
		self.envs = envs
		self.observations = torch.zeros(steps, envs, OBSERVATION_SIZE, device=device, dtype=HOST_DTYPE)
		self.actions = torch.zeros(steps, envs, device=device, dtype=torch.int64)
		self.log_probs = torch.zeros(steps, envs, device=device)
		self.values = torch.zeros(steps, envs, device=device)
		self.rewards = torch.zeros(steps, envs, device=device)
		self.dones = torch.zeros(steps, envs, device=device, dtype=torch.bool)
		self.advantages = torch.zeros(steps, envs, device=device)
		self.returns = torch.zeros(steps, envs, device=device)
		self.cursor = torch.zeros(1, device=device, dtype=torch.int64)

	def rewind(self) -> None:
		self.cursor.zero_()

	def write_decision(self, observations: Tensor, actions: Tensor, log_probs: Tensor, values: Tensor) -> None:
		"""Stores the decision a step is about to play, all four tensors shaped `(1, envs, ...)`."""
		self.observations.index_copy_(0, self.cursor, observations.to(HOST_DTYPE))
		self.actions.index_copy_(0, self.cursor, actions)
		self.log_probs.index_copy_(0, self.cursor, log_probs)
		self.values.index_copy_(0, self.cursor, values)

	def write_outcome(self, rewards: Tensor, dones: Tensor) -> None:
		"""Closes the step the last `write_decision` opened and moves the cursor on, wrapping at the end."""
		self.rewards.index_copy_(0, self.cursor, rewards.unsqueeze(0))
		self.dones.index_copy_(0, self.cursor, dones.unsqueeze(0))
		self.cursor += 1
		self.cursor.remainder_(self.steps)


class ReturnScaler:
	"""
	Divides the rewards by the running spread of their own discounted return.

	Touching the flag is worth a thousand ticks of shaping, and a critic regressing that raw would spend the
	whole run chasing the spike instead of the gradient that leads to it. The estimate carries across
	rollouts, so an episode that spans several of them is scaled by one consistent number. Every piece of it
	is updated in place, so the scan that feeds it can be replayed from a captured graph.
	"""

	def __init__(self, envs: int, gamma: float, device: torch.device) -> None:
		self.gamma = gamma
		self._running = torch.zeros(envs, device=device)
		self._mean = torch.zeros((), device=device)
		self._var = torch.ones((), device=device)
		self._count = torch.full((), 1e-4, device=device)

	@property
	def scale(self) -> Tensor:
		return self._var.sqrt().clamp_min(RETURN_SCALE_FLOOR)

	def rescale(self, rewards: Tensor, dones: Tensor) -> Tensor:
		"""
		Updates the estimate from this rollout's returns and returns the rewards divided by it.

		The scan itself is sequential, but everything that does not depend on the step before it is lifted
		out of the loop: a whole-rollout call costs one kernel where a per-step one costs the length of the
		rollout, and this loop used to be a fifth of what an update spent.
		"""
		keep = (~dones).to(rewards.dtype)
		# What the running return is multiplied by on the way into each step: the discount, and zero if the
		# step before it ended the episode. `_running` is carried already reset, so the first step only discounts
		carry = torch.cat((torch.full_like(keep[:1], 1.0), keep[:-1])) * self.gamma

		returns = torch.empty_like(rewards)
		running = self._running
		for step in range(rewards.shape[0]):
			running = torch.addcmul(rewards[step], running, carry[step])
			returns[step] = running
		self._running.copy_(running * keep[-1])
		self._absorb(returns)
		return rewards / self.scale

	def _absorb(self, returns: Tensor) -> None:
		"""Merges a batch's mean and variance into the running ones, the parallel form of Welford's update."""
		count = float(returns.numel())
		mean, var = returns.mean(), returns.var(unbiased=False)
		delta = mean - self._mean
		total = self._count + count
		self._var.copy_((self._var * self._count + var * count + delta.square() * self._count * count / total) / total)
		self._mean.add_(delta * (count / total))
		self._count.copy_(total)


class CapturedCall:
	"""
	Runs a function eagerly the first time, then captures it and replays the capture on every later call.

	The first call is real work that doubles as the warmup a capture needs, so nothing is played twice and no
	state has to be saved around it. Everything the function touches has to live at a fixed address, since the
	graph replays the addresses it saw. Off CUDA, or if the capture fails, it keeps running eagerly.
	"""

	def __init__(self, function: Callable[[], None], device: torch.device) -> None:
		self.function = function
		self.capturable = device.type == 'cuda'
		self.graph: torch.cuda.CUDAGraph | None = None

	def __call__(self) -> None:
		if self.graph is not None:
			self.graph.replay()
		elif not self.capturable:
			self.function()
		else:
			self.graph = captured(self.function, 1)
			self.capturable = self.graph is not None


class PPOTrainer:
	"""
	The update half of PPO: GAE over the rollout, then clipped epochs of minibatch ascent on one policy.

	The whole population plays the same network here, so a rollout of `steps` decisions across `envs`
	environments is one batch of `steps * envs` transitions rather than one trajectory per agent. That is
	the entire reason this reaches further than the genetic path on a long level.
	"""

	def __init__(
		self,
		population: Population,
		envs: int,
		learning_rate: float = DEFAULT_LEARNING_RATE,
		gamma: float = DEFAULT_GAMMA,
		gae_lambda: float = DEFAULT_GAE_LAMBDA,
		clip_range: float = DEFAULT_CLIP_RANGE,
		epochs: int = DEFAULT_EPOCHS,
		minibatches: int = DEFAULT_MINIBATCHES,
		value_coef: float = DEFAULT_VALUE_COEF,
		entropy_coef: float = DEFAULT_ENTROPY_COEF,
		max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
		target_kl: float = DEFAULT_TARGET_KL,
	) -> None:
		self.population = population
		self.gamma = gamma
		self.gae_lambda = gae_lambda
		self.clip_range = clip_range
		self.epochs = epochs
		self.minibatches = minibatches
		self.value_coef = value_coef
		self.entropy_base = entropy_coef  # What the command line asked for, which a promotion goes back to
		self.entropy_start = entropy_coef  # What the current rung started on, which a stall pushes up
		self.entropy_coef = entropy_coef
		self.max_grad_norm = max_grad_norm
		self.target_kl = target_kl
		self.updates = 0
		self.rung_updates = 0  # Updates since the curriculum last moved, which is what the entropy bonus anneals on
		device = population.device
		cuda = device.type == 'cuda'
		# Capturable keeps Adam's step count on the device, which is what lets a whole epoch sit in one graph
		self.optimizer = torch.optim.Adam(population.parameters(), lr=learning_rate, eps=ADAM_EPS, fused=cuda, capturable=cuda)
		self.scaler = ReturnScaler(envs, gamma, device)
		# Inductor fuses the trunk's norms and activations into fewer passes over them, which is what a
		# minibatch is bound by, and is worth about a third of one
		self._compiled = torch.compile(population.evaluate, dynamic=False)
		self._compiles = True

		# What the captured halves of an update read and write, at addresses that never move
		self._bootstrap = torch.zeros(envs, device=device)
		self._entropy_weight = torch.zeros((), device=device)
		self._totals = torch.zeros(5, device=device)
		self._epoch_kl = torch.zeros((), device=device)
		self._graphs: tuple[RolloutBuffer, int, CapturedCall, CapturedCall, Tensor] | None = None

	def evaluate(self, observations: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
		"""The policy under inductor, falling back to eager for the rest of the run if it cannot compile."""
		if not self._compiles:
			return self.population.evaluate(observations, actions)
		try:
			return self._compiled(observations, actions)
		except Exception as error:  # A missing backend surfaces as whatever it failed on, so nothing narrower catches it
			print(f'Compiling the policy is unavailable, falling back to eager mode: {error}')
			self._compiles = False
			return self.population.evaluate(observations, actions)

	def _estimate(self, buffer: RolloutBuffer, steps: int) -> None:
		"""
		GAE(lambda) over the first `steps` of the rollout, written into the buffer's advantages and value targets.

		An episode that ended inside the rollout cuts the bootstrap at its last step, which is what stops the
		next episode's return from leaking backwards across the reset. The one that is still running at the
		far end bootstraps from the value of the observation the runner is holding.
		"""
		dones, values = buffer.dones[:steps], buffer.values[:steps]
		rewards = self.scaler.rescale(buffer.rewards[:steps], dones)
		keep = (~dones).to(rewards.dtype)
		# Every term but the recurrence itself is the same whole-rollout call, so the loop is left with one
		next_values = torch.cat((values[1:], self._bootstrap.unsqueeze(0)))
		deltas = rewards + self.gamma * next_values * keep - values
		decay = keep * (self.gamma * self.gae_lambda)

		advantages = buffer.advantages[:steps]
		running = torch.zeros_like(self._bootstrap)
		for step in range(steps - 1, -1, -1):
			running = torch.addcmul(deltas[step], running, decay[step])
			advantages[step] = running
		torch.add(advantages, values, out=buffer.returns[:steps])

	def _epoch(self, buffer: RolloutBuffer, steps: int, order: Tensor) -> None:
		"""One pass of clipped minibatch ascent over the rollout in `order`, adding what it measured to the totals."""
		observations = buffer.observations[:steps].view(-1, OBSERVATION_SIZE)
		actions = buffer.actions[:steps].view(-1)
		old_log_probs = buffer.log_probs[:steps].view(-1)
		advantages = buffer.advantages[:steps].view(-1)
		returns = buffer.returns[:steps].view(-1)

		size = max(1, order.shape[0] // self.minibatches)
		for start in range(0, order.shape[0], size):
			batch = order[start:start + size]
			log_probs, entropy, values = self.evaluate(
				observations.index_select(0, batch).to(self.population.dtype).unsqueeze(0),
				actions.index_select(0, batch).unsqueeze(0),
			)
			log_probs, entropy, values = log_probs.squeeze(0), entropy.squeeze(0), values.squeeze(0)
			old = old_log_probs.index_select(0, batch)

			# Normalised per minibatch, which is what every reference implementation converged on
			batch_advantages = advantages.index_select(0, batch)
			batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

			ratio = (log_probs - old).exp()
			clipped = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range)
			policy_loss = -torch.min(ratio * batch_advantages, clipped * batch_advantages).mean()
			value_loss = F.mse_loss(values, returns.index_select(0, batch))
			entropy_loss = entropy.mean()
			loss = policy_loss + self.value_coef * value_loss - self._entropy_weight * entropy_loss

			self.optimizer.zero_grad(set_to_none=True)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.population.parameters(), self.max_grad_norm)
			self.optimizer.step()

			with torch.no_grad():
				# Schulman's k3 estimator: unbiased and never negative, unlike the plain log ratio mean
				divergence = ((ratio - 1.0) - (log_probs - old)).mean()
				clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean()
				self._totals += torch.stack([policy_loss.detach(), value_loss.detach(), entropy_loss.detach(), divergence, clip_fraction])
				self._epoch_kl += divergence

	def _halves(self, buffer: RolloutBuffer, steps: int) -> tuple[CapturedCall, CapturedCall, Tensor]:
		"""
		The estimate and the epoch for a rollout of `steps` in `buffer`, with the order tensor the epoch reads.

		Launching them kernel by kernel left the device idle for more than half of an update, so both are
		captured. A graph only fits the rollout length it was captured on, and a rollout stopped by hand is
		shorter than the rest, so it gets its own pair and the full length is captured again after it.
		"""
		if self._graphs is None or self._graphs[0] is not buffer or self._graphs[1] != steps:
			device = self.population.device
			order = torch.empty(steps * buffer.envs, device=device, dtype=torch.int64)
			self._graphs = (
				buffer, steps,
				CapturedCall(lambda: self._estimate(buffer, steps), device),
				CapturedCall(lambda: self._epoch(buffer, steps, order), device),
				order,
			)
		_, _, estimate, epoch, order = self._graphs
		return estimate, epoch, order

	def anneal_entropy(self, floor: float = DEFAULT_ENTROPY_FINAL, updates: int = DEFAULT_ENTROPY_UPDATES) -> None:
		"""
		Walks the entropy bonus down to its floor: exploration on new ground, a sharper policy once it is known.

		The schedule counts updates on the current rung, not updates in the run. A curriculum can sit on one
		stretch of level for hundreds of updates, and a bonus annealing against the run would be at its floor
		by the time the hard rungs come up, leaving the policy nothing to explore the new ground with.
		"""
		share = min(1.0, self.rung_updates / max(1, updates))
		self.entropy_coef = self.entropy_start + (floor - self.entropy_start) * share

	def restart_exploration(self, stalled: bool = False) -> None:
		"""
		Restarts the entropy schedule, either for new ground or for ground the policy cannot get off.

		A rung that will not fall is a rung whose crossing the policy has already converged away from: the
		hardest jump on this level is found by 1 random agent in 16,384 against 50 for the ones either side
		of it, and a settled policy explores a good deal less than a random one. Each stall doubles what the
		rung restarts on, so exploration climbs back until the jump is inside what the policy still tries.
		"""
		self.rung_updates = 0
		self.entropy_start = min(MAX_ENTROPY_COEF, self.entropy_start * STALL_ENTROPY_BOOST) if stalled else self.entropy_base
		self.entropy_coef = self.entropy_start

	def update(
		self,
		buffer: RolloutBuffer,
		last_value: Tensor,
		steps: int | None = None,
		on_epoch: Callable[[], None] | None = None,
	) -> dict[str, float]:
		"""
		Runs the clipped epochs over one rollout and returns what the run's log and HUD show of them.

		`steps` is how much of the buffer was filled, which is all of it unless the rollout was stopped by
		hand. Anything past it is the previous rollout's data and is left out rather than trained on twice.

		`on_epoch` runs once an epoch has been launched, while the device works through it. An update over a
		quarter of a million transitions takes a good fraction of a second, which is a dead window unless
		something draws inside it.
		"""
		steps = buffer.steps if steps is None else min(steps, buffer.steps)
		total = steps * buffer.envs
		per_epoch = -(-total // max(1, total // self.minibatches))
		estimate, epoch, order = self._halves(buffer, steps)

		self._bootstrap.copy_(last_value)
		self._entropy_weight.fill_(self.entropy_coef)
		self._totals.zero_()
		estimate()
		epochs = 0
		for _ in range(self.epochs):
			torch.randperm(total, device=self.population.device, out=order)
			self._epoch_kl.zero_()
			epoch()
			epochs += 1
			if on_epoch is not None:
				on_epoch()
			# The one host read of the update: the rollout is off policy as soon as the weights have moved far enough
			if float(self._epoch_kl) / per_epoch > self.target_kl:
				break

		self.updates += 1
		self.rung_updates += 1
		self.anneal_entropy()
		averages = (self._totals / (per_epoch * epochs)).tolist()
		return {
			'policy_loss': averages[0],
			'value_loss': averages[1],
			'entropy': averages[2],
			'approx_kl': averages[3],
			'clip_fraction': averages[4],
			'return_scale': float(self.scaler.scale),
			'transitions': total,
		}

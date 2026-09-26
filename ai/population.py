import math
import random
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor

from game.world import OBSERVATION_DTYPE, OBSERVATION_SIZE

ACTION_COUNT: Final[int] = 3
DEFAULT_HIDDEN_SIZES: Final[tuple[int, int, int]] = (256, 128, 64)
LAYER_NORM_EPS: Final[float] = 1e-5
HOST_DTYPE: Final[torch.dtype] = torch.from_numpy(np.empty(0, dtype=OBSERVATION_DTYPE)).dtype  # The world's observation dtype, torch side


def pick_dtype(device: torch.device) -> torch.dtype:
	"""Half precision on CUDA, where it nearly halves the forward pass, full precision on CPU where it is slower."""
	return torch.float16 if device.type == 'cuda' else torch.float32


def seed_everything(seed: int) -> None:
	"""Seeds python, numpy and torch, on the host and on the device, so a run can be replayed exactly."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def pick_device(name: str = 'auto') -> torch.device:
	if name != 'auto':
		return torch.device(name)
	return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Population:
	"""
	The whole population as a single batched policy network.

	Every layer is one `(agents, in, out)` tensor, so a forward pass for the entire population is one
	`baddbmm` per layer instead of one module call per agent. Because each agent's weights only touch its
	own outputs, a single Adam over these tensors trains every agent independently, and the genetic
	operators (elites, crossover, mutation) are plain tensor ops on the same weights.

	Shapes follow `nn.Linear`'s state dict on export, and are widened back to float32 there, so a weight file
	stays readable as fc1/norm1/actor whatever the population runs in.

	`size` is how many independent networks the tensors hold and `batch` how many observations each one is
	asked about per pass. Evolution runs `size` networks on one observation each; PPO runs a single network
	on one observation per environment, which is the same `baddbmm` with the two dimensions swapped and
	reads a thousandth of the weight bytes per decision.
	"""

	def __init__(
		self,
		size: int,
		hidden_sizes: tuple[int, int, int] = DEFAULT_HIDDEN_SIZES,
		device: torch.device | None = None,
		batch: int = 1,
		critic: bool = False,
	) -> None:
		self.size = size
		self.batch = batch
		self.device = device if device is not None else pick_device()
		# Evolution only ever reads the argmax of the logits, so half precision drops nothing it uses; a
		# gradient does care, so a critic pins the whole policy to float32 rather than to a loss scaler
		self.dtype = torch.float32 if critic else pick_dtype(self.device)
		self.hidden_sizes = hidden_sizes
		first, second, third = hidden_sizes

		self.linear_shapes: dict[str, tuple[int, int]] = {
			'fc1': (OBSERVATION_SIZE, first),
			'fc2': (first, second),
			'fc3': (second, third),
			'actor': (third, ACTION_COUNT),
		}
		if critic:
			self.linear_shapes['critic'] = (third, 1)
		self.norm_shapes: dict[str, int] = {'norm1': first, 'norm2': second}

		self.weights: dict[str, Tensor] = {}
		self.biases: dict[str, Tensor] = {}
		for name, (fan_in, fan_out) in self.linear_shapes.items():
			self.weights[name] = torch.empty(size, fan_in, fan_out, device=self.device, dtype=self.dtype)
			self.biases[name] = torch.empty(size, 1, fan_out, device=self.device, dtype=self.dtype)
		for name, features in self.norm_shapes.items():
			self.weights[name] = torch.empty(size, 1, features, device=self.device, dtype=self.dtype)
			self.biases[name] = torch.empty(size, 1, features, device=self.device, dtype=self.dtype)

		self.randomize(torch.arange(size, device=self.device))
		if critic:
			# Evolution rewrites these tensors by hand; a policy gradient needs autograd to reach them instead
			for tensor in self.parameters():
				tensor.requires_grad_(True)

	def parameters(self) -> list[Tensor]:
		return [*self.weights.values(), *self.biases.values()]

	def randomize(self, indices: Tensor) -> None:
		"""Re-initialises the given agents, matching how nn.Linear and nn.LayerNorm initialise themselves."""
		if indices.numel() == 0:
			return
		with torch.no_grad():
			for name, (fan_in, _) in self.linear_shapes.items():
				bound = 1.0 / math.sqrt(fan_in)
				self.weights[name][indices] = torch.empty_like(self.weights[name][indices]).uniform_(-bound, bound)
				self.biases[name][indices] = torch.empty_like(self.biases[name][indices]).uniform_(-bound, bound)
			for name in self.norm_shapes:
				self.weights[name][indices] = 1.0
				self.biases[name][indices] = 0.0

	@staticmethod
	def _trunk(observations: Tensor, weights: dict[str, Tensor], biases: dict[str, Tensor]) -> Tensor:
		def linear(x: Tensor, name: str) -> Tensor:
			return torch.baddbmm(biases[name], x, weights[name])

		def layer_norm(x: Tensor, name: str) -> Tensor:
			# The fused kernel does the normalisation, the affine part stays per agent
			return F.layer_norm(x, (x.shape[-1],), eps=LAYER_NORM_EPS) * weights[name] + biases[name]

		x = F.leaky_relu(layer_norm(linear(observations, 'fc1'), 'norm1'))
		x = F.leaky_relu(layer_norm(linear(x, 'fc2'), 'norm2'))
		return F.leaky_relu(linear(x, 'fc3'))

	@staticmethod
	def forward_with(observations: Tensor, weights: dict[str, Tensor], biases: dict[str, Tensor]) -> Tensor:
		"""The policy's logits under any set of weights shaped like the population's, a subset of its agents included."""
		return torch.baddbmm(biases['actor'], Population._trunk(observations, weights, biases), weights['actor'])

	def forward(self, observations: Tensor) -> Tensor:
		"""Runs the population on `(agents, batch, OBSERVATION_SIZE)` observations, returning `(agents, batch, 3)` logits."""
		return self.forward_with(observations, self.weights, self.biases)

	def forward_actor_critic(self, observations: Tensor) -> tuple[Tensor, Tensor]:
		"""Logits and state values off one trunk pass, `(agents, batch, 3)` and `(agents, batch)`."""
		features = self._trunk(observations, self.weights, self.biases)
		actor = torch.baddbmm(self.biases['actor'], features, self.weights['actor'])
		return actor, torch.baddbmm(self.biases['critic'], features, self.weights['critic']).squeeze(-1)

	@torch.no_grad()
	def sample(self, observations: Tensor) -> tuple[Tensor, Tensor, Tensor]:
		"""
		Samples one action per environment and returns it with its log probability and the state value.

		Sampling is what PPO's ratio is defined against, so the rollout plays the distribution rather than
		its argmax. Gumbel-max draws from the softmax in one elementwise pass, with no host read and no
		kernel that a graph capture would refuse.
		"""
		logits, values = self.forward_actor_critic(observations)
		log_probs = F.log_softmax(logits.float(), dim=-1)
		uniform = torch.rand_like(log_probs).clamp_(1e-20, 1.0)
		actions = (log_probs - (-uniform.log()).log()).argmax(dim=-1)
		return actions, log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1), values.float()

	def evaluate(self, observations: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
		"""Log probability of `actions`, the policy's entropy and its values, with the graph the update needs."""
		logits, values = self.forward_actor_critic(observations)
		log_probs = F.log_softmax(logits.float(), dim=-1)
		taken = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
		entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
		return taken, entropy, values.float()

	@torch.no_grad()
	def act(self, observations: NDArray[np.float16]) -> NDArray[np.int64]:
		"""The greedy action of every agent on the world's `(agents, OBSERVATION_SIZE)` observations, for the numpy path."""
		batch = torch.from_numpy(observations).to(self.device, self.dtype).view(self.size, self.batch, OBSERVATION_SIZE)
		return self.forward(batch).squeeze(1).argmax(dim=-1).cpu().numpy()

	def evolve(self, fitness: NDArray[np.float32], elite_count: int, random_count: int, mutation_rate: float, mutation_strength: float) -> NDArray[np.int64]:
		"""
		Builds the next generation in place: elites are copied untouched, the middle is a mutated crossover
		of two random elites, the tail is re-randomised for diversity. Returns the elite indices.
		"""
		order = np.argsort(-fitness)
		elites = torch.from_numpy(np.ascontiguousarray(order[:elite_count])).to(self.device)
		offspring_count = max(0, self.size - elite_count - random_count)

		parents_a = elites[torch.randint(elite_count, (offspring_count,), device=self.device)]
		parents_b = elites[torch.randint(elite_count, (offspring_count,), device=self.device)]
		elite_slice = slice(0, elite_count)
		child_slice = slice(elite_count, elite_count + offspring_count)

		with torch.no_grad():
			for tensor in self.parameters():
				children = (tensor[parents_a] + tensor[parents_b]) / 2
				if offspring_count:
					mutated = torch.rand(offspring_count, device=self.device) < mutation_rate
					noise = torch.randn_like(children) * mutation_strength
					children += noise * mutated.view(-1, *([1] * (children.dim() - 1)))
				kept = tensor[elites].clone()
				tensor[elite_slice] = kept
				tensor[child_slice] = children

			self.randomize(torch.arange(elite_count + offspring_count, self.size, device=self.device))
		return order[:elite_count]

	def state_dict(self, index: int) -> dict[str, Tensor]:
		"""Exports one agent in plain nn.Linear / nn.LayerNorm layout, the format used by the weight files."""
		state: dict[str, Tensor] = {}
		for name in self.linear_shapes:
			state[f'{name}.weight'] = self.weights[name][index].t().float().contiguous().cpu()
			state[f'{name}.bias'] = self.biases[name][index, 0].float().contiguous().cpu()
		for name in self.norm_shapes:
			state[f'{name}.weight'] = self.weights[name][index, 0].float().contiguous().cpu()
			state[f'{name}.bias'] = self.biases[name][index, 0].float().contiguous().cpu()
		return state

	def load_state_dict(self, state: dict[str, Tensor]) -> None:
		"""Loads a single agent's state dict into every agent of the population."""
		with torch.no_grad():
			for name in self.linear_shapes:
				if f'{name}.weight' not in state:
					continue  # A file saved by the other trainer carries no critic, which keeps its random init
				weight = state[f'{name}.weight'].to(self.device, self.dtype).t()
				if weight.shape != self.weights[name].shape[1:]:
					raise ValueError(f'{name} expects {tuple(self.weights[name].shape[1:])}, got {tuple(weight.shape)}')
				self.weights[name][:] = weight
				self.biases[name][:] = state[f'{name}.bias'].to(self.device, self.dtype)
			for name in self.norm_shapes:
				self.weights[name][:] = state[f'{name}.weight'].to(self.device, self.dtype)
				self.biases[name][:] = state[f'{name}.bias'].to(self.device, self.dtype)

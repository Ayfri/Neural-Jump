import math
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor

from game.world import OBSERVATION_SIZE

ACTION_COUNT: Final[int] = 3
DEFAULT_HIDDEN_SIZES: Final[tuple[int, int, int]] = (256, 128, 64)
CRITIC_HIDDEN_SIZE: Final[int] = 32
LAYER_NORM_EPS: Final[float] = 1e-5


def pick_device(name: str = 'auto') -> torch.device:
	if name != 'auto':
		return torch.device(name)
	return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Population:
	"""
	The whole population as a single batched actor-critic network.

	Every layer is one `(agents, in, out)` tensor, so a forward pass for the entire population is one
	`baddbmm` per layer instead of one module call per agent. Because each agent's weights only touch its
	own outputs, a single Adam over these tensors trains every agent independently, and the genetic
	operators (elites, crossover, mutation) are plain tensor ops on the same weights.

	Shapes follow `nn.Linear`'s state dict on export, so a weight file stays readable as fc1/norm1/actor/critic.
	"""

	def __init__(self, size: int, hidden_sizes: tuple[int, int, int] = DEFAULT_HIDDEN_SIZES, device: torch.device | None = None) -> None:
		self.size = size
		self.device = device if device is not None else pick_device()
		self.hidden_sizes = hidden_sizes
		first, second, third = hidden_sizes

		self.linear_shapes: dict[str, tuple[int, int]] = {
			'fc1': (OBSERVATION_SIZE, first),
			'fc2': (first, second),
			'fc3': (second, third),
			'actor': (third, ACTION_COUNT),
			'critic_hidden': (third, CRITIC_HIDDEN_SIZE),
			'critic': (CRITIC_HIDDEN_SIZE, 1),
		}
		self.norm_shapes: dict[str, int] = {'norm1': first, 'norm2': second}

		self.weights: dict[str, Tensor] = {}
		self.biases: dict[str, Tensor] = {}
		for name, (fan_in, fan_out) in self.linear_shapes.items():
			self.weights[name] = torch.empty(size, fan_in, fan_out, device=self.device)
			self.biases[name] = torch.empty(size, 1, fan_out, device=self.device)
		for name, features in self.norm_shapes.items():
			self.weights[name] = torch.empty(size, 1, features, device=self.device)
			self.biases[name] = torch.empty(size, 1, features, device=self.device)

		self.randomize(torch.arange(size, device=self.device))
		for tensor in self.parameters():
			tensor.requires_grad_(True)

		self._graph: torch.cuda.CUDAGraph | None = None
		self._capture_graph()

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

	def _linear(self, x: Tensor, name: str) -> Tensor:
		return torch.baddbmm(self.biases[name], x, self.weights[name])

	def _layer_norm(self, x: Tensor, name: str) -> Tensor:
		# The fused kernel does the normalisation, the affine part stays per agent
		normalized = F.layer_norm(x, (x.shape[-1],), eps=LAYER_NORM_EPS)
		return normalized * self.weights[name] + self.biases[name]

	def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
		"""
		Runs the population on `(agents, batch, OBSERVATION_SIZE)` observations.

		Returns action logits `(agents, batch, 3)` and state values `(agents, batch)`.
		"""
		x = F.leaky_relu(self._layer_norm(self._linear(observations, 'fc1'), 'norm1'))
		x = F.leaky_relu(self._layer_norm(self._linear(x, 'fc2'), 'norm2'))
		x = F.leaky_relu(self._linear(x, 'fc3'))
		logits = self._linear(x, 'actor')
		value = self._linear(F.leaky_relu(self._linear(x, 'critic_hidden')), 'critic')
		return logits, value.squeeze(-1)

	@torch.no_grad()
	def _sample(self, observations: Tensor, deterministic: bool = False) -> Tensor:
		logits, _ = self.forward(observations)
		logits = logits.squeeze(1)
		if deterministic:
			return logits.argmax(dim=-1)
		# Gumbel-max: argmax(logits + Gumbel noise) samples exactly like softmax + multinomial, in fewer kernels
		gumbel = -torch.empty_like(logits).exponential_().log()
		return (logits + gumbel).argmax(dim=-1)

	def _capture_graph(self) -> None:
		"""
		Captures the sampling pass as a CUDA graph.

		One tick is a few dozen tiny kernels, so the pass is bound by launch latency rather than by maths.
		Replaying a captured graph collapses those launches into one, and the shapes never change here.
		The weights are updated in place by the optimizer and by evolution, so the graph keeps reading the
		current values.
		"""
		if self.device.type != 'cuda':
			return
		try:
			self._graph_input = torch.zeros(self.size, 1, OBSERVATION_SIZE, device=self.device)
			warmup = torch.cuda.Stream()
			warmup.wait_stream(torch.cuda.current_stream())
			with torch.cuda.stream(warmup):
				for _ in range(3):
					self._sample(self._graph_input)
			torch.cuda.current_stream().wait_stream(warmup)

			self._graph = torch.cuda.CUDAGraph()
			with torch.cuda.graph(self._graph):
				self._graph_output = self._sample(self._graph_input)
		except RuntimeError as error:
			print(f'CUDA graph capture unavailable, falling back to eager mode: {error}')
			self._graph = None

	def act(self, observations: NDArray[np.float32], deterministic: bool = False) -> NDArray[np.int64]:
		"""Picks one action per agent, sampled from the policy unless `deterministic` is set."""
		if self._graph is not None and not deterministic:
			self._graph_input.copy_(torch.from_numpy(observations).to(self.device, non_blocking=True).unsqueeze(1))
			self._graph.replay()
			return self._graph_output.cpu().numpy()

		x = torch.from_numpy(observations).to(self.device).unsqueeze(1)
		return self._sample(x, deterministic).cpu().numpy()

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
			state[f'{name}.weight'] = self.weights[name][index].t().contiguous().cpu()
			state[f'{name}.bias'] = self.biases[name][index, 0].contiguous().cpu()
		for name in self.norm_shapes:
			state[f'{name}.weight'] = self.weights[name][index, 0].contiguous().cpu()
			state[f'{name}.bias'] = self.biases[name][index, 0].contiguous().cpu()
		return state

	def load_state_dict(self, state: dict[str, Tensor]) -> None:
		"""Loads a single agent's state dict into every agent of the population."""
		with torch.no_grad():
			for name in self.linear_shapes:
				weight = state[f'{name}.weight'].to(self.device).t()
				if weight.shape != self.weights[name].shape[1:]:
					raise ValueError(f'{name} expects {tuple(self.weights[name].shape[1:])}, got {tuple(weight.shape)}')
				self.weights[name][:] = weight
				self.biases[name][:] = state[f'{name}.bias'].to(self.device)
			for name in self.norm_shapes:
				self.weights[name][:] = state[f'{name}.weight'].to(self.device)
				self.biases[name][:] = state[f'{name}.bias'].to(self.device)

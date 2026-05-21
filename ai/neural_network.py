import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Final

GRID_FEATURES: Final[int] = 7 * 7 * 4   # is_solid, is_flag, has_reward, is_empty per tile
PLAYER_FEATURES: Final[int] = 3          # vx_norm, vy_norm, on_ground
INPUT_SIZE: Final[int] = GRID_FEATURES + PLAYER_FEATURES  # 199


class NeuralNetwork(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.fc1 = nn.Linear(INPUT_SIZE, 256)
		self.norm1 = nn.LayerNorm(256)
		self.fc2 = nn.Linear(256, 128)
		self.norm2 = nn.LayerNorm(128)
		self.fc3 = nn.Linear(128, 64)
		self.actor = nn.Linear(64, 3)
		self.critic_hidden = nn.Linear(64, 32)
		self.critic = nn.Linear(32, 1)

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		x = F.leaky_relu(self.norm1(self.fc1(x)))
		x = F.leaky_relu(self.norm2(self.fc2(x)))
		x = F.leaky_relu(self.fc3(x))
		action_probs = F.softmax(self.actor(x), dim=-1)
		state_value = self.critic(F.leaky_relu(self.critic_hidden(x)))
		return action_probs, state_value

	def get_action(self, x: torch.Tensor, deterministic: bool = False) -> tuple[int, torch.Tensor, torch.Tensor]:
		action_probs, state_value = self.forward(x)
		if deterministic:
			action = action_probs.argmax(dim=-1)
		else:
			action = torch.distributions.Categorical(action_probs).sample()
		return int(action.item()), action_probs, state_value

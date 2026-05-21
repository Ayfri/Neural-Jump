import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		input_size = 7 * 7 * 2
		self.fc1 = nn.Linear(input_size, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 64)
		self.actor = nn.Linear(64, 3)
		self.critic = nn.Linear(64, 1)

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		action_probs = F.softmax(self.actor(x), dim=-1)
		state_value = self.critic(x)
		return action_probs, state_value

	def get_action(self, x: torch.Tensor, deterministic: bool = False) -> tuple[int, torch.Tensor, torch.Tensor]:
		action_probs, state_value = self.forward(x)
		if deterministic:
			action = torch.argmax(action_probs, dim=-1)
		else:
			action = torch.multinomial(action_probs, 1).squeeze(-1)
		action_int = int(action.item() if action.dim() == 0 else action[0].item())
		return action_int, action_probs, state_value

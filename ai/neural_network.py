import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
	def __init__(self) -> None:
		super(NeuralNetwork, self).__init__()
		self.fc1 = nn.Linear(7 * 7 * 2, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 64)
		self.fc4 = nn.Linear(64, 3)  # 3 directions: up, left, right

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = torch.relu(self.fc3(x))
		x = self.fc4(x)
		return x

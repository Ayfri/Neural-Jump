import torch
import torch.nn as nn


class SmallConvNeuralNetwork(nn.Module):
	def __init__(self) -> None:
		super(SmallConvNeuralNetwork, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
		self.fc1 = nn.Linear(32 * 9 * 9, 64)
		self.fc2 = nn.Linear(64, 3)  # 3 directions: up, left, right
		self.dropout = nn.Dropout(0.05)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = torch.relu(self.conv1(x))
		x = torch.relu(self.conv2(x))
		x = x.reshape(x.size(0), -1)  # Flatten the tensor
		x = torch.relu(self.fc1(x))
		x = self.dropout(x)
		x = self.fc2(x)
		return x

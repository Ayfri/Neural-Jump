import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
	"""
	Neural network for agent decision making.
	
	Input: 7x7 grid with 2 channels (rewards and solids) = 98 features
	Output: 3 actions (jump, left, right)
	
	Architecture:
	- Input layer: 98 features
	- Hidden layer 1: 256 neurons with ReLU
	- Hidden layer 2: 128 neurons with ReLU
	- Hidden layer 3: 64 neurons with ReLU
	- Output layer: 3 actions
	"""
	
	def __init__(self) -> None:
		super().__init__()
		input_size = 7 * 7 * 2  # 7x7 grid with 2 channels
		self.fc1 = nn.Linear(input_size, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 64)
		self.fc4 = nn.Linear(64, 3)  # 3 directions: jump, left, right

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass through the network.
		
		Args:
			x: Input tensor of shape (batch_size, 98)
		
		Returns:
			Output tensor of shape (batch_size, 3) with action logits
		"""
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = torch.relu(self.fc3(x))
		x = self.fc4(x)
		return x

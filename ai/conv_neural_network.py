import torch
import torch.nn as nn


class ConvNeuralNetwork(nn.Module):
	def __init__(self) -> None:
		super(ConvNeuralNetwork, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1)  # 2 channels: reward, is_solid
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # 64 filters, 3x3 kernel
		self.fc1 = nn.Linear(64 * 9 * 9, 128)  # 7x7 grid
		self.fc2 = nn.Linear(128, 64)  # 128 neurons in the hidden layer
		self.fc3 = nn.Linear(64, 3)  # 3 directions: up, left, right
		self.dropout = nn.Dropout(0.5)  # Dropout layer with 50% probability

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = torch.relu(self.conv1(x))  # Apply ReLU activation function
		x = torch.relu(self.conv2(x))  # Apply ReLU activation function
		x = x.reshape(x.size(0), -1)  # Flatten the tensor
		x = torch.relu(self.fc1(x))  # Apply ReLU activation function
		x = self.dropout(x)  # Apply dropout
		x = torch.relu(self.fc2(x))  # Apply ReLU activation function
		x = self.fc3(x)  # Output layer
		return x

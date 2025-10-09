import numpy as np
import pygame
import torch
from pygame import Surface
from torch import nn, optim

from ai.neural_network import NeuralNetwork
from game.game import Game
from game.player import Player
from game.tiles import Tile


class Agent:
	def __init__(self, tick_rate: int, show_window: bool, generation: 'Generation') -> None:
		self.tick_rate = tick_rate
		self.show_window = show_window
		self.generation = generation

		self.current_reward = 0
		self.playing = False
		self.player: Player | None = None
		self.screen: Surface | None = None

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = NeuralNetwork().to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
		self.criterion = nn.MSELoss()

	@property
	def current_index(self) -> int:
		return self.generation.agents.index(self)

	def calculate_move(self, grid: list[list[Tile]]) -> int:
		"""
		Calculate the movement to be made according to the surrounding grid.
		:param grid: The 7x7 grid around the player, each cell contains the properties of the tile
		:return: Direction of movement (0: jump, 1: left, 2: right)
		"""
		input_data = np.array(
			[[tile.get('reward', 0), tile.get('is_solid', 0)] for row in grid for tile in row],
			dtype=np.float32
		)
		input_data = input_data.flatten()
		input_tensor = torch.tensor(input_data).unsqueeze(0).to(self.device)

		with torch.no_grad():
			output = self.model(input_tensor)

		best_direction = torch.argmax(output).item()
		return best_direction

	def draw_minimap(self, game: Game, grid: list[list[Tile]], action: int) -> None:
		"""
		Draw the minimap showing the grid and the action taken by the agent.
		:param game: The game object
		:param grid: The 7x7 grid around the player
		:param action: The action chosen by the agent (0: jump, 1: left, 2: right)
		"""
		minimap_size = 200
		tile_size = minimap_size // 7
		minimap = pygame.Surface((minimap_size, minimap_size * 2))
		minimap.fill((255, 255, 255))
		action_colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0)]  # Colors for jump, left, right

		for y in range(7):
			for x in range(7):
				tile = grid[y][x]
				color = tile.get('color', (255, 255, 255))
				is_solid = tile.get('is_solid', False)
				is_player = tile.get('is_player', False)
				reward = tile.get('reward', 0)
				pygame.draw.rect(minimap, color, (x * tile_size, y * tile_size, tile_size, tile_size))

				# Draw additional properties
				if is_solid:
					pygame.draw.line(minimap, (0, 0, 0), (x * tile_size, y * tile_size), ((x + 1) * tile_size, (y + 1) * tile_size), 2)
				if is_player:
					pygame.draw.circle(
						minimap,
						(0, 0, 0),
						((x * tile_size + tile_size // 2), (y * tile_size + tile_size // 2)),
						tile_size // 4
						)
				if reward != 0:
					pygame.draw.rect(minimap, (0, 255, 255), (x * tile_size, y * tile_size, tile_size, tile_size), 2)

		# Draw the action color
		pygame.draw.rect(minimap, action_colors[action], (3 * tile_size, 3 * tile_size, tile_size, tile_size), 3)

		# Draw legend
		legend_y = minimap_size + 10
		legend_items = [
			("Player", (0, 0, 0), pygame.draw.circle),
			("Solid Block", (0, 0, 0), pygame.draw.line),
			("Reward Block", (0, 255, 255), pygame.draw.rect),
			("Action: Jump", (255, 0, 0), pygame.draw.rect),
			("Action: Left", (0, 0, 255), pygame.draw.rect),
			("Action: Right", (255, 255, 0), pygame.draw.rect),
		]

		for i, (text, color, draw_func) in enumerate(legend_items):
			text_surface = game._get_font(20).render(text, True, (0, 0, 0))
			minimap.blit(text_surface, (10, legend_y + i * 20))
			if draw_func == pygame.draw.circle:
				pygame.draw.circle(minimap, color, (150, legend_y + i * 20 + 10), 5)
			elif draw_func == pygame.draw.line:
				pygame.draw.line(minimap, color, (140, legend_y + i * 20 + 5), (160, legend_y + i * 20 + 15), 2)
			elif draw_func == pygame.draw.rect:
				pygame.draw.rect(minimap, color, (140, legend_y + i * 20, 20, 20), 2 if text == "Reward Block" else 0)

		self.screen = minimap

	def calculate_reward(self) -> float:
		"""
		Calculate the reward based on the game outcome.
		The reward system heavily emphasizes speed:
		- 0.5s = 10000 points (optimal)
		- 1s = 5000 points
		- 2s = 2500 points
		- 5s = 1000 points
		- 10s+ = very low rewards
		Exponential decay encourages agents to go as fast as possible.
		:return: The reward value
		"""
		if self.player.finished_reward is not None:
			if self.player.finished_reward == 1 and self.player.win_tick is not None:
				# Agent reached the flag - reward based purely on speed
				time_taken = self.player.win_tick / self.generation.tick_rate
				
				# Optimal time: 0.5 seconds
				optimal_time = 0.5
				
				# Exponential decay formula: reward = max_reward * (optimal_time / time_taken)^2
				# This creates a steep curve where being faster gives exponentially more reward
				max_reward = 10000
				
				if time_taken <= 0.01:  # Avoid division by zero
					time_taken = 0.01
				
				# Calculate reward with exponential decay
				reward = max_reward * (optimal_time / time_taken) ** 2
				
				# Cap minimum reward at 100 for very slow completions
				reward = max(100, reward)
				
				# Cap maximum reward at 20000 for sub-optimal times
				reward = min(20000, reward)
				
				return reward
			else:
				# Other rewards (obstacles, etc.)
				return self.player.finished_reward * 10
		else:
			# Agent didn't complete - small reward based on progress
			# Much smaller than completion rewards to prioritize finishing
			progress_reward = self.player.rect.x / 20
			
			# Penalty for dying
			if self.player.dead:
				progress_reward -= 20
			
			return max(0, progress_reward)

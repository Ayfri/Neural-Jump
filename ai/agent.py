import numpy as np
import pygame
import torch
from pygame import Surface
from torch import nn, optim

from ai.small_conv_neural_network import SmallConvNeuralNetwork
from game.game import Game
from game.player import Player
from game.tiles import Tile


class Agent:
	def __init__(self, tick_rate: int, show_window: bool, running_time: float, generation: 'Generation') -> None:
		self.tick_rate = tick_rate
		self.show_window = show_window
		self.running_time = running_time
		self.generation = generation

		self.current_reward = 0
		self.playing = False
		self.player: Player | None = None
		self.screen: Surface | None = None

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = SmallConvNeuralNetwork()
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
		self.criterion = nn.MSELoss()

	@property
	def current_index(self) -> int:
		return self.generation.agents.index(self)

	def calculate_move(self, grid: list[list[Tile]]) -> int:
		"""
		Calculate the movement to be made according to the surrounding grid.
		:param grid: The 7x7 grid around the player, each cell contains the properties of the tile
		:return: Direction of movement (0: up, 1: down, 2: left, 3: right)
		"""
		# Create a NumPy array directly from the grid without intermediate lists
		input_data = np.zeros((9, 9, 2), dtype=np.float32)
		for y, row in enumerate(grid):
			for x, tile in enumerate(row):
				input_data[y, x, 0] = tile.get('reward', 0)
				input_data[y, x, 1] = tile.get('is_solid', 0)

		input_tensor = torch.tensor(input_data).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)

		with torch.no_grad():
			output = self.model(input_tensor)

		best_direction = torch.argmax(output).item()
		return best_direction

	def draw_minimap(self, game: Game, grid: list[list[Tile]], action: int) -> None:
		"""
		Draw the minimap showing the grid and the action taken by the agent.
		:param game: The game object
		:param grid: The 7x7 grid around the player
		:param action: The action chosen by the agent
		"""
		minimap_size = 200
		tile_size = minimap_size // 7
		minimap = pygame.Surface((minimap_size, minimap_size * 2))
		minimap.fill((255, 255, 255))
		action_colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0)]  # Colors for actions up, left, right

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
			("Action: Up", (255, 0, 0), pygame.draw.rect),
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
		:return: The reward value
		"""
		if self.player.finished_reward is not None:
			return self.player.finished_reward
		else:
			player_reward = self.player.rect.x / 100  # Reward based on how far the player has moved right
			if self.player.dead:
				player_reward -= 10

			return player_reward

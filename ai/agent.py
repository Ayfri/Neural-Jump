import numpy as np
import pygame
import torch
from pygame import Surface
from torch import nn, optim

from ai.neural_network import NeuralNetwork
from game.game import Game
from game.settings import BLACK
from game.tiles import Tile


class Agent:
	def __init__(self, tick_rate: int, show_window: bool, running_time: float, generation: 'Generation') -> None:
		self.tick_rate = tick_rate
		self.show_window = show_window
		self.running_time = running_time
		self.generation = generation

		self.model = NeuralNetwork()
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
		self.criterion = nn.MSELoss()
		self.current_reward = 0
		self.playing = False

		self.screen: Surface | None = None

	def calculate_move(self, grid: list[list[Tile]]) -> int:
		"""
		Calculate the movement to be made according to the surrounding grid.
		:param grid: The 7x7 grid around the player, each cell contains the properties of the tile
		:return: Direction of movement (0: up, 1: down, 2: left, 3: right)
		"""
		input_data = np.array(
			[[tile.get('reward', 0), tile.get('is_solid', 0), tile.get('is_player', 0)] for row in grid for tile in row],
			dtype=np.float32
		)
		input_data = input_data.flatten()
		input_tensor = torch.tensor(input_data).unsqueeze(0)

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

	def calculate_reward(self, game: Game) -> float:
		"""
		Calculate the reward based on the game outcome.
		:param game: The game object
		:return: The reward value
		"""
		if game.player.finished_reward is not None:
			return game.player.finished_reward
		else:
			player_reward = game.player.rect.x / 100  # Reward based on how far the player has moved right
			level_reward = -game.level.world_shift[0] / 100  # Reward based on how far the player has moved right
			if game.player.dead:
				player_reward -= 10

			return player_reward + level_reward

	def play_game(self) -> float:
		"""
		Plays a game using the agent's current weights and returns the reward earned.
		"""
		self.playing = True
		game = Game(display_window=self.show_window, tick_rate=self.tick_rate)

		if game.display_window:
			pygame.init()
			pygame.font.init()
			game.init_window()

		# Time limit for the game in seconds
		time_limit = self.running_time

		# Start timer
		tick = 0

		while not game.level.finished and not game.player.dead and tick / 1000 < time_limit:
			# Get the surrounding grid
			grid = game.player.get_surrounding_grid()

			# Calculate the move using the agent
			direction = self.calculate_move(grid)

			# Execute the move
			game.player.execute_move(direction)
			game.handle_inputs(movement=False)
			game.update()
			tick += game.clock.get_time()

			# Display the timer
			if game.display_window:
				elapsed_time = tick / 1000
				game.draw_text(f"Agent: {self.generation.agents.index(self) + 1}/{self.generation.population_size}, Generation: {self.generation.generation}", 10, 10, font_size=24, color=BLACK)

				game.draw_text(f"FPS: {1000 / game.clock.get_time():.1f}", 10, 70, font_size=24, color=BLACK)
				game.draw_text(f"Time: {elapsed_time:.2f}/{time_limit}", 10, 100, font_size=24, color=BLACK)
				game.draw_text(f"Player: X: {game.player.rect.x}, Y: {game.player.rect.y}", 10, 130, font_size=24, color=BLACK)
				game.draw_text(f"Reward: {self.calculate_reward(game):.2f}", 10, 160, font_size=24, color=BLACK)
				game.draw_text(f"Level World Shift: X: {game.level.world_shift[0]:.2f}, Y: {game.level.world_shift[1]:.2f}", 10, 190, font_size=24, color=BLACK)

				self.draw_minimap(game, grid, direction)

				game.additional_draws += [(self.screen, (game.screen.get_width() - self.screen.get_width(), 0))]
				game.draw()

		# Update the agent's reward based on the game outcome
		reward = self.calculate_reward(game)

		game.quit()
		self.playing = False
		self.current_reward = reward
		return reward

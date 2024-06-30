import numpy as np
import pygame

from game.game import Game
from game.settings import BLACK
from game.tiles import Tile


class Agent:
	def __init__(self, tick_rate: int, show_window: bool, running_time: float, generation: 'Generation', weights: np.ndarray = None) -> None:
		self.tick_rate = tick_rate
		self.show_window = show_window
		self.running_time = running_time
		self.generation = generation

		self.weights = weights if weights is not None else np.random.rand(
			7,
			7,
			4
		)  # Weight for each direction (up, down, left, right) based on the 7x7 grid around the player
		self.current_reward = 0
		self.playing = False

	def calculate_move(self, grid: list[list[Tile]]) -> int:
		"""
		Calculate the movement to be made according to the surrounding grid.
		:param grid: The 7x7 grid around the player, each cell contains the properties of the tile
		:return: Direction of movement (0: up, 1: down, 2: left, 3: right)
		"""
		scores = np.zeros(4)  # Scores for each direction
		for i in range(7):
			for j in range(7):
				tile = grid[i][j]
				for direction in range(4):
					# Add the tile's contribution to the management's score
					if 'reward' in tile:
						scores[direction] += self.weights[i, j, direction] * tile['reward']
					if tile.get('is_solid', False):
						scores[direction] -= self.weights[i, j, direction]

		# Choose the direction with the highest score
		best_direction = np.argmax(scores)
		return best_direction

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
				game.draw()

		# Update the agent's reward based on the game outcome
		reward = self.calculate_reward(game)

		game.quit()
		self.playing = False
		self.current_reward = reward
		return reward

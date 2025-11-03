import numpy as np
import pygame
import torch
from numpy import ndarray
from pygame import Surface
from torch import nn, optim
from typing import TYPE_CHECKING, Final

from ai.neural_network import NeuralNetwork
from game.constants import MOVE_JUMP, MOVE_LEFT, MOVE_RIGHT
from game.game import Game
from game.player import Player
from game.tiles import Tile

if TYPE_CHECKING:
	from ai.generation import Generation

# Reward constants
FORWARD_MOVEMENT_REWARD: Final[float] = 0.02
NEW_MAX_POSITION_BONUS: Final[float] = 0.1
BACKWARD_MOVEMENT_PENALTY: Final[float] = -0.1
STATIONARY_PENALTY: Final[float] = -0.05
STATIONARY_THRESHOLD: Final[int] = 5  # Ticks before penalty kicks in
FALLING_PENALTY: Final[float] = -0.02
FALLING_THRESHOLD: Final[int] = 5  # Y distance before penalty

DEATH_PENALTY: Final[float] = -20.0
WIN_TIME_BONUS_MULTIPLIER: Final[float] = 100.0
WIN_TIME_BONUS_BASE: Final[float] = 10.0
DISTANCE_REWARD_DIVISOR: Final[float] = 10.0
PROGRESS_REWARD_DIVISOR: Final[float] = 20.0
MIN_REWARD: Final[float] = -30.0


class Agent:
	def __init__(self, tick_rate: int, show_window: bool, generation: 'Generation') -> None:
		self.tick_rate = tick_rate
		self.show_window = show_window
		self.generation = generation
		self.current_reward = 0.0
		self.player: Player | None = None
		self.screen: Surface | None = None
		
		# Reward shaping tracking
		self.previous_x = 0
		self.previous_y = 0
		self.ticks_stationary = 0
		self.max_x_reached = 0

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = NeuralNetwork().to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
		self.criterion = nn.MSELoss()

	@property
	def current_index(self) -> int:
		return self.generation.agents.index(self)

	def calculate_move(self, grid: list[list[Tile]]) -> int:
		"""Calculate movement based on 7x7 grid (MOVE_JUMP, MOVE_LEFT, MOVE_RIGHT)"""
		rewards = [tile.get('reward', 0) for row in grid for tile in row]
		solids = [float(tile.get('is_solid', False)) for row in grid for tile in row]
		input_data: ndarray[np.float32] = np.array(
			rewards + solids,
			dtype=np.float32
		)
		input_tensor = torch.tensor(input_data, device=self.device).unsqueeze(0)

		with torch.no_grad():
			output = self.model(input_tensor)

		action = int(torch.argmax(output).item())
		return action

	def draw_minimap(self, game: Game, grid: list[list[Tile]], action: int) -> None:
		"""Draw minimap showing grid and chosen action"""
		minimap_size = 200
		tile_size = minimap_size // 7
		minimap = pygame.Surface((minimap_size, minimap_size * 2))
		minimap.fill((255, 255, 255))
		action_colors = {
			MOVE_JUMP: (255, 0, 0),    # Red
			MOVE_LEFT: (0, 0, 255),     # Blue
			MOVE_RIGHT: (255, 255, 0),  # Yellow
		}

		for y in range(7):
			for x in range(7):
				tile = grid[y][x]
				color = tile.get('color', (255, 255, 255))
				pygame.draw.rect(minimap, color, (x * tile_size, y * tile_size, tile_size, tile_size))

				if tile.get('is_solid', False):
					pygame.draw.line(minimap, (0, 0, 0), (x * tile_size, y * tile_size), ((x + 1) * tile_size, (y + 1) * tile_size), 2)
				if tile.get('is_player', False):
					pygame.draw.circle(minimap, (0, 0, 0), ((x * tile_size + tile_size // 2), (y * tile_size + tile_size // 2)), tile_size // 4)
				if tile.get('reward', 0) != 0:
					pygame.draw.rect(minimap, (0, 255, 255), (x * tile_size, y * tile_size, tile_size, tile_size), 2)

		action_color = action_colors.get(action, (128, 128, 128))
		pygame.draw.rect(minimap, action_color, (3 * tile_size, 3 * tile_size, tile_size, tile_size), 3)

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
		"""Calculate final episode reward"""
		assert self.player is not None, "Player must be set before calculating reward"
		
		if self.player.finished_reward is not None:
			if self.player.finished_reward == 1 and self.player.win_tick is not None:
				time_taken = self.player.win_tick / self.generation.tick_rate
				distance_reward = self.player.rect.x / DISTANCE_REWARD_DIVISOR
				time_bonus = max(0.0, WIN_TIME_BONUS_BASE - time_taken) * WIN_TIME_BONUS_MULTIPLIER
				return distance_reward + time_bonus
			return self.player.finished_reward * DISTANCE_REWARD_DIVISOR
		
		# Use max_x_reached for progress (rewards exploration, not just final position)
		progress_reward = self.max_x_reached / PROGRESS_REWARD_DIVISOR
		if self.player.dead:
			progress_reward += DEATH_PENALTY
		
		return max(MIN_REWARD, progress_reward)  # Floor at minimum reward

	def calculate_continuous_reward(self) -> float:
		"""Calculate micro-rewards/penalties at each tick"""
		if not self.player or self.player.dead or self.player.win:
			return 0.0

		reward = 0.0
		current_x = self.player.rect.x
		x_delta = current_x - self.previous_x

		if x_delta > 0:
			reward += FORWARD_MOVEMENT_REWARD
			if current_x > self.max_x_reached:
				reward += NEW_MAX_POSITION_BONUS
				self.max_x_reached = current_x
		elif x_delta < 0:
			reward += BACKWARD_MOVEMENT_PENALTY
		else:
			# Aggressive penalty for staying still
			self.ticks_stationary += 1
			if self.ticks_stationary > STATIONARY_THRESHOLD:
				reward += STATIONARY_PENALTY

		if x_delta != 0:
			self.ticks_stationary = 0

		y_delta = self.player.rect.y - self.previous_y
		if y_delta > FALLING_THRESHOLD:
			reward += FALLING_PENALTY

		self.previous_x = current_x
		self.previous_y = self.player.rect.y

		return reward

	def reset_continuous_reward_tracking(self) -> None:
		"""Reset tracking variables for new episode/checkpoint"""
		if self.player:
			self.previous_x = self.player.rect.x
			self.previous_y = self.player.rect.y
			self.max_x_reached = self.player.rect.x
		else:
			self.previous_x = 0
			self.previous_y = 0
			self.max_x_reached = 0
		self.ticks_stationary = 0

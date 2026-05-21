import numpy as np
import pygame
import torch
from pygame import Surface
from torch import optim
import torch.nn.functional as F
from typing import TYPE_CHECKING, Final

from ai.neural_network import NeuralNetwork
from game.constants import MOVE_JUMP, MOVE_LEFT, MOVE_RIGHT
from game.game import Game
from game.player import Player
from game.tiles import Tile

if TYPE_CHECKING:
	from ai.generation import Generation

FORWARD_MOVEMENT_REWARD: Final[float] = 0.02
NEW_MAX_POSITION_BONUS: Final[float] = 0.1
BACKWARD_MOVEMENT_PENALTY: Final[float] = -0.1
STATIONARY_PENALTY: Final[float] = -0.05
STATIONARY_THRESHOLD: Final[int] = 5
FALLING_PENALTY: Final[float] = -0.02
FALLING_THRESHOLD: Final[int] = 5

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

		self.previous_x = 0
		self.previous_y = 0
		self.ticks_stationary = 0
		self.max_x_reached = 0

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = NeuralNetwork().to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
		self.value_loss_coef = 0.5
		self.entropy_coef = 0.01
		self.actions = []
		self.rewards = []
		self.values = []
		self.action_probs = []
		self.gamma = 0.99

	@property
	def current_index(self) -> int:
		return self.generation.agents.index(self)

	def _grid_to_numpy(self, grid: list[list[Tile]]) -> np.ndarray:
		return np.array(
			[
				[
					float(tile.get('is_solid', False)),
					float(tile.get('reward', 0) == 1),           # flag tile (win condition)
					float(tile.get('reward', 0) > 0),            # any reward tile
					float(not tile.get('is_solid', False) and tile.get('reward', 0) == 0),  # empty/passable
				]
				for row in grid for tile in row
			],
			dtype=np.float32,
		).flatten()  # 49 * 4 = 196

	def _encode_state(self, grid: list[list[Tile]]) -> torch.Tensor:
		grid_data = self._grid_to_numpy(grid)
		if self.player is not None:
			player_state = np.array([
				self.player.change_x / 8.0,                    # normalised by max speed
				self.player.change_y / 20.0,                   # normalised by ~max fall speed
				float(abs(self.player.change_y) <= 2.0),       # approximately on ground
			], dtype=np.float32)
		else:
			player_state = np.zeros(3, dtype=np.float32)
		return torch.from_numpy(np.concatenate([grid_data, player_state])).unsqueeze(0).to(self.device)

	def calculate_move(self, grid: list[list[Tile]]) -> int:
		with torch.no_grad():
			action, _, _ = self.model.get_action(self._encode_state(grid), deterministic=False)
		return action

	def calculate_move_with_learning(self, grid: list[list[Tile]]) -> int:
		action, action_probs, state_value = self.model.get_action(self._encode_state(grid), deterministic=False)
		self.actions.append(action)
		self.values.append(state_value)
		self.action_probs.append(action_probs)
		return action

	def store_reward(self, reward: float) -> None:
		self.rewards.append(reward)

	def compute_returns(self) -> list[float]:
		returns = []
		R = 0.0
		for r in reversed(self.rewards):
			R = r + self.gamma * R
			returns.append(R)
		returns.reverse()
		return returns

	def update_policy(self) -> tuple[float, float, float]:
		if len(self.actions) == 0:
			return 0.0, 0.0, 0.0

		returns = self.compute_returns()
		returns_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)
		if len(returns) > 1:
			returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

		actions = torch.tensor(self.actions, dtype=torch.long).to(self.device)
		values = torch.cat(self.values, dim=0).squeeze()
		action_probs = torch.cat(self.action_probs, dim=0)

		dist = torch.distributions.Categorical(action_probs)
		advantages = returns_tensor - values.detach()
		actor_loss = -(dist.log_prob(actions) * advantages).mean()
		entropy = dist.entropy().mean()
		critic_loss = F.mse_loss(values, returns_tensor)
		total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

		self.optimizer.zero_grad()
		total_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
		self.optimizer.step()
		self.clear_trajectory()
		return total_loss.item(), actor_loss.item(), critic_loss.item()

	def clear_trajectory(self) -> None:
		self.actions = []
		self.rewards = []
		self.values = []
		self.action_probs = []

	def calculate_moves_batch(self, grids: list[list[list[Tile]]]) -> list[int]:
		pad = np.zeros(3, dtype=np.float32)
		batch = np.stack([np.concatenate([self._grid_to_numpy(g), pad]) for g in grids])
		with torch.no_grad():
			action_probs, _ = self.model(torch.from_numpy(batch).to(self.device))
		return torch.argmax(action_probs, dim=1).tolist()

	def draw_minimap(self, game: Game, grid: list[list[Tile]], action: int) -> None:
		"""Draw minimap showing grid and chosen action"""
		minimap_size = 200
		tile_size = minimap_size // 7
		minimap = pygame.Surface((minimap_size, minimap_size * 2))
		minimap.fill((255, 255, 255))
		action_colors = {
			MOVE_JUMP: (255, 0, 0),
			MOVE_LEFT: (0, 0, 255),
			MOVE_RIGHT: (255, 255, 0),
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

		progress_reward = self.max_x_reached / PROGRESS_REWARD_DIVISOR
		if self.player.dead:
			progress_reward += DEATH_PENALTY

		return max(MIN_REWARD, progress_reward)

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

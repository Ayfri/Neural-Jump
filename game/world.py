import os
from typing import Final

import numpy as np
from numpy.typing import NDArray

from game.constants import AGENT_VISION_DISTANCE, MOVE_JUMP, MOVE_LEFT, MOVE_RIGHT
from game.settings import (
	PLAYER_GRAVITY, PLAYER_HEIGHT, PLAYER_JUMP_STRENGTH, PLAYER_SPEED, PLAYER_WIDTH, SCREEN_HEIGHT, TILE_SIZE,
)
from game.tiles import TILES

PLAYER_W: Final[int] = int(PLAYER_WIDTH)
PLAYER_H: Final[int] = int(PLAYER_HEIGHT)
GRID_SIDE: Final[int] = AGENT_VISION_DISTANCE * 2 + 1
GRID_TILES: Final[int] = GRID_SIDE * GRID_SIDE
GRID_CHANNELS: Final[int] = 4  # is_solid, is_flag, has_reward, is_empty per tile
GRID_FEATURES: Final[int] = GRID_TILES * GRID_CHANNELS
PLAYER_FEATURES: Final[int] = 3  # change_x normalised, change_y normalised, on ground
OBSERVATION_SIZE: Final[int] = GRID_FEATURES + PLAYER_FEATURES
MAX_FALL_SPEED: Final[float] = 20.0  # Normalisation divisor for the vertical speed feature
ON_GROUND_SPEED: Final[float] = 2.0  # Vertical speed under which the player counts as grounded
DEATH_ROW_MARGIN: Final[int] = 2  # Rows above the bottom of the map that kill the player
GRID_PADDING: Final[int] = 32  # Air border baked around the grids so lookups never need bounds checks


def search_maps_folder(folder: str) -> str:
	"""Returns the absolute path to the maps folder, walking up from this file until it is found."""
	current_folder = os.path.dirname(os.path.abspath(__file__))
	while True:
		if folder in os.listdir(current_folder):
			return os.path.join(current_folder, folder)
		parent = os.path.dirname(current_folder)
		if parent == current_folder:
			return ''
		current_folder = parent


def resolve_map_path(map_path: str) -> str:
	"""Resolves a map path like 'maps/level_1.txt' to an absolute path."""
	folder = search_maps_folder(os.path.dirname(map_path))
	return folder + os.sep + os.path.basename(map_path)


class World:
	"""
	Batched world simulating `count` players at once with numpy, without pygame or sprites.

	Everything is stored as parallel arrays of shape (count,), so one tick is a handful of vectorised
	operations instead of one Python loop per player. Collisions are resolved against a static tile
	grid, so cost is independent of the map size.
	"""

	def __init__(self, map_path: str, count: int) -> None:
		self.count = count
		self.map_path = map_path
		self._load_map(map_path)

		self.x = np.zeros(count, dtype=np.float64)
		self.y = np.zeros(count, dtype=np.float64)
		self.change_x = np.zeros(count, dtype=np.float64)
		self.change_y = np.zeros(count, dtype=np.float64)
		self.dead = np.zeros(count, dtype=np.bool_)
		self.win = np.zeros(count, dtype=np.bool_)
		self.finished_reward = np.zeros(count, dtype=np.float32)
		self.win_tick = np.full(count, -1, dtype=np.int32)

		self._observation = np.zeros((count, OBSERVATION_SIZE), dtype=np.float32)
		# One strided view per tile channel, so a gathered (count, 7, 7) block is written without any copy
		self._channel_views = [self._observation[:, channel:GRID_FEATURES:GRID_CHANNELS] for channel in range(GRID_CHANNELS)]
		self._player_view = self._observation[:, GRID_FEATURES:]
		self._offsets = np.arange(-AGENT_VISION_DISTANCE, AGENT_VISION_DISTANCE + 1)

	def _load_map(self, map_path: str) -> None:
		with open(resolve_map_path(map_path)) as file:
			lines = [line.strip() for line in file.readlines() if line.strip()]

		self.height = len(lines)
		self.width = max(len(line) for line in lines)
		self.offset_y = SCREEN_HEIGHT - self.height * TILE_SIZE
		self.death_y = (self.height - DEATH_ROW_MARGIN) * TILE_SIZE

		self.chars = np.full((self.height, self.width), '.', dtype='<U1')
		self.solid = np.zeros((self.height, self.width), dtype=np.bool_)
		self.reward = np.zeros((self.height, self.width), dtype=np.float32)
		self.spawn_point = (0, 0)
		self.checkpoints: list[tuple[int, int]] = []

		for y, line in enumerate(lines):
			for x, char in enumerate(line):
				tile = TILES.get(char)
				if tile is None:
					continue
				self.chars[y, x] = char
				self.solid[y, x] = tile.get('is_solid', False)
				self.reward[y, x] = tile.get('reward', 0)
				if tile.get('is_player', False):
					self.spawn_point = (x * TILE_SIZE, y * TILE_SIZE + self.offset_y)
				elif tile.get('is_checkpoint', False):
					self.checkpoints.append((x * TILE_SIZE, y * TILE_SIZE + self.offset_y))

		# Padded copies: any tile index is clamped into the air border instead of being bounds-checked
		self.padded_solid = np.pad(self.solid, GRID_PADDING)
		self.padded_reward = np.pad(self.reward, GRID_PADDING)
		# One float grid per observation channel, gathered as-is every tick
		self.padded_channels = [
			self.padded_solid.astype(np.float32),
			(self.padded_reward == 1).astype(np.float32),
			(self.padded_reward > 0).astype(np.float32),
			(~self.padded_solid & (self.padded_reward == 0)).astype(np.float32),
		]
		self._max_row = self.height + 2 * GRID_PADDING - 1
		self._max_column = self.width + 2 * GRID_PADDING - 1

	def reset(self, spawn_x: int, spawn_y: int) -> None:
		"""Places every player on the given spawn point and clears their state."""
		self.x.fill(spawn_x)
		self.y.fill(spawn_y)
		self.change_x.fill(0.0)
		self.change_y.fill(0.0)
		self.dead.fill(False)
		self.win.fill(False)
		self.finished_reward.fill(0.0)
		self.win_tick.fill(-1)

	def alive(self) -> NDArray[np.bool_]:
		return ~(self.dead | self.win)

	def kill(self, mask: NDArray[np.bool_]) -> None:
		self.dead |= mask

	def _tile_columns(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
		left = np.floor_divide(self.x, TILE_SIZE).astype(np.int64)
		right = np.floor_divide(self.x + PLAYER_W - 1, TILE_SIZE).astype(np.int64)
		return left, right

	def _tile_rows(self, y: NDArray[np.float64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
		top = np.floor_divide(y - self.offset_y, TILE_SIZE).astype(np.int64)
		bottom = np.floor_divide(y + PLAYER_H - 1 - self.offset_y, TILE_SIZE).astype(np.int64)
		return top, bottom

	def _gather(self, padded_grid: NDArray[np.bool_] | NDArray[np.float32], rows: NDArray[np.int64], columns: NDArray[np.int64]) -> NDArray[np.bool_] | NDArray[np.float32]:
		"""Gathers padded_grid at every (row, column) pair, so a 2x(2) index pair returns a (2, 2, count) block."""
		row_index = np.clip(rows + GRID_PADDING, 0, self._max_row)
		column_index = np.clip(columns + GRID_PADDING, 0, self._max_column)
		return padded_grid[row_index[:, None, :], column_index[None, :, :]]

	def observe(self) -> NDArray[np.float32]:
		"""
		Returns the (count, 199) observation: the 7x7 tile window around each player encoded as four
		channels per tile (solid, flag, reward, empty), followed by the player's own speed and ground state.
		"""
		center_x = self.x + PLAYER_W / 2
		center_y = self.y + PLAYER_H / 2 - self.offset_y
		tile_x = np.floor_divide(center_x, TILE_SIZE).astype(np.int64)
		tile_y = np.floor_divide(center_y, TILE_SIZE).astype(np.int64)

		rows = np.clip(tile_y + GRID_PADDING, AGENT_VISION_DISTANCE, self._max_row - AGENT_VISION_DISTANCE)[:, None, None] + self._offsets[None, :, None]
		columns = np.clip(tile_x + GRID_PADDING, AGENT_VISION_DISTANCE, self._max_column - AGENT_VISION_DISTANCE)[:, None, None] + self._offsets[None, None, :]

		for channel, grid in enumerate(self.padded_channels):
			self._channel_views[channel][:] = grid[rows, columns].reshape(self.count, GRID_TILES)

		self._player_view[:, 0] = self.change_x / PLAYER_SPEED
		self._player_view[:, 1] = self.change_y / MAX_FALL_SPEED
		self._player_view[:, 2] = np.abs(self.change_y) <= ON_GROUND_SPEED
		return self._observation

	def step(self, actions: NDArray[np.int64], tick: int) -> None:
		"""Applies one action per player then advances the physics by one tick."""
		alive = self.alive()

		self.change_x = np.where(alive & (actions == MOVE_LEFT), -PLAYER_SPEED, self.change_x)
		self.change_x = np.where(alive & (actions == MOVE_RIGHT), PLAYER_SPEED, self.change_x)
		self._jump(alive & (actions == MOVE_JUMP))

		gravity = np.where(self.change_y == 0.0, 1.0, self.change_y + PLAYER_GRAVITY)
		self.change_y = np.where(alive, gravity, self.change_y)

		self.x = np.where(alive, np.trunc(self.x + self.change_x), self.x)
		self._resolve_horizontal(alive)

		self.dead |= alive & (self.y >= self.death_y)
		alive = self.alive()

		self.y = np.where(alive, np.trunc(self.y + self.change_y), self.y)
		self._resolve_vertical(alive)
		self._collect_rewards(alive, tick)

	def _jump(self, mask: NDArray[np.bool_]) -> None:
		if not mask.any():
			return
		left, right = self._tile_columns()
		_, bottom = self._tile_rows(self.y + 2.0)
		block = self._gather(self.padded_solid, np.stack([bottom, bottom]), np.stack([left, right]))
		grounded = block[0, 0] | block[0, 1]
		self.change_y = np.where(mask & grounded, PLAYER_JUMP_STRENGTH, self.change_y)

	def _resolve_horizontal(self, alive: NDArray[np.bool_]) -> None:
		left, right = self._tile_columns()
		top, bottom = self._tile_rows(self.y)
		two_rows = bottom > top
		two_columns = right > left

		block = self._gather(self.padded_solid, np.stack([top, bottom]), np.stack([left, right]))
		hit_left = block[0, 0] | (block[1, 0] & two_rows)
		hit_right = (block[0, 1] | (block[1, 1] & two_rows)) & two_columns

		going_right = self.change_x > 0
		going_left = self.change_x < 0
		x = self.x
		x = np.where(going_right & hit_right, right * TILE_SIZE - PLAYER_W, x)
		x = np.where(going_right & hit_left & ~hit_right, left * TILE_SIZE - PLAYER_W, x)
		x = np.where(going_left & hit_left, (left + 1) * TILE_SIZE, x)
		x = np.where(going_left & hit_right & ~hit_left, (right + 1) * TILE_SIZE, x)
		self.x = np.where(alive, x, self.x)

	def _resolve_vertical(self, alive: NDArray[np.bool_]) -> None:
		left, right = self._tile_columns()
		top, bottom = self._tile_rows(self.y)
		two_rows = bottom > top
		two_columns = right > left

		block = self._gather(self.padded_solid, np.stack([top, bottom]), np.stack([left, right]))
		hit_top = block[0, 0] | (block[0, 1] & two_columns)
		hit_bottom = (block[1, 0] | (block[1, 1] & two_columns)) & two_rows

		going_down = self.change_y > 0
		going_up = self.change_y < 0
		y = self.y
		y = np.where(going_down & hit_bottom, bottom * TILE_SIZE + self.offset_y - PLAYER_H, y)
		y = np.where(going_down & hit_top & ~hit_bottom, top * TILE_SIZE + self.offset_y - PLAYER_H, y)
		y = np.where(going_up & hit_top, (top + 1) * TILE_SIZE + self.offset_y, y)
		y = np.where(going_up & hit_bottom & ~hit_top, (bottom + 1) * TILE_SIZE + self.offset_y, y)

		blocked = (going_down | going_up) & (hit_top | hit_bottom)
		self.y = np.where(alive, y, self.y)
		self.change_y = np.where(alive & blocked, 0.0, self.change_y)

	def _collect_rewards(self, alive: NDArray[np.bool_], tick: int) -> None:
		left, right = self._tile_columns()
		top, bottom = self._tile_rows(self.y)
		values = self._gather(self.padded_reward, np.stack([top, bottom]), np.stack([left, right]))
		best = values.reshape(4, -1).max(axis=0)

		touched = alive & (best != 0)
		self.finished_reward = np.where(touched, best, self.finished_reward)
		won = touched & (best == 1)
		self.win_tick = np.where(won, tick, self.win_tick)
		self.win |= won

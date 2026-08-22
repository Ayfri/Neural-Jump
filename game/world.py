import os
from typing import Final

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
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
# Observations are flags and small normalised ratios, so half precision keeps every bit that matters while
# halving the host-to-device copy done every tick.
OBSERVATION_DTYPE: Final[np.dtype[np.float16]] = np.dtype(np.float16)
MAX_FALL_SPEED: Final[float] = 20.0  # Normalisation divisor for the vertical speed feature
ON_GROUND_SPEED: Final[float] = 2.0  # Vertical speed under which the player counts as grounded
DEATH_ROW_MARGIN: Final[int] = 2  # Rows above the bottom of the map that kill the player
GRID_PADDING: Final[int] = 32  # Air border baked around the grids so lookups never need bounds checks
GRID_ORIGIN: Final[int] = GRID_PADDING * TILE_SIZE  # Pixels the padding adds to a tile coordinate

MOVE_SPEEDS: Final[NDArray[np.float64]] = np.zeros(3)  # Horizontal speed per action, jumping keeps the current one
MOVE_SPEEDS[[MOVE_LEFT, MOVE_RIGHT]] = (-PLAYER_SPEED, PLAYER_SPEED)


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


def tile_of(values: NDArray[np.float64]) -> NDArray[np.int64]:
	"""Tile index of a pixel coordinate, dividing then flooring: `np.floor_divide` on floats costs twice as much."""
	return np.floor(values / TILE_SIZE).astype(np.int64)


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

		self._observation = np.zeros((count, OBSERVATION_SIZE), dtype=OBSERVATION_DTYPE)
		self._offsets = np.arange(-AGENT_VISION_DISTANCE, AGENT_VISION_DISTANCE + 1)
		self._box = np.zeros((4, count), dtype=np.float64)  # Scratch the collision passes rebuild every call
		self._cell_limits = np.array([[self._max_row], [self._max_row], [self._max_column], [self._max_column]])

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
		# Channel-last grid, so the whole 7x7x4 window of every player comes out of one contiguous gather
		self.padded_grid = np.stack([
			self.padded_solid,
			self.padded_reward == 1,
			self.padded_reward > 0,
			~self.padded_solid & (self.padded_reward == 0),
		], axis=-1).astype(OBSERVATION_DTYPE)
		self._max_row = self.height + 2 * GRID_PADDING - 1
		self._max_column = self.width + 2 * GRID_PADDING - 1
		# Collision lookups index these flat views: `take` on one flat array beats a broadcast fancy index
		self._padded_width = self.width + 2 * GRID_PADDING
		self._flat_solid = self.padded_solid.ravel()
		self._flat_reward = self.padded_reward.ravel()
		# True where any of the 2x2 tiles from (row, column) carries a reward: one lookup then skips the whole pass
		rewarding = self.padded_reward != 0
		near_reward = rewarding.copy()
		near_reward[:-1] |= rewarding[1:]
		near_reward[:, :-1] |= near_reward[:, 1:].copy()
		self._flat_near_reward = near_reward.ravel()

		# Every 7x7x4 window of the map, flattened and baked once: an observation is then a single gather of
		# contiguous rows instead of a broadcast fancy index rebuilt per tick.
		windows = sliding_window_view(self.padded_grid, (GRID_SIDE, GRID_SIDE), axis=(0, 1))
		self._window_stride = windows.shape[1]
		self._windows = np.ascontiguousarray(windows.transpose(0, 1, 3, 4, 2)).reshape(-1, GRID_FEATURES)
		self._max_window_row = windows.shape[0] - 1
		self._max_window_column = self._window_stride - 1

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

	def _cells(self, y: NDArray[np.float64]) -> NDArray[np.int64]:
		"""
		The four tiles the player box touches at (x, y): top row, bottom row, left column, right column.

		They come back as one (4, count) block of padded grid coordinates, built in a handful of numpy calls
		over the whole block, because on 300 players a call costs far more than the arithmetic inside it.
		Coordinates are clamped into the air border, which only ever moves a player already off the map.
		"""
		# The padding is added here in pixels, so the coordinates come out of the floor already padded
		box = self._box
		np.subtract(y, self.offset_y - GRID_ORIGIN, out=box[0])
		np.add(box[0], PLAYER_H - 1, out=box[1])
		np.add(self.x, GRID_ORIGIN, out=box[2])
		np.add(box[2], PLAYER_W - 1, out=box[3])

		np.divide(box, TILE_SIZE, out=box)
		np.floor(box, out=box)
		cells = box.astype(np.int64)
		np.maximum(cells, 0, out=cells)
		np.minimum(cells, self._cell_limits, out=cells)
		return cells

	def _row_offsets(self, cells: NDArray[np.int64]) -> NDArray[np.int64]:
		"""The two rows of a `_cells` block as offsets into the flat padded grid."""
		return cells[:2] * self._padded_width

	def on_ground(self) -> NDArray[np.bool_]:
		"""Players whose vertical speed is small enough to count as standing on something."""
		return np.abs(self.change_y) <= ON_GROUND_SPEED

	def observe(self, out: NDArray[np.float16] | None = None) -> NDArray[np.float16]:
		"""
		Fills `out` (or the world's own buffer) with the (count, 199) observation: the 7x7 tile window
		around each player encoded as four channels per tile (solid, flag, reward, empty), followed by the
		player's own speed and ground state.
		"""
		target = self._observation if out is None else out
		tile_x = tile_of(self.x + PLAYER_W / 2)
		tile_y = tile_of(self.y + (PLAYER_H / 2 - self.offset_y))

		# The window table is indexed by its top-left corner, so the centre clamp becomes a corner clamp
		rows = np.clip(tile_y + GRID_PADDING - AGENT_VISION_DISTANCE, 0, self._max_window_row)
		columns = np.clip(tile_x + GRID_PADDING - AGENT_VISION_DISTANCE, 0, self._max_window_column)
		rows *= self._window_stride
		rows += columns

		np.take(self._windows, rows, axis=0, out=target[:, :GRID_FEATURES])
		target[:, GRID_FEATURES] = self.change_x * (1.0 / PLAYER_SPEED)
		target[:, GRID_FEATURES + 1] = self.change_y * (1.0 / MAX_FALL_SPEED)
		target[:, GRID_FEATURES + 2] = self.on_ground()
		return target

	def step(self, actions: NDArray[np.int64], tick: int) -> None:
		"""Applies one action per player then advances the physics by one tick."""
		alive = self.alive()

		# One table lookup instead of a mask per direction; jumping leaves the horizontal speed alone
		np.copyto(self.change_x, MOVE_SPEEDS[actions], where=alive & (actions != MOVE_JUMP))
		self._jump(alive & (actions == MOVE_JUMP))

		# Falling players accelerate, resting ones get the initial nudge that unsticks them from the floor
		falling = alive & (self.change_y != 0.0)
		np.copyto(self.change_y, 1.0, where=alive & ~falling)
		np.add(self.change_y, PLAYER_GRAVITY, out=self.change_y, where=falling)

		np.add(self.x, self.change_x, out=self.x, where=alive)
		np.trunc(self.x, out=self.x, where=alive)
		self._resolve_horizontal(alive)

		self.dead |= alive & (self.y >= self.death_y)
		alive = self.alive()

		np.add(self.y, self.change_y, out=self.y, where=alive)
		np.trunc(self.y, out=self.y, where=alive)
		# x is final once the horizontal pass is done, so the last two passes share its tile columns
		columns = self._resolve_vertical(alive)
		self._collect_rewards(alive, tick, columns)

	def _jump(self, mask: NDArray[np.bool_]) -> None:
		if not mask.any():
			return
		cells = self._cells(self.y + 2.0)
		row = cells[1] * self._padded_width
		solid = self._flat_solid
		grounded = np.take(solid, row + cells[2]) | np.take(solid, row + cells[3])
		np.copyto(self.change_y, PLAYER_JUMP_STRENGTH, where=mask & grounded)

	def _resolve_horizontal(self, alive: NDArray[np.bool_]) -> None:
		cells = self._cells(self.y)
		rows = self._row_offsets(cells)
		left, right = cells[2], cells[3]

		# A player one row tall gathers the same row twice, so the two rows can be OR'd without a guard
		solid = self._flat_solid
		hit_left = np.take(solid, rows[0] + left) | np.take(solid, rows[1] + left)
		hit_right = (np.take(solid, rows[0] + right) | np.take(solid, rows[1] + right)) & (right > left)

		# The blocking column and the side of the player touching it are one select each way
		going_right = self.change_x > 0
		column = np.where(going_right, np.where(hit_right, right, left), np.where(hit_left, left, right))
		snapped = column * TILE_SIZE + np.where(going_right, -PLAYER_W - GRID_ORIGIN, TILE_SIZE - GRID_ORIGIN)
		np.copyto(self.x, snapped, where=alive & (hit_left | hit_right) & (self.change_x != 0))

	def _resolve_vertical(self, alive: NDArray[np.bool_]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
		"""Resolves the vertical move and hands back the tile columns, which the reward pass reuses as is."""
		cells = self._cells(self.y)
		rows = self._row_offsets(cells)
		top, bottom = cells[0], cells[1]
		left, right = cells[2], cells[3]

		# A player one column wide gathers the same column twice, so the two columns can be OR'd without a guard
		solid = self._flat_solid
		hit_top = np.take(solid, rows[0] + left) | np.take(solid, rows[0] + right)
		hit_bottom = (np.take(solid, rows[1] + left) | np.take(solid, rows[1] + right)) & (bottom > top)

		going_down = self.change_y > 0
		row = np.where(going_down, np.where(hit_bottom, bottom, top), np.where(hit_top, top, bottom))
		snapped = row * TILE_SIZE + (self.offset_y - GRID_ORIGIN) + np.where(going_down, -PLAYER_H, TILE_SIZE)

		blocked = alive & (hit_top | hit_bottom) & (self.change_y != 0)
		np.copyto(self.y, snapped, where=blocked)
		np.copyto(self.change_y, 0.0, where=blocked)
		return left, right

	def _collect_rewards(self, alive: NDArray[np.bool_], tick: int, columns: tuple[NDArray[np.int64], NDArray[np.int64]]) -> None:
		left, right = columns
		rows = self._row_offsets(self._cells(self.y))

		# Reward tiles are rare, so the whole gather is skipped unless someone actually stands next to one
		if not np.take(self._flat_near_reward, rows[0] + left).any():
			return

		values = self._flat_reward
		best = np.maximum(
			np.maximum(np.take(values, rows[0] + left), np.take(values, rows[0] + right)),
			np.maximum(np.take(values, rows[1] + left), np.take(values, rows[1] + right)),
		)

		touched = alive & (best != 0)
		np.copyto(self.finished_reward, best, where=touched)
		won = touched & (best == 1)
		np.copyto(self.win_tick, tick, where=won)
		self.win |= won

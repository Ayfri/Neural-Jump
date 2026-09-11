import math
from pathlib import Path
from typing import Final

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from game.constants import AGENT_VISION_DISTANCE, MOVE_IDLE, MOVE_JUMP, MOVE_LEFT, MOVE_RIGHT
from game.settings import (
	ENEMY_HEIGHT, ENEMY_SPEED, ENEMY_WAKE_DISTANCE, ENEMY_WIDTH, PLAYER_GRAVITY, PLAYER_HEIGHT, PLAYER_JUMP_STRENGTH,
	PLAYER_SPEED, PLAYER_WIDTH, SCREEN_HEIGHT, STOMP_BOUNCE, TILE_SIZE,
)
from game.tiles import TILE_CHARS, TILE_REWARDS, TileKind

PLAYER_W: Final[int] = int(PLAYER_WIDTH)
PLAYER_H: Final[int] = int(PLAYER_HEIGHT)
ENEMY_W: Final[int] = int(ENEMY_WIDTH)
ENEMY_H: Final[int] = int(ENEMY_HEIGHT)
GRID_SIDE: Final[int] = AGENT_VISION_DISTANCE * 2 + 1
GRID_TILES: Final[int] = GRID_SIDE * GRID_SIDE  # One solid flag each, which is the whole terrain
NEAREST_FEATURES: Final[int] = 3  # in view, its offset in x and y, for the closest tile of one kind
WINDOW_FEATURES: Final[int] = GRID_TILES + 2 * NEAREST_FEATURES  # The closest goal tile, then the closest coin
ENEMY_FEATURES: Final[int] = 4  # in view, its offset in x and y, its heading, for the closest enemy
PLAYER_START: Final[int] = WINDOW_FEATURES + ENEMY_FEATURES  # Where the player's own features begin
PLAYER_FEATURES: Final[int] = 3  # change_x normalised, change_y normalised, on ground
OBSERVATION_SIZE: Final[int] = PLAYER_START + PLAYER_FEATURES
ENEMY_REACH: Final[float] = (AGENT_VISION_DISTANCE + 0.5) * TILE_SIZE  # Centre offset under which an enemy is inside the vision window
ENEMY_SCALE: Final[float] = 1.0 / (AGENT_VISION_DISTANCE * TILE_SIZE)  # Normalises an enemy offset like a tile offset
ENEMY_PATH_TICKS: Final[int] = 8192  # Ticks of every enemy path baked at load, past which an enemy stays where it got to
# Observations are flags and small normalised ratios, so half precision keeps every bit that matters while
# halving the host-to-device copy done every tick.
OBSERVATION_DTYPE: Final[np.dtype[np.float16]] = np.dtype(np.float16)
MAX_FALL_SPEED: Final[float] = 20.0  # Normalisation divisor for the vertical speed feature
ON_GROUND_SPEED: Final[float] = 2.0  # Vertical speed under which the player counts as grounded
DEATH_ROW_MARGIN: Final[int] = 2  # Rows above the bottom of the map that kill the player
GRID_PADDING: Final[int] = 32  # Air border baked around the grids so lookups never need bounds checks
GRID_ORIGIN: Final[int] = GRID_PADDING * TILE_SIZE  # Pixels the padding adds to a tile coordinate
MIN_SPAWN_SPACING: Final[int] = 3  # Tiles under which two rungs of the spawn ladder are the same place

MOVE_SPEEDS: Final[NDArray[np.float64]] = np.zeros(MOVE_IDLE + 1)  # Horizontal speed per action, jumping keeps the current one
MOVE_SPEEDS[[MOVE_LEFT, MOVE_RIGHT]] = (-PLAYER_SPEED, PLAYER_SPEED)


def search_maps_folder(folder: str | Path) -> Path:
	"""Returns the absolute path to the maps folder, walking up from this file until it is found."""
	current_folder = Path(__file__).resolve().parent
	while True:
		candidate = current_folder / folder
		if candidate.exists():
			return candidate
		parent = current_folder.parent
		if parent == current_folder:
			return Path()
		current_folder = parent


def resolve_map_path(map_path: str) -> Path:
	"""Resolves a map path like 'maps/level_1.txt' to an absolute path."""
	path = Path(map_path)
	return search_maps_folder(path.parent) / path.name


def list_maps() -> list[str]:
	"""Every level under the maps folder, as the relative paths `--map` takes, in name order."""
	folder = search_maps_folder('maps')
	return sorted(path.relative_to(folder.parent).as_posix() for path in folder.rglob('*.txt'))


def closest_tile(windows: NDArray[np.float32 | np.bool_]) -> NDArray[np.float32]:
	"""
	Summarises the non-zero tiles of a window: whether one is in view and where it sits.

	A map holds a handful of goal tiles in total, so a channel per tile spends most of the observation
	saying "still nothing here". Three numbers carry what a player can act on instead, and point at the tile
	directly rather than leaving the network to read a position out of a one-hot grid.
	"""
	rows, columns = np.divmod(np.arange(GRID_TILES), GRID_SIDE)
	rows = rows - AGENT_VISION_DISTANCE
	columns = columns - AGENT_VISION_DISTANCE
	# Tiles walked nearest first, so the first hit of the scan is the closest one
	order = np.argsort(rows ** 2 + columns ** 2, kind='stable')

	found = windows[:, order] != 0
	closest = order[found.argmax(axis=1)]
	in_view = found.any(axis=1)

	features = np.zeros((len(windows), NEAREST_FEATURES), dtype=np.float32)
	features[:, 0] = in_view
	features[:, 1] = columns[closest] / AGENT_VISION_DISTANCE * in_view
	features[:, 2] = rows[closest] / AGENT_VISION_DISTANCE * in_view
	return features


def near_grid(grid: NDArray[np.bool_]) -> NDArray[np.bool_]:
	"""True where any of the 2x2 tiles starting at (row, column) is set, which is the box a player can touch."""
	near = grid.copy()
	near[:-1] |= grid[1:]
	near[:, :-1] |= near[:, 1:].copy()
	return near


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
		self.win_tick = np.full(count, -1, dtype=np.int32)
		self.coins = np.zeros(count, dtype=np.int32)
		self.collected = np.zeros((count, max(1, self.coin_count)), dtype=np.bool_)
		self.enemy_step = np.full((count, self.enemy_count), -1, dtype=np.int64)  # Ticks each enemy has walked for this agent, -1 asleep
		self.stomped = np.zeros((count, self.enemy_count), dtype=np.bool_)
		self._agents = np.arange(count)
		self._enemy_columns = np.arange(self.enemy_count)

		self._observation = np.zeros((count, OBSERVATION_SIZE), dtype=OBSERVATION_DTYPE)
		self._offsets = np.arange(-AGENT_VISION_DISTANCE, AGENT_VISION_DISTANCE + 1)
		self._box = np.zeros((4, count), dtype=np.float64)  # Scratch the collision passes rebuild every call
		self._corners = np.zeros((4, count), dtype=np.int64)  # Scratch the coin pass gathers its four corners through
		self._box_extent = np.array([[PLAYER_H - 1.0], [PLAYER_W - 1.0]])  # Bottom row and right column, off the top row and the left column
		self._cell_limits = np.array([[self.max_row], [self.max_row], [self.max_column], [self.max_column]])

	def _pixels(self, cell: NDArray[np.int64]) -> tuple[int, int]:
		"""The top-left pixel of a (row, column) map cell, in the same space the players live in."""
		return int(cell[1]) * TILE_SIZE, int(cell[0]) * TILE_SIZE + self.offset_y

	def _load_map(self, map_path: str) -> None:
		with resolve_map_path(map_path).open() as file:
			lines = [line.strip() for line in file.readlines() if line.strip()]

		self.height = len(lines)
		self.width = max(len(line) for line in lines)
		self.offset_y = SCREEN_HEIGHT - self.height * TILE_SIZE
		self.death_y = (self.height - DEATH_ROW_MARGIN) * TILE_SIZE + self.offset_y

		# One character-code lookup per row turns the whole map into its tile kinds, without a dict hit per cell
		lookup = np.zeros(256, dtype=np.uint8)
		for char, kind in TILE_CHARS.items():
			lookup[ord(char)] = kind
		self.kinds = np.zeros((self.height, self.width), dtype=np.uint8)
		for y, line in enumerate(lines):
			codes = np.frombuffer(line.encode('ascii', 'replace'), dtype=np.uint8)
			self.kinds[y, :codes.size] = lookup[codes]

		self.solid = self.kinds == TileKind.SOLID
		# The flag is the one tile a player wins on, kept as a float grid because the vision windows read it
		self.goal = np.zeros((self.height, self.width), dtype=np.float32)
		self.goal[self.kinds == TileKind.FLAG] = TILE_REWARDS[TileKind.FLAG]

		# Coins are numbered in reading order, so an agent's collected set is one bool per coin
		coins = np.argwhere(self.kinds == TileKind.COIN)
		self.coin_ids = np.full((self.height, self.width), -1, dtype=np.int64)
		self.coin_ids[coins[:, 0], coins[:, 1]] = np.arange(len(coins))
		self.coin_count = len(coins)
		self.coin_positions: list[tuple[int, int]] = [self._pixels(cell) for cell in coins]

		# The near edge of the flag, which is how far along the level a run has to get to have finished it
		flags = np.argwhere(self.kinds == TileKind.FLAG)
		self.goal_x = int(flags[:, 1].min()) * TILE_SIZE if len(flags) else self.width * TILE_SIZE
		spawns = np.argwhere(self.kinds == TileKind.SPAWN)
		self.spawn_point = self._pixels(spawns[0]) if len(spawns) else (0, 0)
		self.checkpoints: list[tuple[int, int]] = [self._pixels(cell) for cell in np.argwhere(self.kinds == TileKind.CHECKPOINT)]

		# Padded copies: any tile index is clamped into the air border instead of being bounds-checked
		self.padded_solid = np.pad(self.solid, GRID_PADDING)
		self.padded_goal = np.pad(self.goal, GRID_PADDING)
		self.padded_coins = np.pad(self.coin_ids, GRID_PADDING, constant_values=-1)
		self.max_row = self.height + 2 * GRID_PADDING - 1
		self.max_column = self.width + 2 * GRID_PADDING - 1
		# Collision lookups index these flat views: `take` on one flat array beats a broadcast fancy index
		self.padded_width = self.width + 2 * GRID_PADDING
		self.flat_solid = self.padded_solid.ravel()
		self.flat_goal = self.padded_goal.ravel()
		self.flat_coins = self.padded_coins.ravel()
		# One lookup on the 2x2 box a player can touch then skips the whole goal or coin pass
		self.flat_near_goal = near_grid(self.padded_goal != 0).ravel()
		self.flat_near_coin = near_grid(self.padded_coins >= 0).ravel()

		# Every window of the map, baked once: an observation is then a single gather of contiguous rows
		# instead of a broadcast fancy index rebuilt per tick.
		solid_windows = sliding_window_view(self.padded_solid, (GRID_SIDE, GRID_SIDE))
		goal_windows = sliding_window_view(self.padded_goal, (GRID_SIDE, GRID_SIDE))
		coin_windows = sliding_window_view(self.padded_coins >= 0, (GRID_SIDE, GRID_SIDE))
		self.max_window_row = solid_windows.shape[0] - 1
		self.window_stride = solid_windows.shape[1]
		self.max_window_column = self.window_stride - 1

		self.windows = np.empty((solid_windows.shape[0] * self.window_stride, WINDOW_FEATURES), dtype=OBSERVATION_DTYPE)
		self.windows[:, :GRID_TILES] = solid_windows.reshape(-1, GRID_TILES)
		self.windows[:, GRID_TILES:GRID_TILES + NEAREST_FEATURES] = closest_tile(goal_windows.reshape(-1, GRID_TILES))
		# Coins are baked like the terrain, so a window still points at one the agent has already taken
		self.windows[:, GRID_TILES + NEAREST_FEATURES:] = closest_tile(coin_windows.reshape(-1, GRID_TILES))
		self._bake_enemy_paths(np.argwhere(self.kinds == TileKind.ENEMY))

	def _bake_enemy_paths(self, cells: NDArray[np.int64]) -> None:
		"""
		Where every enemy stands on each tick after it wakes up, baked once for the whole map.

		An enemy never reacts to a player: it walks at a fixed speed, turns around on a wall and falls off a
		ledge. Its path only depends on how long it has walked, so a player's whole enemy state is one step
		counter per enemy, and both simulation paths read the same tables. Stomped is the only other bit.
		"""
		self.enemy_count = count = len(cells)
		self.enemy_spawn_x = (cells[:, 1] * TILE_SIZE + (TILE_SIZE - ENEMY_W) // 2).astype(np.float64)
		spawn_y = cells[:, 0] * TILE_SIZE + self.offset_y + TILE_SIZE - ENEMY_H
		paths = np.empty((ENEMY_PATH_TICKS, count, 4))
		ticks = np.arange(ENEMY_PATH_TICKS)
		for enemy in range(count):
			path, loop = self._walk_enemy(float(self.enemy_spawn_x[enemy]), float(spawn_y[enemy]))
			# Past its last walked tick a path replays its loop, which is its last row alone once it is dead
			paths[:, enemy] = np.asarray(path)[np.where(ticks < len(path), ticks, loop + (ticks - loop) % (len(path) - loop))]
		# Flat, so the (agents, enemies) lookup of a tick is one `take` on step * count + enemy
		self.enemy_path_x, self.enemy_path_y = paths[..., 0].ravel(), paths[..., 1].ravel()
		self.enemy_path_live, self.enemy_path_heading = paths[..., 2].ravel() != 0, paths[..., 3].ravel()

	def _walk_enemy(self, x: float, y: float) -> tuple[list[tuple[float, float, float, float]], int]:
		"""
		One enemy's path as `(x, y, live, heading)` rows, and the row it loops back to.

		The walk stops on the first state it has already been in, since everything after it replays the loop from
		there, or on its death. An enemy pacing between two pipes repeats within a few hundred ticks, so a map
		bakes without stepping every tick of the table. Each tick is a player's, in the same order: gravity, the
		horizontal move and its snap, the death row, then the fall and its snap.
		"""
		solid, width, top = self.flat_solid, self.padded_width, self.offset_y - GRID_ORIGIN

		def row_of(py: float) -> int:
			return min(max(math.floor((py - top) / TILE_SIZE), 0), self.max_row)

		def column_of(px: float) -> int:
			return min(max(math.floor((px + GRID_ORIGIN) / TILE_SIZE), 0), self.max_column)

		change_x, change_y, live = -float(ENEMY_SPEED), 0.0, True  # Heading for the player, the way every enemy in SMB starts
		path: list[tuple[float, float, float, float]] = []
		seen: dict[tuple[float, float, float, float], int] = {}
		while len(path) < ENEMY_PATH_TICKS:
			state = (x, y, change_x, change_y)
			if live and state in seen:
				return path, seen[state]
			seen[state] = len(path)
			path.append((x, y, float(live), math.copysign(1.0, change_x)))
			if not live:
				break
			change_y = change_y + PLAYER_GRAVITY if change_y != 0.0 else 1.0

			x = float(math.trunc(x + change_x))
			column = column_of(x + (ENEMY_W - 1 if change_x > 0 else 0))
			if solid[row_of(y) * width + column] or solid[row_of(y + ENEMY_H - 1) * width + column]:
				x = float(column * TILE_SIZE - ENEMY_W - GRID_ORIGIN if change_x > 0 else (column + 1) * TILE_SIZE - GRID_ORIGIN)
				change_x = -change_x

			if y >= self.death_y:
				live = False
				continue
			y = float(math.trunc(y + change_y))
			row = row_of(y + (ENEMY_H - 1 if change_y > 0 else 0))
			if solid[row * width + column_of(x)] or solid[row * width + column_of(x + ENEMY_W - 1)]:
				y = float((row * TILE_SIZE - ENEMY_H if change_y > 0 else (row + 1) * TILE_SIZE) + top)
				change_y = 0.0
		return path, len(path) - 1

	def enemy_lookup(self, steps: NDArray[np.int64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
		"""Position, whether it is still on the map, and heading of each enemy at its step, a sleeping one at its spawn."""
		index = np.maximum(steps, 0) * self.enemy_count + self._enemy_columns
		return self.enemy_path_x.take(index), self.enemy_path_y.take(index), self.enemy_path_live.take(index), self.enemy_path_heading.take(index)

	def ground_spawns(self, spacing: int) -> list[tuple[int, int]]:
		"""
		Somewhere to restart an episode roughly every `spacing` tiles, in map order, starting with the spawn point.

		A level carries a handful of hand-placed checkpoints, which is far too coarse a ladder to climb down:
		the last one here is still 169 tiles from the flag. The geometry gives a finer one for free. A column's
		landing is the lowest solid tile with two clear rows over it, which is the main floor where there is
		one and the platform bridging a pit where there is not, and a column with no landing at all is skipped.
		"""
		clear = ~self.solid[:-2] & ~self.solid[1:-1] & self.solid[2:]  # Rows r-2 and r-1 empty over a solid row r
		landing = np.where(clear.any(axis=0), self.height - 1 - clear[::-1].argmax(axis=0), -1)
		# The near edge of every surface, so a rung lands on the platforms a pit is crossed by rather than
		# straddling the whole crossing: those are the jumps the ladder exists to break up
		edge = np.r_[True, landing[:-1] < 0]

		points = [self.spawn_point]
		last = self.spawn_point[0] // TILE_SIZE
		for column in np.flatnonzero(landing >= 0):
			if column - last >= spacing or (edge[column] and column - last >= MIN_SPAWN_SPACING):
				points.append((int(column) * TILE_SIZE, int(landing[column] - 1) * TILE_SIZE + self.offset_y))
				last = int(column)
		return points

	def reset(self, spawn_x: int, spawn_y: int) -> None:
		"""Places every player on the given spawn point and clears their state."""
		self.x.fill(spawn_x)
		self.y.fill(spawn_y)
		self.change_x.fill(0.0)
		self.change_y.fill(0.0)
		self.dead.fill(False)
		self.win.fill(False)
		self.win_tick.fill(-1)
		self.coins.fill(0)
		self.collected.fill(False)
		self.enemy_step.fill(-1)
		self.stomped.fill(False)

	def alive(self) -> NDArray[np.bool_]:
		return ~(self.dead | self.win)

	def kill(self, mask: NDArray[np.bool_]) -> None:
		self.dead |= mask

	def _cells(self, y: NDArray[np.float64], down: float = 0.0) -> NDArray[np.int64]:
		"""
		The four tiles the player box touches at (x, y): top row, bottom row, left column, right column.

		They come back as one (4, count) block of padded grid coordinates, built in a handful of numpy calls
		over the whole block, because on 300 players a call costs far more than the arithmetic inside it.
		Coordinates are clamped into the air border, which only ever moves a player already off the map.
		"""
		# The padding is added here in pixels, so the coordinates come out of the floor already padded
		box = self._box
		np.subtract(y, self.offset_y - GRID_ORIGIN - down, out=box[0])
		np.add(self.x, GRID_ORIGIN, out=box[2])
		# The far side of the box is one extent past the near side on both axes, so both come out of one add
		np.add(box[::2], self._box_extent, out=box[1::2])

		np.divide(box, TILE_SIZE, out=box)
		np.floor(box, out=box)
		cells = box.astype(np.int64)
		np.maximum(cells, 0, out=cells)
		np.minimum(cells, self._cell_limits, out=cells)
		return cells

	def _row_offsets(self, cells: NDArray[np.int64]) -> NDArray[np.int64]:
		"""The two rows of a `_cells` block as offsets into the flat padded grid."""
		return cells[:2] * self.padded_width

	def _touch_block(self, y: NDArray[np.float64]) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
		"""The flat row offsets and the two tile columns of the box at `y`, which is all a grid lookup reads."""
		cells = self._cells(y)
		return self._row_offsets(cells), cells[2], cells[3]

	def on_ground(self) -> NDArray[np.bool_]:
		"""Players whose vertical speed is small enough to count as standing on something."""
		return np.abs(self.change_y) <= ON_GROUND_SPEED

	def observe(self, out: NDArray[np.float16] | None = None) -> NDArray[np.float16]:
		"""
		Fills `out` (or the world's own buffer) with the (count, 62) observation: the solid flag of every
		tile in the 7x7 window around the player, where the closest goal tile, coin and enemy in it sit, and
		the player's own speed and ground state.
		"""
		target = self._observation if out is None else out
		tile_x = tile_of(self.x + PLAYER_W / 2)
		tile_y = tile_of(self.y + (PLAYER_H / 2 - self.offset_y))

		# The window table is indexed by its top-left corner, so the centre clamp becomes a corner clamp
		rows = np.clip(tile_y + GRID_PADDING - AGENT_VISION_DISTANCE, 0, self.max_window_row)
		columns = np.clip(tile_x + GRID_PADDING - AGENT_VISION_DISTANCE, 0, self.max_window_column)
		rows *= self.window_stride
		rows += columns

		self.windows.take(rows, axis=0, out=target[:, :WINDOW_FEATURES])
		self._enemy_features(target[:, WINDOW_FEATURES:PLAYER_START])
		target[:, PLAYER_START] = self.change_x * (1.0 / PLAYER_SPEED)
		target[:, PLAYER_START + 1] = self.change_y * (1.0 / MAX_FALL_SPEED)
		target[:, PLAYER_START + 2] = self.on_ground()
		return target

	def _enemy_features(self, out: NDArray[np.float16]) -> None:
		"""The closest enemy inside the vision window, as `in view, dx, dy, heading`, all zero when there is none."""
		if not self.enemy_count:
			out.fill(0)
			return
		x, y, live, heading = self.enemy_lookup(self.enemy_step)
		dx = x + (ENEMY_W / 2 - PLAYER_W / 2) - self.x[:, None]
		dy = y + (ENEMY_H / 2 - PLAYER_H / 2) - self.y[:, None]
		visible = live & ~self.stomped & (np.abs(dx) <= ENEMY_REACH) & (np.abs(dy) <= ENEMY_REACH)
		nearest = np.where(visible, dx * dx + dy * dy, np.inf).argmin(axis=1)
		agents = self._agents
		seen = visible[agents, nearest]
		out[:, 0] = seen
		out[:, 1] = dx[agents, nearest] * ENEMY_SCALE * seen
		out[:, 2] = dy[agents, nearest] * ENEMY_SCALE * seen
		out[:, 3] = heading[agents, nearest] * seen

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

		# A stomp is told apart from a side hit by where the player's feet were before this tick's fall
		descending = self.change_y > 0
		previous_bottom = self.y + PLAYER_H
		np.add(self.y, self.change_y, out=self.y, where=alive)
		np.trunc(self.y, out=self.y, where=alive)
		self._resolve_vertical(alive)
		# Both remaining passes read the same tiles, so the box is resolved once here rather than inside each
		block = self._touch_block(self.y)
		self._touch_goal(alive, tick, block)
		self._collect_coins(alive, block)
		self._touch_enemies(descending, previous_bottom)

	def grounded(self) -> NDArray[np.bool_]:
		"""Players with solid ground right under their feet, which is the only state a jump fires from."""
		cells = self._cells(self.y, 2.0)
		row = cells[1] * self.padded_width
		solid = self.flat_solid
		return solid.take(row + cells[2]) | solid.take(row + cells[3])

	def _jump(self, mask: NDArray[np.bool_]) -> None:
		if not mask.any():
			return
		np.copyto(self.change_y, PLAYER_JUMP_STRENGTH, where=mask & self.grounded())

	def _resolve_horizontal(self, alive: NDArray[np.bool_]) -> None:
		rows, left, right = self._touch_block(self.y)

		# A player one row tall gathers the same row twice, so the two rows can be OR'd without a guard
		solid = self.flat_solid
		hit_left = solid.take(rows[0] + left) | solid.take(rows[1] + left)
		hit_right = (solid.take(rows[0] + right) | solid.take(rows[1] + right)) & (right > left)

		# The blocking column and the side of the player touching it are one select each way
		going_right = self.change_x > 0
		column = np.where(going_right, np.where(hit_right, right, left), np.where(hit_left, left, right))
		snapped = column * TILE_SIZE + np.where(going_right, -PLAYER_W - GRID_ORIGIN, TILE_SIZE - GRID_ORIGIN)
		np.copyto(self.x, snapped, where=alive & (hit_left | hit_right) & (self.change_x != 0))

	def _resolve_vertical(self, alive: NDArray[np.bool_]) -> None:
		cells = self._cells(self.y)
		rows = self._row_offsets(cells)
		top, bottom = cells[0], cells[1]
		left, right = cells[2], cells[3]

		# A player one column wide gathers the same column twice, so the two columns can be OR'd without a guard
		solid = self.flat_solid
		hit_top = solid.take(rows[0] + left) | solid.take(rows[0] + right)
		hit_bottom = (solid.take(rows[1] + left) | solid.take(rows[1] + right)) & (bottom > top)

		going_down = self.change_y > 0
		row = np.where(going_down, np.where(hit_bottom, bottom, top), np.where(hit_top, top, bottom))
		snapped = row * TILE_SIZE + (self.offset_y - GRID_ORIGIN) + np.where(going_down, -PLAYER_H, TILE_SIZE)

		blocked = alive & (hit_top | hit_bottom) & (self.change_y != 0)
		np.copyto(self.y, snapped, where=blocked)
		np.copyto(self.change_y, 0.0, where=blocked)

	def _touch_goal(self, alive: NDArray[np.bool_], tick: int, block: tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]) -> None:
		"""Wins the episode for every player whose box overlaps the flag, on the tick it got there."""
		rows, left, right = block

		# The flag is a handful of tiles on a whole map, so the gather is skipped unless someone stands next to it
		if not self.flat_near_goal.take(rows[0] + left).any():
			return

		goal = self.flat_goal
		hit = goal.take(rows[0] + left) != 0
		hit |= goal.take(rows[0] + right) != 0
		hit |= goal.take(rows[1] + left) != 0
		hit |= goal.take(rows[1] + right) != 0

		won = alive & hit
		np.copyto(self.win_tick, tick, where=won)
		self.win |= won

	def _collect_coins(self, alive: NDArray[np.bool_], block: tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]) -> None:
		"""Banks every coin the player box overlaps, once each: `collected` is one bit per agent per coin."""
		if not self.coin_count:
			return

		rows, left, right = block
		if not self.flat_near_coin.take(rows[0] + left).any():
			return

		# The four corners of the box are one gather rather than four, and one test rules the whole pass out
		corners = self._corners
		np.add(rows[0], left, out=corners[0])
		np.add(rows[0], right, out=corners[1])
		np.add(rows[1], left, out=corners[2])
		np.add(rows[1], right, out=corners[3])
		ids = self.flat_coins.take(corners)
		standing = ids >= 0
		standing &= alive
		if not standing.any():
			return

		# Clamped index: an agent standing on no coin reads slot 0 and is masked out anyway
		np.maximum(ids, 0, out=ids)
		agents = self._agents
		# Still one corner at a time, because a box wide enough to touch the same coin twice must only bank it once
		for slots, on_coin in zip(ids, standing):
			fresh = on_coin & ~self.collected[agents, slots]
			if not fresh.any():
				continue
			self.collected[agents[fresh], slots[fresh]] = True
			self.coins += fresh

	def _touch_enemies(self, descending: NDArray[np.bool_], previous_bottom: NDArray[np.float64]) -> None:
		"""
		Walks every awake enemy one step, wakes the ones a player came close to, then settles every contact.

		Landing on an enemy from above, feet over its middle on the tick before, stomps it for that agent
		alone and bounces the player off; any other contact kills the player.
		"""
		if not self.enemy_count:
			return
		alive = self.alive()
		living = alive[:, None]
		steps = self.enemy_step
		awake = steps >= 0
		np.add(steps, 1, out=steps, where=living & awake)
		np.minimum(steps, ENEMY_PATH_TICKS - 1, out=steps)
		np.copyto(steps, 0, where=living & ~awake & (self.x[:, None] + ENEMY_WAKE_DISTANCE >= self.enemy_spawn_x))

		x, y, live, _ = self.enemy_lookup(steps)
		player_x, player_y = self.x[:, None], self.y[:, None]
		hit = live & ~self.stomped & living
		hit &= (player_x < x + ENEMY_W) & (x < player_x + PLAYER_W) & (player_y < y + ENEMY_H) & (y < player_y + PLAYER_H)
		if not hit.any():
			return
		stomp = hit & descending[:, None] & (previous_bottom[:, None] <= y + ENEMY_H / 2)
		self.stomped |= stomp
		self.dead |= (hit & ~stomp).any(axis=1)
		np.copyto(self.change_y, STOMP_BOUNCE, where=stomp.any(axis=1) & ~self.dead)

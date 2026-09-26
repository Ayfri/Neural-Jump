from typing import Final

import numpy as np
import torch
from torch import Tensor

from game.constants import AGENT_VISION_DISTANCE, MOVE_JUMP
from game.settings import ENEMY_WAKE_DISTANCE, PLAYER_GRAVITY, PLAYER_JUMP_STRENGTH, PLAYER_SPEED, STOMP_BOUNCE, TILE_SIZE
from game.world import (
	ENEMY_H, ENEMY_PATH_TICKS, ENEMY_REACH, ENEMY_SCALE, ENEMY_W, GRID_ORIGIN, GRID_PADDING, MAX_FALL_SPEED, MOVE_SPEEDS,
	ON_GROUND_SPEED, PLAYER_H, PLAYER_START, PLAYER_W, WINDOW_FEATURES, World,
)

# Positions are truncated to whole pixels every tick, so the only precision that matters is the speed a
# fall accumulates: in float32 the gravity sum drifts enough to flip a truncation after about 90 ticks
POSITION_DTYPE: Final[torch.dtype] = torch.float64


class CudaWorld:
	"""
	The batched world on the device, stepping the same physics as `World` over the same map.

	Every operation is a whole-population tensor call with no host read and no data dependent branch, so a
	tick compiles into a handful of fused kernels and a whole action window can be captured as one CUDA
	graph. State lives here while a generation runs and is copied back into the numpy world on demand.
	"""

	def __init__(self, world: World, device: torch.device) -> None:
		self.world = world
		self.device = device
		self.count = world.count
		self.offset_y = float(world.offset_y)
		self.death_y = float(world.death_y)

		def constant(array: object, dtype: torch.dtype) -> Tensor:
			return torch.as_tensor(np.ascontiguousarray(array), device=device).to(dtype)

		self.solid = constant(world.flat_solid, torch.bool)
		self.goal = constant(world.flat_goal, torch.float32)
		self.coin_ids = constant(world.flat_coins, torch.int64)
		self.windows = constant(world.windows, torch.float16)
		self.move_speeds = constant(MOVE_SPEEDS, POSITION_DTYPE)
		self.enemy_count = world.enemy_count
		self.enemy_path_x = constant(world.enemy_path_x, POSITION_DTYPE)
		self.enemy_path_y = constant(world.enemy_path_y, POSITION_DTYPE)
		self.enemy_path_live = constant(world.enemy_path_live, torch.bool)
		self.enemy_path_heading = constant(world.enemy_path_heading, POSITION_DTYPE)
		self.enemy_spawn_x = constant(world.enemy_spawn_x, POSITION_DTYPE)
		self.enemy_columns = torch.arange(self.enemy_count, device=device)

		def zeros(dtype: torch.dtype) -> Tensor:
			return torch.zeros(self.count, device=device, dtype=dtype)

		self.x, self.y = zeros(POSITION_DTYPE), zeros(POSITION_DTYPE)
		self.change_x, self.change_y = zeros(POSITION_DTYPE), zeros(POSITION_DTYPE)
		self.dead, self.win = zeros(torch.bool), zeros(torch.bool)
		self.win_tick = torch.full((self.count,), -1, device=device, dtype=torch.int64)
		self.coins = zeros(torch.int32)
		self.collected = torch.zeros(self.count, max(1, world.coin_count), device=device, dtype=torch.bool)
		self.enemy_step = torch.full((self.count, self.enemy_count), -1, device=device, dtype=torch.int64)
		self.stomped = torch.zeros(self.count, self.enemy_count, device=device, dtype=torch.bool)
		# One tick counter per player rather than one for the batch, so an episode can end and restart per agent
		self.tick = torch.zeros(self.count, device=device, dtype=torch.int64)

	def reset(self, spawn_x: int, spawn_y: int) -> None:
		"""Places every player on the given spawn point and clears their state, tick counter included."""
		self.x.fill_(spawn_x)
		self.y.fill_(spawn_y)
		self.change_x.zero_()
		self.change_y.zero_()
		self.dead.zero_()
		self.win.zero_()
		self.win_tick.fill_(-1)
		self.coins.zero_()
		self.collected.zero_()
		self.enemy_step.fill_(-1)
		self.stomped.zero_()
		self.tick.zero_()

	def reset_where(self, mask: Tensor, spawn_x: Tensor, spawn_y: Tensor) -> None:
		"""
		Restarts only the players `mask` selects, each on its own spawn point, leaving the others untouched.

		Every write is a whole-population `where`, so a reset costs the same whoever it hits and stays inside
		a captured graph: this is what lets an agent that died carry on collecting experience from a fresh
		episode while the rest of the batch keeps playing theirs.
		"""
		keep = ~mask
		torch.where(mask, spawn_x, self.x, out=self.x)
		torch.where(mask, spawn_y, self.y, out=self.y)
		self.change_x *= keep
		self.change_y *= keep
		self.dead &= keep
		self.win &= keep
		torch.where(mask, torch.full_like(self.win_tick, -1), self.win_tick, out=self.win_tick)
		self.coins *= keep
		self.collected &= keep.unsqueeze(1)
		torch.where(mask.unsqueeze(1), torch.full_like(self.enemy_step, -1), self.enemy_step, out=self.enemy_step)
		self.stomped &= keep.unsqueeze(1)
		self.tick *= keep

	def sync(self) -> World:
		"""Writes the device state back into the numpy world the renderer and the fitness pass read."""
		world = self.world
		np.copyto(world.x, self.x.cpu().numpy())
		np.copyto(world.y, self.y.cpu().numpy())
		np.copyto(world.change_x, self.change_x.cpu().numpy())
		np.copyto(world.change_y, self.change_y.cpu().numpy())
		np.copyto(world.dead, self.dead.cpu().numpy())
		np.copyto(world.win, self.win.cpu().numpy())
		np.copyto(world.win_tick, self.win_tick.cpu().numpy().astype(np.int32))
		np.copyto(world.coins, self.coins.cpu().numpy())
		np.copyto(world.collected, self.collected.cpu().numpy())
		np.copyto(world.enemy_step, self.enemy_step.cpu().numpy())
		np.copyto(world.stomped, self.stomped.cpu().numpy())
		return world

	def enemy_lookup(self, steps: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
		"""Position, whether it is still on the map, and heading of each enemy at its step, a sleeping one at its spawn."""
		index = steps.clamp_min(0) * self.enemy_count + self.enemy_columns
		return self.enemy_path_x[index], self.enemy_path_y[index], self.enemy_path_live[index], self.enemy_path_heading[index]

	def alive(self) -> Tensor:
		return ~(self.dead | self.win)

	def kill(self, mask: Tensor) -> None:
		self.dead |= mask

	def on_ground(self) -> Tensor:
		"""Players whose vertical speed is small enough to count as standing on something."""
		return self.change_y.abs() <= ON_GROUND_SPEED

	def _cells(self, y: Tensor, down: float = 0.0) -> tuple[Tensor, Tensor, Tensor, Tensor]:
		"""The top row, bottom row, left column and right column of the player box, in padded grid coordinates."""
		top_edge = y - (self.offset_y - GRID_ORIGIN - down)
		left_edge = self.x + GRID_ORIGIN
		top = torch.div(top_edge, TILE_SIZE).floor_().to(torch.int64)
		bottom = torch.div(top_edge + (PLAYER_H - 1.0), TILE_SIZE).floor_().to(torch.int64)
		left = torch.div(left_edge, TILE_SIZE).floor_().to(torch.int64)
		right = torch.div(left_edge + (PLAYER_W - 1.0), TILE_SIZE).floor_().to(torch.int64)
		return (top.clamp_(0, self.world.max_row), bottom.clamp_(0, self.world.max_row),
				left.clamp_(0, self.world.max_column), right.clamp_(0, self.world.max_column))

	def _touch_block(self, y: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
		"""The two rows of the box as flat grid offsets, and its two columns, which is all a lookup reads."""
		top, bottom, left, right = self._cells(y)
		return top * self.world.padded_width, bottom * self.world.padded_width, left, right

	def grounded(self) -> Tensor:
		"""Players with solid ground right under their feet, which is the only state a jump fires from."""
		_, bottom, left, right = self._cells(self.y, 2.0)
		row = bottom * self.world.padded_width
		return self.solid[row + left] | self.solid[row + right]

	def step(self, actions: Tensor) -> None:
		"""Applies one action per player then advances the physics by one tick."""
		alive = self.alive()
		jumping = actions == MOVE_JUMP
		torch.where(alive & ~jumping, self.move_speeds[actions], self.change_x, out=self.change_x)
		jumped = alive & jumping & self.grounded()
		torch.where(jumped, torch.full_like(self.change_y, PLAYER_JUMP_STRENGTH), self.change_y, out=self.change_y)

		# Falling players accelerate, resting ones get the initial nudge that unsticks them from the floor
		falling = alive & (self.change_y != 0.0)
		torch.where(alive & ~falling, torch.ones_like(self.change_y), self.change_y, out=self.change_y)
		torch.where(falling, self.change_y + PLAYER_GRAVITY, self.change_y, out=self.change_y)

		torch.where(alive, (self.x + self.change_x).trunc_(), self.x, out=self.x)
		self._resolve_horizontal(alive)

		self.dead |= alive & (self.y >= self.death_y)
		alive = self.alive()

		descending = self.change_y > 0
		previous_bottom = self.y + PLAYER_H
		torch.where(alive, (self.y + self.change_y).trunc_(), self.y, out=self.y)
		self._resolve_vertical(alive)

		# Both remaining passes read the same tiles, so the box is resolved once here rather than inside each
		block = self._touch_block(self.y)
		self._touch_goal(alive, block)
		self._collect_coins(alive, block)
		self._touch_enemies(descending, previous_bottom)
		self.tick += 1

	def _touch_enemies(self, descending: Tensor, previous_bottom: Tensor) -> None:
		"""Walks the awake enemies, wakes the near ones and settles every contact, exactly as `World` does."""
		if not self.enemy_count:
			return
		living = self.alive().unsqueeze(1)
		awake = self.enemy_step >= 0
		steps = torch.where(living & awake, (self.enemy_step + 1).clamp_max(ENEMY_PATH_TICKS - 1), self.enemy_step)
		wake = living & ~awake & (self.x.unsqueeze(1) + ENEMY_WAKE_DISTANCE >= self.enemy_spawn_x)
		torch.where(wake, torch.zeros_like(steps), steps, out=self.enemy_step)

		x, y, live, _ = self.enemy_lookup(self.enemy_step)
		player_x, player_y = self.x.unsqueeze(1), self.y.unsqueeze(1)
		hit = live & ~self.stomped & living
		hit &= (player_x < x + ENEMY_W) & (x < player_x + PLAYER_W) & (player_y < y + ENEMY_H) & (y < player_y + PLAYER_H)
		stomp = hit & descending.unsqueeze(1) & (previous_bottom.unsqueeze(1) <= y + ENEMY_H / 2)
		self.stomped |= stomp
		self.dead |= (hit & ~stomp).any(dim=1)
		bounced = stomp.any(dim=1) & ~self.dead
		torch.where(bounced, torch.full_like(self.change_y, STOMP_BOUNCE), self.change_y, out=self.change_y)

	def _resolve_horizontal(self, alive: Tensor) -> None:
		top_row, bottom_row, left, right = self._touch_block(self.y)
		solid = self.solid
		hit_left = solid[top_row + left] | solid[bottom_row + left]
		hit_right = (solid[top_row + right] | solid[bottom_row + right]) & (right > left)

		going_right = self.change_x > 0
		column = torch.where(going_right, torch.where(hit_right, right, left), torch.where(hit_left, left, right))
		snapped = column * TILE_SIZE + torch.where(going_right, -PLAYER_W - GRID_ORIGIN, TILE_SIZE - GRID_ORIGIN)
		blocked = alive & (hit_left | hit_right) & (self.change_x != 0)
		torch.where(blocked, snapped.to(POSITION_DTYPE), self.x, out=self.x)

	def _resolve_vertical(self, alive: Tensor) -> None:
		top, bottom, left, right = self._cells(self.y)
		top_row, bottom_row = top * self.world.padded_width, bottom * self.world.padded_width
		solid = self.solid
		hit_top = solid[top_row + left] | solid[top_row + right]
		hit_bottom = (solid[bottom_row + left] | solid[bottom_row + right]) & (bottom > top)

		going_down = self.change_y > 0
		row = torch.where(going_down, torch.where(hit_bottom, bottom, top), torch.where(hit_top, top, bottom))
		snapped = row * TILE_SIZE + (self.offset_y - GRID_ORIGIN) + torch.where(going_down, float(-PLAYER_H), float(TILE_SIZE))

		blocked = alive & (hit_top | hit_bottom) & (self.change_y != 0)
		torch.where(blocked, snapped.to(POSITION_DTYPE), self.y, out=self.y)
		torch.where(blocked, torch.zeros_like(self.change_y), self.change_y, out=self.change_y)

	def _touch_goal(self, alive: Tensor, block: tuple[Tensor, Tensor, Tensor, Tensor]) -> None:
		"""Wins the episode for every player whose box overlaps the flag, on the tick it got there."""
		top_row, bottom_row, left, right = block
		goal = self.goal
		hit = (goal[top_row + left] != 0) | (goal[top_row + right] != 0)
		hit |= (goal[bottom_row + left] != 0) | (goal[bottom_row + right] != 0)

		won = alive & hit
		torch.where(won, self.tick, self.win_tick, out=self.win_tick)
		self.win |= won

	def _collect_coins(self, alive: Tensor, block: tuple[Tensor, Tensor, Tensor, Tensor]) -> None:
		"""Banks every coin the player box overlaps, once each: `collected` is one bit per agent per coin."""
		if not self.world.coin_count:
			return

		top_row, bottom_row, left, right = block
		# Still one corner at a time, because a box wide enough to touch the same coin twice must only bank it once
		for corner in (top_row + left, top_row + right, bottom_row + left, bottom_row + right):
			ids = self.coin_ids[corner]
			# Clamped index: an agent standing on no coin reads slot 0 and is masked out anyway
			slots = ids.clamp_min(0).unsqueeze(1)
			banked = self.collected.gather(1, slots).squeeze(1)
			fresh = (ids >= 0) & alive & ~banked
			self.collected.scatter_(1, slots, (banked | fresh).unsqueeze(1))
			self.coins += fresh

	def observe(self, out: Tensor) -> Tensor:
		"""
		Fills `out` with the same observation `World.observe` builds: the solid flag of every tile in the
		window around the player, where the closest goal tile and coin in it sit, and the player's own state.
		"""
		tile_x = torch.div(self.x + PLAYER_W / 2, TILE_SIZE).floor_().to(torch.int64)
		tile_y = torch.div(self.y + (PLAYER_H / 2 - self.offset_y), TILE_SIZE).floor_().to(torch.int64)

		# The window table is indexed by its top-left corner, so the centre clamp becomes a corner clamp
		rows = (tile_y + (GRID_PADDING - AGENT_VISION_DISTANCE)).clamp_(0, self.world.max_window_row) * self.world.window_stride
		rows += (tile_x + (GRID_PADDING - AGENT_VISION_DISTANCE)).clamp_(0, self.world.max_window_column)

		flat = out.view(self.count, -1)
		# An assignment rather than `out=`: the slice is not contiguous, and dynamo breaks the graph on that
		flat[:, :WINDOW_FEATURES] = self.windows.index_select(0, rows)
		self._enemy_features(flat[:, WINDOW_FEATURES:PLAYER_START])
		flat[:, PLAYER_START] = self.change_x * (1.0 / PLAYER_SPEED)
		flat[:, PLAYER_START + 1] = self.change_y * (1.0 / MAX_FALL_SPEED)
		flat[:, PLAYER_START + 2] = self.on_ground()
		return out

	def _enemy_features(self, out: Tensor) -> None:
		"""The closest enemy inside the vision window, as `in view, dx, dy, heading`, all zero when there is none."""
		if not self.enemy_count:
			out.zero_()
			return
		x, y, live, heading = self.enemy_lookup(self.enemy_step)
		dx = x + (ENEMY_W / 2 - PLAYER_W / 2) - self.x.unsqueeze(1)
		dy = y + (ENEMY_H / 2 - PLAYER_H / 2) - self.y.unsqueeze(1)
		visible = live & ~self.stomped & (dx.abs() <= ENEMY_REACH) & (dy.abs() <= ENEMY_REACH)
		nearest = torch.where(visible, dx * dx + dy * dy, torch.inf).argmin(dim=1, keepdim=True)
		seen = visible.gather(1, nearest).squeeze(1)
		out[:, 0] = seen
		out[:, 1] = dx.gather(1, nearest).squeeze(1) * ENEMY_SCALE * seen
		out[:, 2] = dy.gather(1, nearest).squeeze(1) * ENEMY_SCALE * seen
		out[:, 3] = heading.gather(1, nearest).squeeze(1) * seen

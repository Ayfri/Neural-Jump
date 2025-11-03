import pygame
from pygame.sprite import Sprite

from game.constants import AGENT_NEAR_PLATFORM_DISTANCE, AGENT_VISION_DISTANCE, MOVE_JUMP, MOVE_LEFT, MOVE_RIGHT
from game.level import Level
from game.platform import Platform
from game.settings import (
	BLACK, PLAYER_SPEED, PLAYER_WIDTH, PLAYER_HEIGHT, PLAYER_COLOR,
	PLAYER_GRAVITY, PLAYER_JUMP_STRENGTH, SCREEN_HEIGHT, TILE_SIZE,
)
from game.tiles import Tile, TILES


class Player(Sprite):
	level: Level

	def __init__(self, x: int, y: int) -> None:
		super().__init__()
		self.image = pygame.Surface((PLAYER_WIDTH, PLAYER_HEIGHT))
		self.image.fill(BLACK)
		self.image.fill(PLAYER_COLOR, rect=self.image.get_rect().inflate(-5, -5))
		self.rect = self.image.get_rect()
		self.rect.x = x
		self.rect.y = y
		self.change_x: float = 0.0
		self.change_y: float = 0.0
		self.dead = False
		self.finished_reward: int | None = None
		self.win = False
		self.win_tick: int | None = None
		self._near_platforms: list[Platform] = []

	def update(self, tick: int | None = None) -> None:
		if self.dead or self.win:
			return

		self.calc_grav()
		self.rect.x += self.change_x

		if self.check_death():
			return

		self.calculate_near_platforms()

		# Check horizontal collisions
		for block in self.rect.collideobjectsall(self._near_platforms):
			if not isinstance(block, Platform):
				continue
				
			if block.tile_data.get('reward', False):
				self.finished_reward = block.tile_data['reward']
				if block.tile_data['reward'] == 1:
					self.win = True
					if tick is not None:
						self.win_tick = tick
				break

			if not block.tile_data.get('is_solid', False):
				continue

			if self.change_x > 0:
				self.rect.right = block.rect.left
			elif self.change_x < 0:
				self.rect.left = block.rect.right

		self.rect.y += self.change_y

		# Check vertical collisions
		for block in self.rect.collideobjectsall(self._near_platforms):
			if not isinstance(block, Platform):
				continue
				
			if block.tile_data.get('reward', False):
				self.finished_reward = block.tile_data['reward']
				if block.tile_data['reward'] == 1:
					self.win = True
					if tick is not None:
						self.win_tick = tick
				break

			if not block.tile_data.get('is_solid', False):
				continue

			if self.change_y > 0:
				self.rect.bottom = block.rect.top
			elif self.change_y < 0:
				self.rect.top = block.rect.bottom

			self.change_y = 0.0

	def calc_grav(self) -> None:
		if self.change_y == 0.0:
			self.change_y = 1.0
		else:
			self.change_y += PLAYER_GRAVITY

	def check_death(self) -> bool:
		if self.rect.top >= (self.level.height - 2) * TILE_SIZE:
			self.set_dead()
			return True
		return False

	def set_dead(self) -> None:
		self.dead = True
		self.image.set_alpha(40)

	def jump(self) -> None:
		if self.check_death():
			return

		self.rect.y += 2
		platform_hit_list = self.rect.collideobjectsall(self._near_platforms)
		self.rect.y -= 2

		if platform_hit_list or self.rect.bottom >= SCREEN_HEIGHT:
			self.change_y = PLAYER_JUMP_STRENGTH

	def go_left(self) -> None:
		self.change_x = -PLAYER_SPEED

	def go_right(self) -> None:
		self.change_x = PLAYER_SPEED

	def stop(self) -> None:
		self.change_x = 0.0

	def revive(self) -> None:
		self.dead = False
		self.image.set_alpha(255)
		self.change_x = 0.0
		self.change_y = 0.0

	def get_surrounding_tiles(self) -> list[list[Tile]]:
		grid: list[list[Tile]] = []
		for dy in range(-AGENT_VISION_DISTANCE, AGENT_VISION_DISTANCE + 1):
			row: list[Tile] = []
			for dx in range(-AGENT_VISION_DISTANCE, AGENT_VISION_DISTANCE + 1):
				x = (self.rect.centerx // TILE_SIZE) + dx
				y = (self.rect.centery // TILE_SIZE) + dy
				if 0 <= x < self.level.width and 0 <= y < self.level.height and y < len(self.level.tile_map) and x < len(self.level.tile_map[y]):
					tile = self.level.tile_map[round(y)][round(x)]
					row.append(TILES[tile])
				else:
					row.append({})
			grid.append(row)
		return grid

	def calculate_near_platforms(self) -> None:
		"""Collect platforms near the player for collision detection"""
		platforms = [sprite for sprite in self.level.platform_list if isinstance(sprite, Platform)]
		self._near_platforms = [
			platform for platform in platforms
			if abs(platform.rect.centerx - self.rect.centerx) <= AGENT_NEAR_PLATFORM_DISTANCE 
			and abs(platform.rect.centery - self.rect.centery) <= AGENT_NEAR_PLATFORM_DISTANCE
		]

	def execute_move(self, direction: int) -> None:
		"""Execute movement (MOVE_JUMP: jump, MOVE_LEFT: left, MOVE_RIGHT: right)"""
		if direction == MOVE_JUMP:
			self.jump()
		elif direction == MOVE_LEFT:
			self.go_left()
		elif direction == MOVE_RIGHT:
			self.go_right()

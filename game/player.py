import pygame
from pygame.sprite import Sprite

from game.constants import AGENT_NEAR_PLATFORM_DISTANCE
from game.level import Level
from game.platform import Platform
from game.settings import (
	BLACK, PLAYER_SPEED, PLAYER_WIDTH, PLAYER_HEIGHT, PLAYER_COLOR,
	PLAYER_GRAVITY, PLAYER_JUMP_STRENGTH, SCREEN_HEIGHT, TILE_SIZE,
)


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

		# x is resolved fully before y moves at all, so a diagonal move can't tunnel through a corner
		for block in self.rect.collideobjectsall(self._near_platforms):
			if not isinstance(block, Platform):
				continue

			if block.tile_data.get('reward', False):
				self.finished_reward = block.tile_data['reward']
				if block.tile_data['reward'] == 1:
					self.win = True
					if tick is not None:
						self.win_tick = tick
				break  # A reward tile isn't solid, so nothing after it should push the player back out

			if not block.tile_data.get('is_solid', False):
				continue

			if self.change_x > 0:
				self.rect.right = block.rect.left
			elif self.change_x < 0:
				self.rect.left = block.rect.right

		self.rect.y += self.change_y

		for block in self.rect.collideobjectsall(self._near_platforms):
			if not isinstance(block, Platform):
				continue

			if block.tile_data.get('reward', False):
				self.finished_reward = block.tile_data['reward']
				if block.tile_data['reward'] == 1:
					self.win = True
					if tick is not None:
						self.win_tick = tick
				break  # Same as above: a reward tile is never solid

			if not block.tile_data.get('is_solid', False):
				continue

			if self.change_y > 0:
				self.rect.bottom = block.rect.top
			elif self.change_y < 0:
				self.rect.top = block.rect.bottom

			self.change_y = 0.0

	def calc_grav(self) -> None:
		if self.change_y == 0.0:
			# Nudge instead of leaving it at 0, otherwise a grounded player never reports as falling
			self.change_y = 1.0
		else:
			self.change_y += PLAYER_GRAVITY

	def check_death(self) -> bool:
		# 2 rows of margin: dying exactly at the bottom row would clip the sprite off-screen before the death shows
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

		# Probe one pixel below by nudging the rect down, since standing still means zero overlap with the ground
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

	def calculate_near_platforms(self) -> None:
		"""Collect platforms near the player for collision detection"""
		center_x, center_y = self.rect.centerx, self.rect.centery
		self._near_platforms = [
			platform for platform in self.level.platforms_in_range(center_x - AGENT_NEAR_PLATFORM_DISTANCE, center_x + AGENT_NEAR_PLATFORM_DISTANCE)
			if abs(platform.rect.centery - center_y) <= AGENT_NEAR_PLATFORM_DISTANCE
		]

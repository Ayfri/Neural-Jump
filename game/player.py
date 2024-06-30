# player.py

import pygame
from pygame.sprite import Sprite

from game.level import Level
from game.settings import (
	PLAYER_WIDTH,
	PLAYER_HEIGHT,
	PLAYER_COLOR,
	PLAYER_GRAVITY,
	PLAYER_JUMP_STRENGTH,
	SCREEN_HEIGHT,
	SCREEN_WIDTH,
	TILE_SIZE,
	SHIFT_THRESHOLD_X,
	SHIFT_THRESHOLD_Y
)


class Player(Sprite):
	level: Level

	def __init__(self, x: int, y: int) -> None:
		super().__init__()
		self.image = pygame.Surface((PLAYER_WIDTH, PLAYER_HEIGHT))
		self.image.fill(PLAYER_COLOR)
		self.rect = self.image.get_rect()
		self.rect.x = x
		self.rect.y = y
		self.change_x = 0
		self.change_y = 0

	def update(self) -> None:
		self.calc_grav()
		self.rect.x += self.change_x

		# Check for horizontal collisions
		block_hit_list = pygame.sprite.spritecollide(self, self.level.solid_platforms, False)
		for block in block_hit_list:
			if self.change_x > 0:
				self.rect.right = block.rect.left
			elif self.change_x < 0:
				self.rect.left = block.rect.right

		self.rect.y += self.change_y

		# Check for vertical collisions
		block_hit_list = pygame.sprite.spritecollide(self, self.level.solid_platforms, False)
		for block in block_hit_list:
			if self.change_y > 0:
				self.rect.bottom = block.rect.top
			elif self.change_y < 0:
				self.rect.top = block.rect.bottom

			self.change_y = 0

		# Horizontal shift
		if self.rect.right >= SCREEN_WIDTH - SHIFT_THRESHOLD_X:
			diff = self.rect.right - (SCREEN_WIDTH - SHIFT_THRESHOLD_X)
			self.rect.right = SCREEN_WIDTH - SHIFT_THRESHOLD_X
			self.level.shift_world(-diff, 0)
		elif self.rect.left <= SHIFT_THRESHOLD_X:
			# 	self.rect.left = SHIFT_THRESHOLD_X

			diff = SHIFT_THRESHOLD_X - self.rect.left
			# if level is at the left edge of the screen, do not shift
			if self.level.world_shift[0] > 0:
				self.rect.left = TILE_SIZE
				self.level.shift_world(diff, 0)
			else:
				self.rect.left = max(0, self.rect.left)

		# Vertical shift
		if self.rect.top <= SHIFT_THRESHOLD_Y:
			diff = SHIFT_THRESHOLD_Y - self.rect.top
			self.rect.top = SHIFT_THRESHOLD_Y
			self.level.shift_world(0, diff)
		elif self.rect.bottom >= SCREEN_HEIGHT - SHIFT_THRESHOLD_Y:
			diff = self.rect.bottom - (SCREEN_HEIGHT - SHIFT_THRESHOLD_Y)
			self.rect.bottom = SCREEN_HEIGHT - SHIFT_THRESHOLD_Y
			self.level.shift_world(0, -diff)

	def calc_grav(self) -> None:
		if self.change_y == 0:
			self.change_y = 1
		else:
			self.change_y += PLAYER_GRAVITY

		if self.rect.y >= SCREEN_HEIGHT - self.rect.height and self.change_y >= 0:
			self.change_y = 0
			self.rect.y = SCREEN_HEIGHT - self.rect.height

	def jump(self):
		self.rect.y += 2
		platform_hit_list = pygame.sprite.spritecollide(self, self.level.solid_platforms, False)
		self.rect.y -= 2

		if len(platform_hit_list) > 0 or self.rect.bottom >= SCREEN_HEIGHT:
			self.change_y = PLAYER_JUMP_STRENGTH

	def go_left(self) -> None:
		self.change_x = -6

	def go_right(self) -> None:
		self.change_x = 6

	def stop(self) -> None:
		self.change_x = 0

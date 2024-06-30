import pygame
from pygame.sprite import Sprite

from game.level import Level
from game.platform import Platform
from game.settings import (
	PLAYER_SPEED, PLAYER_WIDTH,
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
from game.tiles import Tile


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
		self.dead = False
		self.finished_reward: int | None = None

	def update(self) -> None:
		if self.dead or self.level.finished:
			return

		self.calc_grav()
		self.rect.x += self.change_x

		# Check for horizontal collisions
		block_hit_list = pygame.sprite.spritecollide(self, self.level.platform_list, False)
		for block in block_hit_list:
			if isinstance(block, Platform):
				if block.tile_data.get('reward', False):
					self.finished_reward = block.tile_data['reward']
					self.level.finished = True
					break

				if not block.tile_data.get('is_solid', False):
					continue

				if self.change_x > 0:
					self.rect.right = block.rect.left
				elif self.change_x < 0:
					self.rect.left = block.rect.right

		self.rect.y += self.change_y

		# Check for vertical collisions
		block_hit_list = pygame.sprite.spritecollide(self, self.level.platform_list, False)
		for block in block_hit_list:
			if isinstance(block, Platform):
				if block.tile_data.get('reward', False):
					self.finished_reward = block.tile_data['reward']
					self.level.finished = True
					break

				if not block.tile_data.get('is_solid', False):
					continue

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

		# check if player is below the screen
		if self.level.world_shift[1] < -SCREEN_HEIGHT / 2:
			self.dead = True
			self.image.set_alpha(64)

	def calc_grav(self) -> None:
		if self.change_y == 0:
			self.change_y = 1
		else:
			self.change_y += PLAYER_GRAVITY

		if self.rect.y >= SCREEN_HEIGHT - self.rect.height and self.change_y >= 0:
			self.change_y = 0
			self.rect.y = SCREEN_HEIGHT - self.rect.height

	def jump(self) -> None:
		self.rect.y += 2
		platform_hit_list = pygame.sprite.spritecollide(self, self.level.platform_list, False)
		self.rect.y -= 2

		if len(platform_hit_list) > 0 or self.rect.bottom >= SCREEN_HEIGHT:
			self.change_y = PLAYER_JUMP_STRENGTH

	def go_left(self) -> None:
		self.change_x = -PLAYER_SPEED

	def go_right(self) -> None:
		self.change_x = PLAYER_SPEED

	def stop(self) -> None:
		self.change_x = 0

	def revive(self) -> None:
		self.dead = False
		self.image.set_alpha(255)
		self.change_x = 0
		self.change_y = 0

	def get_surrounding_grid(self) -> list[list[Tile]]:
		grid: list[list[Tile]] = []
		for dy in range(-3, 4):
			row: list[Tile] = []
			for dx in range(-3, 4):
				x = (self.rect.x // TILE_SIZE) + dx
				y = (self.rect.y // TILE_SIZE) + dy
				if 0 <= x < self.level.width and 0 <= y < self.level.height:
					tile = self.level.tile_map[y][x]
					row += [tile]
				else:
					row += [{}]
			grid += [row]
		return grid

	def execute_move(self, direction: int):
		"""
		Exécute le mouvement basé sur la direction fournie par l'agent.
		:param direction: Direction du mouvement (0: haut, 1: bas, 2: gauche, 3: droite)
		"""
		if direction == 0:  # Haut
			self.jump()
		elif direction == 1:  # Bas
			self.change_y += 1  # Pas de mouvement spécifique vers le bas
		elif direction == 2:  # Gauche
			self.go_left()
		elif direction == 3:  # Droite
			self.go_right()

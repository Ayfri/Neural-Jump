import pygame
from pygame.sprite import Sprite

from game.level import Level
from game.platform import Platform
from game.settings import (
	BLACK, PLAYER_SPEED, PLAYER_WIDTH,
	PLAYER_HEIGHT,
	PLAYER_COLOR,
	PLAYER_GRAVITY,
	PLAYER_JUMP_STRENGTH,
	SCREEN_HEIGHT,
	TILE_SIZE,
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
		self.change_x = 0
		self.change_y = 0
		self.dead = False
		self.win = False
		self.finished_reward: int | None = None

	def update(self) -> None:
		if self.dead or self.win:
			return

		self.calc_grav()
		self.rect.x += self.change_x

		if self.check_death():
			return

		# Check for horizontal collisions
		block_hit_list = pygame.sprite.spritecollide(self, self.level.platform_list, False)
		for block in block_hit_list:
			if isinstance(block, Platform):
				if block.tile_data.get('reward', False):
					self.finished_reward = block.tile_data['reward']
					self.win = True
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

	def calc_grav(self) -> None:
		if self.change_y == 0:
			self.change_y = 1
		else:
			self.change_y += PLAYER_GRAVITY

	def check_death(self) -> bool:
		if self.rect.top >= (self.level.height - 2) * TILE_SIZE:
			self.dead = True
			self.image.set_alpha(64)
			return True

		return False

	def jump(self) -> None:
		if self.check_death():
			return

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
				x = (self.rect.centerx // TILE_SIZE) + dx
				y = (self.rect.centery // TILE_SIZE) + dy
				if 0 <= x < self.level.width and 0 <= y < self.level.height:
					# check if not index out of range
					if y >= len(self.level.tile_map) or x >= len(self.level.tile_map[y]):
						row += [{}]
						continue

					tile = self.level.tile_map[round(y)][round(x)]
					tile_data = TILES[tile]
					row += [tile_data]
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
		elif direction == 1:  # Gauche
			self.go_left()
		elif direction == 2:  # Droite
			self.go_right()

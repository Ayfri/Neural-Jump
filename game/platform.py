# platform.py

import pygame
from pygame.sprite import Sprite

from game.settings import TILE_SIZE
from game.tiles import Tile


class Platform(Sprite):
	def __init__(self, x: int, y: int, tile_data: Tile) -> None:
		super().__init__()
		self.image = pygame.Surface((TILE_SIZE, TILE_SIZE))
		self.tile_data = tile_data
		color = tile_data.get('color', (255, 255, 255))
		self.image.fill(color)
		self.rect = self.image.get_rect()
		self.rect.x = x
		self.rect.y = y

	def shift(self, shift_x: int, shift_y: int) -> None:
		self.rect.x += shift_x
		self.rect.y += shift_y

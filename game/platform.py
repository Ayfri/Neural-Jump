# platform.py

import pygame
from pygame.sprite import Sprite

from game.settings import RED, TILE_SIZE


class Platform(Sprite):
	def __init__(self, x: int, y: int, tile_data: dict) -> None:
		super().__init__()
		self.image = pygame.Surface((TILE_SIZE, TILE_SIZE))
		self.tile_data = tile_data
		self.image.fill(tile_data['color'])
		self.rect = self.image.get_rect()
		self.rect.x = x
		self.rect.y = y

	def shift(self, shift_x: int, shift_y: int) -> None:
		self.rect.x += shift_x
		self.rect.y += shift_y

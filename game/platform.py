import pygame
from pygame.sprite import Sprite

from game.settings import COIN_SHINE, TILE_SIZE
from game.tiles import Tile


class Platform(Sprite):
	def __init__(self, x: int, y: int, tile_data: Tile) -> None:
		super().__init__()
		self.tile_data = tile_data
		self.collected = False  # Coins only: the same coin is met once per axis pass
		color = tile_data.get('color', (255, 255, 255))
		if tile_data.get('is_coin', False):
			# A coin is drawn as a disc inside its tile, so a trail reads as dots instead of a wall
			self.image = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
			center = (TILE_SIZE // 2, TILE_SIZE // 2)
			pygame.draw.circle(self.image, color, center, 9)
			pygame.draw.circle(self.image, COIN_SHINE, (center[0] - 3, center[1] - 3), 3)
		else:
			self.image = pygame.Surface((TILE_SIZE, TILE_SIZE))
			self.image.fill(color)
		self.rect = self.image.get_rect()
		self.rect.x = x
		self.rect.y = y

	def shift(self, shift_x: int, shift_y: int) -> None:
		self.rect.x += shift_x
		self.rect.y += shift_y

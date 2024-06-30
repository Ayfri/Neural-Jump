import os

import pygame
from pygame import Surface
from pygame.sprite import Group

from game.platform import Platform
from game.settings import SCREEN_HEIGHT, TILE_SIZE, WHITE
from game.tiles import TILES


def _search_maps_folder(folder: str) -> str:
	"""
	Returns the absolute path to the maps folder.
	Search into the folder of the current script, then goes into parent folder until root folder.
	"""
	current_folder = os.path.dirname(os.path.abspath(__file__))
	while current_folder != '/':
		if folder in os.listdir(current_folder):
			return os.path.join(current_folder, folder)
		current_folder = os.path.dirname(current_folder)
	return ''


class Level:
	def __init__(self, player: 'game.Player') -> None:
		self.platform_list = Group()
		self.player = player
		self.world_shift = (0, 0)
		self.map: str | None = None
		player.level = self

	@property
	def platforms(self) -> list[Platform]:
		return self.platform_list.sprites()

	@property
	def solid_platforms(self) -> Group:
		return Group(platform for platform in self.platforms if platform.tile_data.get('is_solid', False))

	def update(self) -> None:
		self.platform_list.update()

	def draw(self, screen: Surface) -> None:
		screen.fill(WHITE)
		self.platform_list.draw(screen)

	def load_map(self, map_path: str) -> None:
		maps_folder = os.path.dirname(map_path)
		maps_path = _search_maps_folder(maps_folder) + os.sep + os.path.basename(map_path)
		with open(maps_path) as file:
			lines = file.readlines()

		self.map = map_path

		offset_y = SCREEN_HEIGHT - (len(lines) * TILE_SIZE)

		for y, line in enumerate(lines):
			for x, char in enumerate(line.strip()):
				if char in TILES:
					tile_data = TILES[char]
					if tile_data.get('is_player', False):
						self.player.rect.x = x * TILE_SIZE
						self.player.rect.y = y * TILE_SIZE + offset_y
					else:
						block = Platform(x * TILE_SIZE, y * TILE_SIZE + offset_y, tile_data)
						self.platform_list.add(block)

	def shift_world(self, shift_x: int, shift_y: int) -> None:
		self.world_shift += (shift_x, shift_y)
		for platform in self.platform_list:
			platform.shift(shift_x, shift_y)

	def restart(self) -> None:
		if self.map is None:
			return
		self.platform_list.empty()
		self.load_map(self.map)

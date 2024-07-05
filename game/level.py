import os

import pygame
from pygame import Surface
from pygame.sprite import Group

from game.platform import Platform
from game.settings import SCREEN_HEIGHT, SCREEN_WIDTH, SHIFT_THRESHOLD_X, TILE_SIZE, WHITE
from game.tiles import Tile, TILES


def _search_maps_folder(folder: str) -> str:
	"""
	Returns the absolute path to the maps folder.
	Search into the folder of the current script, then goes into parent folder until root folder.
	"""
	current_folder: str = os.path.dirname(os.path.abspath(__file__))
	while current_folder != '/':
		if folder in os.listdir(current_folder):
			return os.path.join(current_folder, folder)
		current_folder = os.path.dirname(current_folder)
	return ''


class Level:
	def __init__(self) -> None:
		self.camera = pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
		self.platform_list = Group()
		self.map: str | None = None
		self.height = 0
		self.width = 0
		self.tile_map: list[list[str]] = []
		self.spawn_point = (0, 0)

	@property
	def platforms(self) -> list[Platform]:
		return self.platform_list.sprites()

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
		self.width = len(lines[0].strip())
		self.height = len(lines)

		offset_y = SCREEN_HEIGHT - (len(lines) * TILE_SIZE)

		for y, line in enumerate(lines):
			row: list[Tile] = []
			for x, char in enumerate(line.strip()):
				if char in TILES:
					tile_data = TILES[char]
					row += [char]
					if tile_data.get('is_player', False):
						spawn_point_x = x * TILE_SIZE
						spawn_point_y = y * TILE_SIZE + offset_y
						self.spawn_point = (spawn_point_x, spawn_point_y)
					elif not tile_data.get('is_air', False):
						block = Platform(x * TILE_SIZE, y * TILE_SIZE + offset_y, tile_data)
						self.platform_list.add(block)

			self.tile_map += [row]

	def follow_player(self, player: 'Player'):
		"""
		Shift the world according to the player's position.
		:param player: The player object
		"""
		self.camera.centerx = player.rect.centerx
		self.camera.centery = TILE_SIZE * 12
		return

	def restart(self) -> None:
		if self.map is None:
			return
		self.platform_list.empty()
		self.load_map(self.map)

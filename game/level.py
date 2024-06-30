import os
from functools import cached_property
from time import sleep

from pygame import Surface
from pygame.sprite import Group

from game.platform import Platform
from game.settings import SCREEN_HEIGHT, TILE_SIZE, WHITE
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
	def __init__(self, player: 'game.Player') -> None:
		self.platform_list = Group()
		self.player = player
		self.world_shift = (0, 0)
		self.map: str | None = None
		self.width = 0
		self.height = 0
		self.tile_map: list[list[Tile]] = []
		self.finished = False
		player.level = self

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
					row += [tile_data]
					if tile_data.get('is_player', False):
						self.player.rect.x = x * TILE_SIZE
						self.player.rect.y = y * TILE_SIZE + offset_y
					else:
						block = Platform(x * TILE_SIZE, y * TILE_SIZE + offset_y, tile_data)
						self.platform_list.add(block)

			self.tile_map += [row]

	def shift_world(self, shift_x: int, shift_y: int) -> None:
		self.world_shift = (self.world_shift[0] + shift_x, self.world_shift[1] + shift_y)
		for platform in self.platforms:
			platform.shift(shift_x, shift_y)

	def restart(self) -> None:
		if self.map is None:
			return
		self.platform_list.empty()
		self.load_map(self.map)
		self.finished = False
		self.world_shift = (0, 0)
		self.player.revive()

from pathlib import Path
from typing import TYPE_CHECKING

import pygame
from pygame import Surface
from pygame.sprite import Group

from game.platform import Platform
from game.settings import SCREEN_HEIGHT, SCREEN_WIDTH, TILE_SIZE, WHITE
from game.tiles import TILES

if TYPE_CHECKING:
	from game.player import Player


def _search_maps_folder(folder: str | Path) -> Path:
	"""Returns the absolute path to the maps folder, walking up from this file until it is found."""
	current_folder = Path(__file__).resolve().parent
	while True:
		candidate = current_folder / folder
		if candidate.exists():
			return candidate
		parent = current_folder.parent
		if parent == current_folder:
			return Path()
		current_folder = parent


class Level:
	def __init__(self) -> None:
		self.camera = pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
		self.platform_list = Group()
		self.map: str | None = None
		self.height = 0
		self.width = 0
		self.tile_map: list[list[str]] = []
		self.spawn_point = (0, 0)
		self.checkpoints: list[tuple[int, int]] = []
		self.platform_columns: list[list[Platform]] = []

	@property
	def platforms(self) -> list[Platform]:
		sprites = self.platform_list.sprites()
		# Group.sprites() returns List[Sprite] but we know they are all Platforms
		return sprites  # type: ignore[return-value]

	def update(self) -> None:
		self.platform_list.update()

	def draw(self, screen: Surface) -> None:
		screen.fill(WHITE)
		self.platform_list.draw(screen)

	def load_map(self, map_path: str) -> None:
		path = Path(map_path)
		maps_path = _search_maps_folder(path.parent) / path.name
		with maps_path.open() as file:
			lines = file.readlines()

		self.map = map_path
		self.width = len(lines[0].strip())
		self.height = len(lines)
		self.checkpoints = []
		self.platform_columns = [[] for _ in range(self.width)]

		offset_y = SCREEN_HEIGHT - (len(lines) * TILE_SIZE)

		for y, line in enumerate(lines):
			row: list[str] = []
			for x, char in enumerate(line.strip()):
				if char in TILES:
					tile_data = TILES[char]
					row += [char]
					if tile_data.get('is_player', False):
						spawn_point_x = x * TILE_SIZE
						spawn_point_y = y * TILE_SIZE + offset_y
						self.spawn_point = (spawn_point_x, spawn_point_y)
					elif tile_data.get('is_checkpoint', False):
						checkpoint_x = x * TILE_SIZE
						checkpoint_y = y * TILE_SIZE + offset_y
						self.checkpoints.append((checkpoint_x, checkpoint_y))
					elif not tile_data.get('is_air', False):
						block = Platform(x * TILE_SIZE, y * TILE_SIZE + offset_y, tile_data)
						self.platform_list.add(block)
						self.platform_columns[x].append(block)

			self.tile_map += [row]

	def platforms_in_range(self, left: int, right: int) -> list[Platform]:
		"""Returns the platforms whose column overlaps the [left, right] pixel range."""
		first = max(0, left // TILE_SIZE)
		last = min(len(self.platform_columns) - 1, right // TILE_SIZE)
		return [platform for column in self.platform_columns[first:last + 1] for platform in column]

	def get_random_spawn_point(self, use_checkpoints: bool = False) -> tuple[int, int]:
		"""Returns a random checkpoint, or the default spawn point if checkpoints are off or there are none."""
		if use_checkpoints and self.checkpoints:
			import random
			return random.choice(self.checkpoints)
		return self.spawn_point

	def follow_player(self, player: 'Player'):
		"""Centers the camera on the player horizontally, keeping a fixed height."""
		self.camera.centerx = player.rect.centerx
		self.camera.centery = TILE_SIZE * 12

	def restart(self) -> None:
		if self.map is None:
			return
		self.platform_list.empty()
		self.load_map(self.map)

from collections.abc import Callable
from functools import lru_cache
from typing import Final

import pygame
from pygame import Rect, Surface
from pygame.font import Font

from game.settings import BLACK, PLAYER_COLOR, SCREEN_HEIGHT, SCREEN_WIDTH, SEMI_YELLOW, TILE_SIZE, WHITE
from game.tiles import TILES
from game.world import PLAYER_H, PLAYER_W, World

DEAD_COLOR: Final[tuple[int, int, int]] = (200, 200, 210)
HUD_COLOR: Final[tuple[int, int, int]] = BLACK


class Renderer:
	"""
	Camera-culled renderer for a batched World.

	The static level is baked into one big Surface at startup, so a frame is a single blit for the whole
	map plus one rect per visible player, instead of blitting thousands of platform sprites every frame.
	"""

	def __init__(self, world: World, caption: str = 'Neural-Jump') -> None:
		pygame.init()
		pygame.font.init()
		self.world = world
		self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), vsync=True)
		pygame.display.set_caption(caption)
		self.clock = pygame.time.Clock()
		self.camera = Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
		self.key_actions: dict[int, tuple[Callable[[], None], str]] = {}
		self.level_surface = self._bake_level(world)

	@staticmethod
	def _bake_level(world: World) -> Surface:
		surface = Surface((world.width * TILE_SIZE, world.height * TILE_SIZE + max(0, world.offset_y)))
		surface.fill(WHITE)
		for y in range(world.height):
			for x in range(world.width):
				color = TILES.get(str(world.chars[y, x]), {}).get('color')
				if color is not None:
					surface.fill(color, Rect(x * TILE_SIZE, y * TILE_SIZE + world.offset_y, TILE_SIZE, TILE_SIZE))
		for checkpoint_x, checkpoint_y in world.checkpoints:
			checkpoint = Surface((TILE_SIZE, TILE_SIZE))
			checkpoint.fill(SEMI_YELLOW)
			checkpoint.set_alpha(128)
			surface.blit(checkpoint, (checkpoint_x, checkpoint_y))
		return surface

	def add_key_action(self, key: int, action: Callable[[], None], description: str = '') -> None:
		self.key_actions[key] = (action, description)

	def poll_events(self) -> None:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				raise SystemExit
			if event.type == pygame.KEYDOWN and event.key in self.key_actions:
				self.key_actions[event.key][0]()

	@staticmethod
	@lru_cache(maxsize=8)
	def _font(size: int) -> Font:
		return pygame.font.SysFont('Arial', size)

	def draw(self, focus_index: int, hud_lines: list[str], tick_rate: int = 0) -> None:
		world = self.world
		self.camera.centerx = int(world.x[focus_index]) + PLAYER_W // 2
		self.camera.centery = TILE_SIZE * 12
		self.camera.left = max(0, min(self.camera.left, self.level_surface.get_width() - SCREEN_WIDTH))

		self.screen.fill(WHITE)
		self.screen.blit(self.level_surface, (0, 0), self.camera)

		left, top = self.camera.left, self.camera.top
		for index in range(world.count):
			if world.win[index]:
				continue
			x = int(world.x[index]) - left
			y = int(world.y[index]) - top
			if -PLAYER_W < x < SCREEN_WIDTH and -PLAYER_H < y < SCREEN_HEIGHT:
				color = DEAD_COLOR if world.dead[index] else PLAYER_COLOR
				self.screen.fill(color, Rect(x, y, PLAYER_W, PLAYER_H))

		y_offset = 10
		for line in hud_lines:
			self.screen.blit(self._font(24).render(line, True, HUD_COLOR), (10, y_offset))
			y_offset += 26

		y_offset = SCREEN_HEIGHT - 20 * (len(self.key_actions) + 1)
		for key, (_, description) in self.key_actions.items():
			if description:
				text = self._font(16).render(f'{pygame.key.name(key).upper()} - {description}', True, HUD_COLOR)
				text.set_alpha(128)
				self.screen.blit(text, (10, y_offset))
				y_offset += 20

		pygame.display.flip()
		if tick_rate:
			self.clock.tick(tick_rate)

	@staticmethod
	def quit() -> None:
		pygame.quit()

from functools import cache
from pprint import pprint
from typing import Protocol

import pygame
from pygame import Rect, Surface
from pygame.font import Font
from pygame.sprite import Group, Sprite
from pygame.time import Clock

from game.level import Level
from game.player import Player
from game.settings import SCREEN_HEIGHT, SCREEN_WIDTH, WHITE


class HasImageAndRect(Protocol):
	rect: Rect
	image: Surface


class Game:
	player: Player | None = None

	def __init__(self, players_count: int, tick_rate: int = 60, display_window: bool = True, has_playable_player: bool = True) -> None:
		self.players = list[Player]()
		self.level = Level()
		self.level.load_map('maps/level_1.txt')

		if has_playable_player:
			self.player = Player(0, 0)
			self.player.level = self.level
			self.player.revive()
			self.players += [self.player]

		for _ in range(players_count):
			player = Player(0, 0)
			player.level = self.level
			player.revive()
			self.players += [player]

		self.tick_rate = tick_rate
		self.active_sprite_list = Group()
		self.active_sprite_list.add(*self.players)
		self.has_playable_player = has_playable_player
		self.clock = Clock()
		self.screen: Surface | None = None
		self.running = False
		self.display_window = display_window
		self.additional_draws = list[tuple[Surface, tuple[int, int]]]()

	def handle_inputs(self) -> None:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()

			if self.player is None:
				continue
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_LEFT:
					self.player.go_left()
				if event.key == pygame.K_RIGHT:
					self.player.go_right()
				if event.key == pygame.K_UP:
					self.player.jump()
			if event.type == pygame.KEYUP:
				if event.key == pygame.K_LEFT and self.player.change_x < 0:
					self.player.stop()
				if event.key == pygame.K_RIGHT and self.player.change_x > 0:
					self.player.stop()

			# Restart the game when typing R
			if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
				self.level.restart()

	@cache
	def _get_font(self, font_size: int) -> Font:
		return pygame.font.SysFont('Arial', font_size)

	def draw_text(self, text: str, x: int, y: int, font_size: int = 20, color: tuple[int, int, int] = (0, 0, 0)) -> None:
		font = self._get_font(font_size)
		text_surface = font.render(text, True, color)
		self.additional_draws += [(text_surface, (x, y))]

	def init(self) -> None:
		if not self.display_window:
			import os

			os.environ["SDL_VIDEODRIVER"] = "dummy"

		pygame.init()
		pygame.font.init()
		self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), vsync=True)
		pygame.display.set_caption("Neural-Jump")

	def start(self) -> None:
		self.init()

		self.running = True

		while self.running:
			self.tick()

		self.quit()

	def update(self) -> None:
		self.active_sprite_list.update()
		self.level.update()
		self.clock.tick(self.tick_rate)

		if self.has_playable_player:
			alive_players = [player for player in self.players if not player.dead and not player.win]
			if not len(alive_players):
				return

			alive_players.sort(key=lambda player: player.rect.x, reverse=True)
			self.level.follow_player(alive_players[0])

	def draw_sprite(self, sprite: HasImageAndRect):
		self.screen.blit(sprite.image, (sprite.rect.x - self.level.camera.x, sprite.rect.y - self.level.camera.y))

	def draw(self) -> None:
		self.screen.fill(WHITE)
		for platform in self.level.platform_list:
			self.draw_sprite(platform)
		for active_sprite in self.active_sprite_list:
			self.draw_sprite(active_sprite)
		for surface, destination in self.additional_draws.copy():
			self.screen.blit(surface, destination)
		self.additional_draws = []
		pygame.display.flip()

	def tick(self) -> None:
		self.handle_inputs()
		self.update()

		if self.display_window:
			self.draw()

	def quit(self) -> None:
		pygame.quit()

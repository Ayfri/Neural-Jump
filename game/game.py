from functools import cache
from pprint import pprint

import pygame
from pygame import Surface
from pygame.font import Font
from pygame.sprite import Group
from pygame.time import Clock

from game.level import Level
from game.player import Player
from game.settings import SCREEN_HEIGHT, SCREEN_WIDTH


class Game:
	def __init__(self, tick_rate: int = 60, display_window: bool = True) -> None:
		self.player = Player(0, 0)
		self.level = Level(self.player)
		self.level.load_map('maps/level_1.txt')
		self.player.level = self.level
		self.player.revive()
		self.tick_rate = tick_rate
		self.active_sprite_list = Group()
		self.active_sprite_list.add(self.player)
		self.clock = Clock()
		self.screen: Surface | None = None
		self.running = False
		self.display_window = display_window
		self.additional_draws = list[tuple[Surface, tuple[int, int]]]()

	def handle_inputs(self, movement: bool = True) -> None:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()

			if not movement:
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

	def init_window(self) -> None:
		self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), vsync=True)
		pygame.display.set_caption("Mario-Like")

	def start(self) -> None:
		pygame.init()
		if self.display_window:
			self.init_window()

		self.running = True

		while self.running:
			self.tick()

		self.quit()

	def update(self) -> None:
		self.active_sprite_list.update()
		self.level.update()
		self.clock.tick(self.tick_rate)

	def draw(self) -> None:
		self.level.draw(self.screen)
		self.active_sprite_list.draw(self.screen)
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

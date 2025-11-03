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
from game.settings import SCREEN_HEIGHT, SCREEN_WIDTH, WHITE, SEMI_YELLOW


class HasImageAndRect(Protocol):
	rect: Rect
	image: Surface


class Game:
	player: Player | None = None
	followed_player: Player | None = None

	def __init__(self, players_count: int, tick_rate: int = 60, display_window: bool = True, has_playable_player: bool = True) -> None:
		self.key_actions = dict[int, tuple[callable, str]]()
		self.players = list[Player]()
		self.level = Level()
		self.level.load_map('maps/level_1.txt')
		self.use_checkpoints = False

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

	def add_key_action(self, key: int, action: callable, description: str = "") -> None:
		"""Add a custom key action to the game."""
		self.key_actions[key] = (action, description)

	def handle_inputs(self) -> None:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()

			if event.type == pygame.KEYDOWN:
				# Handle custom key actions first
				if event.key in self.key_actions:
					action, _ = self.key_actions[event.key]
					action()

				if self.player is None:
					continue

				if event.key == pygame.K_LEFT:
					self.player.go_left()
				if event.key == pygame.K_RIGHT:
					self.player.go_right()
				if event.key == pygame.K_UP:
					self.player.jump()
				if event.key == pygame.K_c:
					self.use_checkpoints = not self.use_checkpoints
					self.level.restart()
					spawn_point = self.level.get_random_spawn_point(self.use_checkpoints)
					self.player.rect.x = spawn_point[0]
					self.player.rect.y = spawn_point[1]

			if event.type == pygame.KEYUP:
				if event.key == pygame.K_LEFT and self.player.change_x < 0:
					self.player.stop()
				if event.key == pygame.K_RIGHT and self.player.change_x > 0:
					self.player.stop()

			# Restart the game when typing R
			if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
				self.level.restart()
				spawn_point = self.level.get_random_spawn_point(self.use_checkpoints)
				self.player.rect.x = spawn_point[0]
				self.player.rect.y = spawn_point[1]

	@cache
	def _get_font(self, font_size: int) -> Font:
		if not pygame.font.get_init():
			pygame.font.init()
		return pygame.font.SysFont('Arial', font_size)

	def draw_text(self, text: str, x: int, y: int, font_size: int = 20, color: tuple[int, int, int] = (0, 0, 0), alpha: int = 255) -> None:
		font = self._get_font(font_size)
		text_surface = font.render(text, True, color)
		if alpha != 255:
			text_surface.set_alpha(alpha)
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

	def update(self, tick: int = None) -> None:
		self.level.update()
		for player in self.players:
			player.update(tick)
		self.clock.tick(self.tick_rate)

		alive_players = [player for player in self.players if not player.dead and not player.win]
		if not len(alive_players):
			return

		alive_players.sort(key=lambda player: player.rect.x, reverse=True)

		# Update followed players
		if self.followed_player:
			self.followed_player.is_followed = False
		self.followed_player = alive_players[0]
		self.followed_player.is_followed = True
		self.level.follow_player(self.followed_player)

	def draw_sprite(self, sprite: HasImageAndRect):
		assert self.screen is not None
		self.screen.blit(sprite.image, (sprite.rect.x - self.level.camera.x, sprite.rect.y - self.level.camera.y))

	def draw_checkpoints(self) -> None:
		if not self.use_checkpoints:
			return

		assert self.screen is not None
		for checkpoint_x, checkpoint_y in self.level.checkpoints:
			checkpoint_surface = pygame.Surface((40, 40))
			checkpoint_surface.fill(SEMI_YELLOW)
			checkpoint_surface.set_alpha(128)
			self.screen.blit(checkpoint_surface, (checkpoint_x - self.level.camera.x, checkpoint_y - self.level.camera.y))

	def draw(self) -> None:
		assert self.screen is not None
		self.screen.fill(WHITE)
		for platform in self.level.platform_list:
			self.draw_sprite(platform)
		self.draw_checkpoints()
		for active_sprite in self.active_sprite_list:
			self.draw_sprite(active_sprite)

		# Draw keyboard shortcuts with reduced opacity
		y_offset = SCREEN_HEIGHT - 100
		self.draw_text("R - Restart", 10, y_offset, font_size=16, alpha=128)
		y_offset += 20
		self.draw_text("C - Toggle Checkpoints", 10, y_offset, font_size=16, alpha=128)
		y_offset += 20
		self.draw_text("Arrow Keys - Move", 10, y_offset, font_size=16, alpha=128)
		y_offset += 20

		# Draw custom key actions
		for key, (_, description) in self.key_actions.items():
			if description:
				key_name = pygame.key.name(key).upper()
				self.draw_text(f"{key_name} - {description}", 10, y_offset, font_size=16, alpha=128)
				y_offset += 20

		if self.use_checkpoints:
			self.draw_text("Checkpoints Mode: ON", 10, y_offset, font_size=16, alpha=128)
		else:
			self.draw_text("Checkpoints Mode: OFF", 10, y_offset, font_size=16, alpha=128)

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

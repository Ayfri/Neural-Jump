# main.py

import pygame
from game.player import Player
from game.platform import Platform

from game.level import Level
from game.settings import SCREEN_WIDTH, SCREEN_HEIGHT


def start_game() -> None:
	pygame.init()
	screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
	pygame.display.set_caption("Mario-Like")

	player = Player(0, 0)
	level = Level(player)
	level.load_map('maps/level_1.txt')

	active_sprite_list = pygame.sprite.Group()
	player.level = level
	active_sprite_list.add(player)

	clock = pygame.time.Clock()
	done = False

	while not done:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				done = True
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_LEFT:
					player.go_left()
				if event.key == pygame.K_RIGHT:
					player.go_right()
				if event.key == pygame.K_UP:
					player.jump()
			if event.type == pygame.KEYUP:
				if event.key == pygame.K_LEFT and player.change_x < 0:
					player.stop()
				if event.key == pygame.K_RIGHT and player.change_x > 0:
					player.stop()

			# Restart the game when typing R
			if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
				level.restart()

		active_sprite_list.update()
		level.update()

		level.draw(screen)
		active_sprite_list.draw(screen)

		clock.tick(60)
		pygame.display.flip()

	pygame.quit()


if __name__ == "__main__":
	start_game()

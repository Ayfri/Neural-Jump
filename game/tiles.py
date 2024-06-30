from typing import NotRequired, TypedDict

import pygame

from game.settings import AQUA, BLACK, GREEN, RED, SEMI_RED, WHITE


class Tile(TypedDict, total=False):
	color: tuple[int, int, int]
	is_solid: bool
	is_player: bool
	reward: int


TILES: dict[str, Tile] = {
	'#': {
		'color': RED,
		'is_solid': True,
	},
	'*': {
		'color': SEMI_RED,
	},
	'_': {
		'color': BLACK,
		'is_solid': True,
	},
	'P': {
		'is_player': True,
	},
	'F': {
		'color': AQUA,
		'reward': 100,
	},
	'R': {
		'color': GREEN,
		'reward': 200,
	},
}

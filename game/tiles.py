from typing import TypedDict

from game.settings import AQUA, BLACK, GREEN, RED, SEMI_RED, WHITE


class Tile(TypedDict, total=False):
	color: tuple[int, int, int]
	is_air: bool
	is_solid: bool
	is_player: bool
	is_checkpoint: bool
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
	'.': {
		'is_air': True,
	},
	'P': {
		'is_player': True,
	},
	'@': {
		'is_checkpoint': True,
		'is_air': True,
	},
	'F': {
		'color': AQUA,
		'reward': 500,
	},
	'R': {
		'color': GREEN,
		'reward': 800,
	},
}

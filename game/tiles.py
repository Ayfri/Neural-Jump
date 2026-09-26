from enum import IntEnum
from typing import Final


class TileKind(IntEnum):
	"""
	What a map character is.

	One value per tile instead of a bag of `is_*` flags: the map is then a plain int grid, so every property
	the world needs is a comparison over that grid rather than a dict lookup per cell.
	"""
	AIR = 0
	SOLID = 1
	DECOR = 2  # Drawn, never collided with
	SPAWN = 3
	CHECKPOINT = 4
	COIN = 5
	FLAG = 6
	ENEMY = 7  # Where an enemy starts walking from, the cell itself is air


TILE_CHARS: Final[dict[str, TileKind]] = {
	'.': TileKind.AIR,
	'#': TileKind.SOLID,
	'*': TileKind.DECOR,
	'P': TileKind.SPAWN,
	'@': TileKind.CHECKPOINT,
	'o': TileKind.COIN,
	'F': TileKind.FLAG,
	'E': TileKind.ENEMY,
}

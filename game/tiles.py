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


TILE_CHARS: Final[dict[str, TileKind]] = {
	'.': TileKind.AIR,
	'#': TileKind.SOLID,
	'*': TileKind.DECOR,
	'P': TileKind.SPAWN,
	'@': TileKind.CHECKPOINT,
	'o': TileKind.COIN,
	'F': TileKind.FLAG,
}

# Fitness every paying tile is worth, so what a level hands out is read in one place. The flag is worth far
# more than its number says: reaching it ends the episode, and the training loop pays that out on its own
TILE_REWARDS: Final[dict[TileKind, float]] = {
	TileKind.COIN: 5.0,  # Paid per coin, so a full sweep is worth less than reaching the flag
	TileKind.FLAG: 1.0,
}

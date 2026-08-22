"""
Every sprite the level and its players are drawn with, painted in code.

Nothing here loads a file: a sprite is a small character grid, one character per art pixel, blown up to its
final size with nearest-neighbour scaling so the pixels stay square and hard-edged. The whole set is tiny and
cached, so a frame only ever blits surfaces that were painted once.
"""
import random
from collections.abc import Mapping, Sequence
from functools import lru_cache
from math import sin, tau
from typing import Final

import numpy as np
import pygame
from numpy.typing import NDArray
from pygame import Rect, Surface

from game.settings import TILE_SIZE
from game.tiles import TileKind

type Color = tuple[int, int, int]
type Paint = tuple[int, int, int] | tuple[int, int, int, int]

TILE_ART: Final[int] = 8  # Art pixels along a tile edge, so one art pixel is TILE_SIZE // 8 screen pixels
PIXEL: Final[int] = max(1, TILE_SIZE // TILE_ART)
VARIANTS: Final[int] = 8  # Terrain speckle layouts, dealt out per tile so a wall of blocks is never flat paint

# Terrain, a cool slate so the warm fitness ramp on the players stays the brightest thing on screen
ROCK_DEEP: Final[Color] = (28, 32, 50)
ROCK: Final[Color] = (52, 58, 84)
ROCK_LIT: Final[Color] = (70, 78, 110)
ROCK_EDGE: Final[Color] = (116, 130, 172)
ROCK_CREST: Final[Color] = (174, 190, 232)

DECOR_FILL: Final[Color] = (38, 34, 56)
DECOR_LINE: Final[Color] = (54, 48, 76)

SKY_TOP: Final[Color] = (12, 14, 32)
SKY_MID: Final[Color] = (32, 38, 74)
SKY_LOW: Final[Color] = (76, 68, 112)
HILL_FAR: Final[Color] = (44, 48, 88)
HILL_NEAR: Final[Color] = (26, 28, 56)
STAR: Final[Color] = (214, 224, 255)
STAR_DIM: Final[Color] = (120, 132, 176)

COIN_COLOR: Final[Color] = (255, 196, 32)
COIN_DARK: Final[Color] = (168, 106, 16)
COIN_SHINE: Final[Color] = (255, 246, 190)

FLAG_BODY: Final[Color] = (46, 210, 214)
FLAG_DARK: Final[Color] = (18, 104, 128)
FLAG_SHINE: Final[Color] = (198, 252, 252)

CHECKPOINT_COLOR: Final[Color] = (150, 60, 230)
CHECKPOINT_GLOW: Final[Color] = (206, 150, 255)
CHECKPOINT_HAZE: Final[tuple[int, int, int, int]] = (*CHECKPOINT_COLOR, 96)

EYE_WHITE: Final[Color] = (238, 244, 255)
EYE_DARK: Final[Color] = (22, 22, 34)
EYE_CLOSED: Final[Color] = (96, 98, 112)

UP: Final[int] = 1
DOWN: Final[int] = 2
LEFT: Final[int] = 4
RIGHT: Final[int] = 8

# One rounded blob with a face, painted looking forward: the flipped copy is what a player walking left is drawn with
BODY_ART: Final[tuple[str, ...]] = (
	'....DDDDDDDD....',
	'..DDBBBBBBBBDD..',
	'.DBBBBBBBBBBBBD.',
	'.DBBBBBBBBBBBBD.',
	'DBBBBBBBBBBBBBBD',
	'DBBBWWBBBBWWBBBD',
	'DBBBWKBBBBWKBBBD',
	'DBBBWWBBBBWWBBBD',
	'DBBBBBBBBBBBBBBD',
	'DBBBBBBBBBBBBBBD',
	'DBBBBBMMMMBBBBBD',
	'DBBBBBBBBBBBBBBD',
	'DBBBBBBBBBBBBBBD',
	'DBBBBBBBBBBBBBBD',
	'DBBBBBBBBBBBBBBD',
	'.DBBBBBBBBBBBBD.',
	'.DBBBBBBBBBBBBD.',
	'..DDBBBBBBBBDD..',
	'....DDDDDDDD....',
)
# The same blob with both eyes pushed to one side, which is all a heading needs to read at this size
_SIDE_EYES: Final[dict[int, str]] = {
	5: 'DBBBBBWWBBWWBBBD',
	6: 'DBBBBBWKBBWKBBBD',
	7: 'DBBBBBWWBBWWBBBD',
}
BODY_ART_SIDE: Final[tuple[str, ...]] = tuple(_SIDE_EYES.get(index, row) for index, row in enumerate(BODY_ART))

JUMP_ART: Final[tuple[str, ...]] = (
	'..AA..',
	'.AAAA.',
	'AAAAAA',
	'.AAAA.',
	'..AA..',
)

COIN_ART: Final[tuple[str, ...]] = (
	'..DDDD..',
	'.DSSCCD.',
	'DSSCCCCD',
	'DSCCCCCD',
	'DSCCCCCD',
	'DCCCCCCD',
	'.DCCCCD.',
	'..DDDD..',
)

CHECKPOINT_ART: Final[tuple[str, ...]] = (
	'.PFFFFFF',
	'.PFFFFF.',
	'.PFFFF..',
	'.PFFF...',
	'.P......',
	'.P......',
	'.P......',
	'PPPPP...',
)


def shade(color: Color, factor: float) -> Color:
	return (min(255, int(color[0] * factor)), min(255, int(color[1] * factor)), min(255, int(color[2] * factor)))


def _paint(rows: Sequence[str], palette: Mapping[str, Paint], size: tuple[int, int]) -> Surface:
	"""A character grid blown up to `size`: one character is one art pixel, a dot is transparent."""
	art = Surface((len(rows[0]), len(rows)), pygame.SRCALPHA)
	for y, row in enumerate(rows):
		for x, char in enumerate(row):
			paint = palette.get(char)
			if paint is not None:
				art.set_at((x, y), paint)
	return pygame.transform.scale(art, size).convert_alpha()


def _speckles() -> tuple[tuple[tuple[int, int], ...], ...]:
	"""Fixed speckle layouts, drawn once here so a sprite never depends on the order it was first asked for."""
	generator = random.Random(0x5EED)
	return tuple(
		tuple((generator.randrange(1, TILE_ART - 1), generator.randrange(2, TILE_ART - 1)) for _ in range(5))
		for _ in range(VARIANTS)
	)


_SPECKLES: Final[tuple[tuple[tuple[int, int], ...], ...]] = _speckles()


@lru_cache(maxsize=VARIANTS * 16)
def terrain_sprite(mask: int, variant: int) -> Surface:
	"""
	A block of terrain, rimmed on the sides that touch air.

	`mask` is the `UP`/`DOWN`/`LEFT`/`RIGHT` bits of the solid neighbours, so a run of blocks reads as one body
	with a bright crest on top and a dark edge where it ends, out of the sixteen shapes that can occur.
	"""
	art = Surface((TILE_ART, TILE_ART))
	art.fill(ROCK)
	for index, spot in enumerate(_SPECKLES[variant]):
		art.set_at(spot, ROCK_LIT if index % 2 else ROCK_DEEP)
	if not mask & LEFT:
		art.fill(ROCK_DEEP, Rect(0, 0, 1, TILE_ART))
	if not mask & RIGHT:
		art.fill(ROCK_DEEP, Rect(TILE_ART - 1, 0, 1, TILE_ART))
	if not mask & DOWN:
		art.fill(ROCK_DEEP, Rect(0, TILE_ART - 1, TILE_ART, 1))
	if not mask & UP:
		art.fill(ROCK_EDGE, Rect(0, 0, TILE_ART, 2))
		art.fill(ROCK_CREST, Rect(0, 0, TILE_ART, 1))
	return pygame.transform.scale(art, (TILE_SIZE, TILE_SIZE)).convert()


@lru_cache(maxsize=2)
def decor_sprite() -> Surface:
	"""Bricks for the tiles that are only ever drawn, dark enough to read as the wall behind the level."""
	art = Surface((TILE_ART, TILE_ART))
	art.fill(DECOR_FILL)
	art.fill(DECOR_LINE, Rect(0, 0, TILE_ART, 1))
	art.fill(DECOR_LINE, Rect(0, TILE_ART // 2, TILE_ART, 1))
	art.fill(DECOR_LINE, Rect(TILE_ART // 4, 1, 1, TILE_ART // 2 - 1))
	art.fill(DECOR_LINE, Rect(3 * TILE_ART // 4, TILE_ART // 2, 1, TILE_ART // 2))
	return pygame.transform.scale(art, (TILE_SIZE, TILE_SIZE)).convert()


@lru_cache(maxsize=4)
def flag_sprite(top: bool) -> Surface:
	"""One cell of the goal beam, striped so the whole column reads as a single lit gate."""
	art = Surface((TILE_ART, TILE_ART))
	for y in range(TILE_ART):
		for x in range(TILE_ART):
			art.set_at((x, y), FLAG_SHINE if (x + y) % 4 == 0 else FLAG_BODY)
	art.fill(FLAG_DARK, Rect(0, 0, 1, TILE_ART))
	art.fill(FLAG_DARK, Rect(TILE_ART - 1, 0, 1, TILE_ART))
	if top:
		art.fill(FLAG_DARK, Rect(0, 0, TILE_ART, 1))
		art.fill(FLAG_SHINE, Rect(0, 1, TILE_ART, 1))
	return pygame.transform.scale(art, (TILE_SIZE, TILE_SIZE)).convert()


@lru_cache(maxsize=2)
def coin_sprite() -> Surface:
	"""A coin, drawn narrower than its tile so a trail of them reads as a line of dots rather than a wall."""
	inner = max(TILE_ART, round(TILE_SIZE * 0.65 / TILE_ART) * TILE_ART)
	surface = Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
	surface.blit(_paint(COIN_ART, {'D': COIN_DARK, 'C': COIN_COLOR, 'S': COIN_SHINE}, (inner, inner)), ((TILE_SIZE - inner) // 2,) * 2)
	return surface.convert_alpha()


@lru_cache(maxsize=2)
def checkpoint_sprite() -> Surface:
	"""A checkpoint: a violet haze filling the cell with a banner planted in it, so it reads over sky and rock."""
	surface = Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
	surface.fill(CHECKPOINT_HAZE)
	surface.blit(_paint(CHECKPOINT_ART, {'P': CHECKPOINT_GLOW, 'F': CHECKPOINT_COLOR}, (TILE_SIZE, TILE_SIZE)), (0, 0))
	return surface.convert_alpha()


@lru_cache(maxsize=2)
def jump_sprite(color: Color) -> Surface:
	return _paint(JUMP_ART, {'A': color}, (len(JUMP_ART[0]) * 2, len(JUMP_ART) * 2))


@lru_cache(maxsize=8)
def ring_sprite(color: Color, size: tuple[int, int], thickness: int) -> Surface:
	surface = Surface(size, pygame.SRCALPHA)
	pygame.draw.rect(surface, color, Rect(0, 0, *size), width=thickness)
	return surface.convert_alpha()


@lru_cache(maxsize=128)
def body_sprite(fill: Color, size: tuple[int, int], direction: int, dead: bool) -> Surface:
	"""
	One player: `fill` is its body, darkened for the outline and the mouth.

	A heading only moves the eyes, and the left-facing sprite is the right-facing one flipped, so the whole
	population comes out of a handful of cached surfaces whatever it is coloured by.
	"""
	palette: dict[str, Paint] = {
		'D': shade(fill, 0.38),
		'B': fill,
		'M': shade(fill, 0.5),
		'W': EYE_CLOSED if dead else EYE_WHITE,
		'K': EYE_CLOSED if dead else EYE_DARK,
	}
	sprite = _paint(BODY_ART if direction == 0 else BODY_ART_SIDE, palette, size)
	return pygame.transform.flip(sprite, True, False) if direction < 0 else sprite


def _sky_color(ratio: float) -> Color:
	"""Night at the top, dusk at the ground: two blends of three colours, met halfway down the map."""
	start, stop, blend = (SKY_TOP, SKY_MID, ratio * 2) if ratio < 0.5 else (SKY_MID, SKY_LOW, ratio * 2 - 1)
	return (
		int(start[0] + (stop[0] - start[0]) * blend),
		int(start[1] + (stop[1] - start[1]) * blend),
		int(start[2] + (stop[2] - start[2]) * blend),
	)


def _bake_sky(width: int, height: int) -> Surface:
	"""The gradient, its stars and two ranges of hills, painted into the level itself so the sky costs no blit."""
	surface = Surface((width, height))
	for y in range(0, height, PIXEL):
		surface.fill(_sky_color(y / max(1, height - 1)), Rect(0, y, width, PIXEL))

	generator = random.Random(0xC0FFEE)
	for _ in range(width * height // 26000):
		spot = Rect(generator.randrange(width), generator.randrange(int(height * 0.55)), PIXEL, PIXEL)
		surface.fill(STAR if generator.random() < 0.35 else STAR_DIM, spot)

	# The ranges sit near the lowest rows of the map, so they only ever show through the gaps in the terrain
	base = height - 2.5 * TILE_SIZE
	for color, amplitude, period, phase in ((HILL_FAR, 5.0, 41.0, 0.0), (HILL_NEAR, 3.0, 23.0, 1.7)):
		for x in range(0, width, PIXEL):
			ridge = int(base - amplitude * TILE_SIZE * (0.5 + 0.5 * sin(x / (period * TILE_SIZE) * tau + phase)))
			surface.fill(color, Rect(x, ridge, PIXEL, height - ridge))
	return surface.convert()


def bake_level(kinds: NDArray[np.uint8], size: tuple[int, int], top: int) -> Surface:
	"""
	The whole static level on one opaque surface: sky, terrain, decoration, checkpoints and the goal.

	Baking it once means a frame draws the level with a single blit whatever the map size, and the surface is
	converted to the display format so that blit is a straight copy. Coins are left out: which of them are
	still on the map depends on the agent being followed.
	"""
	surface = _bake_sky(*size)
	solid = (kinds == TileKind.SOLID).astype(np.uint8)
	masks = np.zeros(kinds.shape, dtype=np.uint8)
	masks[1:] |= solid[:-1] * UP
	masks[:-1] |= solid[1:] * DOWN
	masks[:, 1:] |= solid[:, :-1] * LEFT
	masks[:, :-1] |= solid[:, 1:] * RIGHT

	# One layout per tile from a fixed seed: scattered enough that no pattern shows, identical on every run
	variants = np.random.default_rng(0xA11CE).integers(0, VARIANTS, kinds.shape, dtype=np.uint8)
	painted = np.argwhere((kinds != TileKind.AIR) & (kinds != TileKind.COIN) & (kinds != TileKind.SPAWN))
	blits: list[tuple[Surface, tuple[int, int]]] = []
	for row, column in painted.tolist():
		kind = TileKind(int(kinds[row, column]))
		if kind is TileKind.SOLID:
			sprite = terrain_sprite(int(masks[row, column]), int(variants[row, column]))
		elif kind is TileKind.DECOR:
			sprite = decor_sprite()
		elif kind is TileKind.FLAG:
			sprite = flag_sprite(row == 0 or kinds[row - 1, column] != TileKind.FLAG)
		else:
			sprite = checkpoint_sprite()
		blits.append((sprite, (column * TILE_SIZE, top + row * TILE_SIZE)))

	surface.blits(blits, doreturn=False)
	return surface

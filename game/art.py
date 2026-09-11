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

from game.settings import ENEMY_HEIGHT, ENEMY_WIDTH, TILE_SIZE
from game.tiles import TileKind

type Color = tuple[int, int, int]
type Paint = tuple[int, int, int] | tuple[int, int, int, int]

TILE_ART: Final[int] = 8  # Art pixels along a tile edge, so one art pixel is TILE_SIZE // 8 screen pixels
PIXEL: Final[int] = max(1, TILE_SIZE // TILE_ART)
VARIANTS: Final[int] = 8  # Terrain speckle layouts, dealt out per tile so a wall of blocks is never flat paint

# Terrain, a light slate: the backdrop behind it is kept dark on purpose so a platform reads as the solid
# thing in the frame, and the outline around it is near black so its silhouette survives over any sky colour
ROCK_SHADOW: Final[Color] = (32, 36, 58)
ROCK_DEEP: Final[Color] = (74, 82, 112)
ROCK: Final[Color] = (104, 114, 150)
ROCK_LIT: Final[Color] = (132, 144, 184)
ROCK_EDGE: Final[Color] = (186, 200, 238)
ROCK_CREST: Final[Color] = (236, 244, 255)

DECOR_FILL: Final[Color] = (34, 31, 50)
DECOR_LINE: Final[Color] = (48, 43, 68)

SKY_TOP: Final[Color] = (7, 8, 20)
SKY_MID: Final[Color] = (19, 22, 48)
SKY_LOW: Final[Color] = (46, 40, 76)
# Three ranges, lightest at the back: the further one reads, the more sky haze sits between it and the level
HILL_FAR: Final[Color] = (30, 33, 64)
HILL_MID: Final[Color] = (21, 23, 48)
HILL_NEAR: Final[Color] = (13, 14, 32)
STAR: Final[Color] = (214, 224, 255)
STAR_DIM: Final[Color] = (110, 122, 166)

# Fraction of the camera each backdrop layer scrolls by: the further a layer reads, the slower it slides
SKY_PARALLAX: Final[float] = 0.08
STAR_PARALLAX: Final[float] = 0.05
HILL_FAR_PARALLAX: Final[float] = 0.11
HILL_MID_PARALLAX: Final[float] = 0.21
HILL_NEAR_PARALLAX: Final[float] = 0.36
COLORKEY: Final[Color] = (255, 0, 255)  # Backdrop layers punch out with a colour key so their empty runs blit for free

COIN_COLOR: Final[Color] = (255, 196, 32)
COIN_DARK: Final[Color] = (168, 106, 16)
COIN_SHINE: Final[Color] = (255, 246, 190)

FLAG_BODY: Final[Color] = (46, 210, 214)
FLAG_DARK: Final[Color] = (18, 104, 128)
FLAG_SHINE: Final[Color] = (198, 252, 252)

CHECKPOINT_COLOR: Final[Color] = (150, 60, 230)
CHECKPOINT_GLOW: Final[Color] = (206, 150, 255)
CHECKPOINT_HAZE: Final[tuple[int, int, int, int]] = (*CHECKPOINT_COLOR, 96)

ENEMY_COLOR: Final[Color] = (214, 72, 52)
ENEMY_DARK: Final[Color] = (92, 26, 24)
ENEMY_SKIN: Final[Color] = (244, 206, 156)

EYE_WHITE: Final[Color] = (238, 244, 255)
EYE_DARK: Final[Color] = (22, 22, 34)
EYE_CLOSED: Final[Color] = (96, 98, 112)

UP: Final[int] = 1
DOWN: Final[int] = 2
LEFT: Final[int] = 4
RIGHT: Final[int] = 8

# A rounded blob lit from above: the top rows are tinted, the bottom ones shaded, so it reads as a volume
# rather than a flat rectangle. Standing still it is symmetric, with both pupils centred in their whites
BODY_ART: Final[tuple[str, ...]] = (
	'.....DDDDDD.....',
	'...DDLLLLLLDD...',
	'..DLLLLLLLLLLD..',
	'.DLLLLLLLLLLLLD.',
	'.DLLBBBBBBBBLLD.',
	'DBBBBBBBBBBBBBBD',
	'DBBWWWBBBBWWWBBD',
	'DBBWKWBBBBWKWBBD',
	'DBBWWWBBBBWWWBBD',
	'DBBBBBBBBBBBBBBD',
	'DBBBBBMMMMBBBBBD',
	'DBBBBBBBBBBBBBBD',
	'DBBBBBBBBBBBBBBD',
	'DSBBBBBBBBBBBBSD',
	'DSSBBBBBBBBBBSSD',
	'.DSSSBBBBBBSSSD.',
	'..DSSSSSSSSSSD..',
	'...DDSSSSSSDD...',
	'.....DDDDDD.....',
)
# The same blob heading right. Three cues stack up instead of the eyes alone: the whole face slides forward,
# both pupils sit against the front of their whites, and the light moves to the leading edge
BODY_ART_SIDE: Final[tuple[str, ...]] = (
	'.....DDDDDD.....',
	'...DDBBBBLLDD...',
	'..DBBBBBBBBLLD..',
	'.DBBBBBBBBBBLLD.',
	'.DBBBBBBBBBBBLD.',
	'DSBBBBBBBBBBBBLD',
	'DSBBBBWWWBWWWBLD',
	'DSBBBBWKKBWKKBLD',
	'DSBBBBWWWBWWWBLD',
	'DSBBBBBBBBBBBBLD',
	'DSBBBBBBBMMMMBLD',
	'DSBBBBBBBBBBBBLD',
	'DSBBBBBBBBBBBBLD',
	'DSSBBBBBBBBBBBLD',
	'DSSSBBBBBBBBBBLD',
	'.DSSSBBBBBBBBLD.',
	'..DSSSSSSSSSSD..',
	'...DDSSSSSSDD...',
	'.....DDDDDD.....',
)

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

# A walking mushroom facing left, which is the way it starts out: both pupils sit against the front of their whites
ENEMY_ART: Final[tuple[str, ...]] = (
	'....DDDD....',
	'..DDBBBBDD..',
	'.DBBBBBBBBD.',
	'DBWWBBWWBBBD',
	'DBKWBBKWBBBD',
	'DBKWBBKWBBBD',
	'DBBBBBBBBBBD',
	'.DBBMMMMBBD.',
	'..DDDDDDDD..',
	'...SSSSSS...',
	'.DDDD..DDDD.',
	'.DDD....DDD.',
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


def tint(color: Color, amount: float) -> Color:
	"""Blends towards white rather than scaling, so a light already close to saturation keeps its hue."""
	return (
		int(color[0] + (255 - color[0]) * amount),
		int(color[1] + (255 - color[1]) * amount),
		int(color[2] + (255 - color[2]) * amount),
	)


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
		art.fill(ROCK_SHADOW, Rect(0, TILE_ART - 1, TILE_ART, 1))
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


@lru_cache(maxsize=4)
def enemy_sprite(heading: int) -> Surface:
	"""An enemy looking the way it walks: the art faces left and is flipped for the other way."""
	palette: dict[str, Paint] = {'D': ENEMY_DARK, 'B': ENEMY_COLOR, 'W': EYE_WHITE, 'K': EYE_DARK, 'M': ENEMY_DARK, 'S': ENEMY_SKIN}
	sprite = _paint(ENEMY_ART, palette, (int(ENEMY_WIDTH), int(ENEMY_HEIGHT)))
	return pygame.transform.flip(sprite, True, False) if heading > 0 else sprite


@lru_cache(maxsize=2)
def jump_sprite(color: Color) -> Surface:
	return _paint(JUMP_ART, {'A': color}, (len(JUMP_ART[0]) * 2, len(JUMP_ART) * 2))


@lru_cache(maxsize=8)
def ring_sprite(color: Color, size: tuple[int, int], thickness: int) -> Surface:
	surface = Surface(size, pygame.SRCALPHA)
	pygame.draw.rect(surface, color, Rect(0, 0, *size), width=thickness)
	return surface.convert_alpha()


@lru_cache(maxsize=512)
def body_sprite(fill: Color, size: tuple[int, int], direction: int, dead: bool, alpha: int = 255) -> Surface:
	"""
	One player: `fill` is its body, darkened for the outline and the mouth, lit for the heading cues.

	Standing still shows a symmetric face, and a heading slides the face forward and moves the light to the
	leading edge; the left-facing sprite is the right-facing one flipped, so the whole population comes out of a handful of
	cached surfaces whatever it is coloured by. `alpha` fades the whole sprite, which is what a dead agent
	in a crowd is drawn with.
	"""
	palette: dict[str, Paint] = {
		'D': shade(fill, 0.36),
		'S': shade(fill, 0.74),
		'B': fill,
		'L': tint(fill, 0.3),
		'M': shade(fill, 0.5),
		'W': EYE_CLOSED if dead else EYE_WHITE,
		'K': EYE_CLOSED if dead else EYE_DARK,
	}
	sprite = _paint(BODY_ART if direction == 0 else BODY_ART_SIDE, palette, size)
	if direction < 0:
		sprite = pygame.transform.flip(sprite, True, False)
	sprite.set_alpha(alpha)
	return sprite


def _sky_color(ratio: float) -> Color:
	"""Night at the top, dusk at the ground: two blends of three colours, met halfway down the layer."""
	start, stop, blend = (SKY_TOP, SKY_MID, ratio * 2) if ratio < 0.5 else (SKY_MID, SKY_LOW, ratio * 2 - 1)
	return (
		int(start[0] + (stop[0] - start[0]) * blend),
		int(start[1] + (stop[1] - start[1]) * blend),
		int(start[2] + (stop[2] - start[2]) * blend),
	)


def _bake_sky(width: int, height: int) -> Surface:
	"""The gradient on its own, opaque, so the layer behind everything else costs one straight copy."""
	surface = Surface((width, height))
	for y in range(0, height, PIXEL):
		surface.fill(_sky_color(y / max(1, height - 1)), Rect(0, y, width, PIXEL))
	return surface.convert()


def _punched(surface: Surface) -> Surface:
	"""A layer whose key colour is cut out, run-length encoded so its empty rows are skipped at blit time."""
	surface = surface.convert()
	surface.set_colorkey(COLORKEY, pygame.RLEACCEL)
	return surface


def _bake_stars(width: int, height: int) -> Surface:
	"""A star field that wraps horizontally: nothing is drawn in the last column, so the seam never cuts a star."""
	surface = Surface((width, height))
	surface.fill(COLORKEY)
	generator = random.Random(0xC0FFEE)
	for _ in range(width * height // 5200):
		spot = Rect(generator.randrange(width - PIXEL), generator.randrange(height), PIXEL, PIXEL)
		surface.fill(STAR if generator.random() < 0.35 else STAR_DIM, spot)
	return _punched(surface)


def _ridge_profile(width: int, waves: int, seed: int) -> list[float]:
	"""
	A ridge line in 0..1, one value per art pixel across the layer.

	Six harmonics with a 1/f falloff and random phases, so the long swells dominate and the silhouette never
	reads as the single sine it would be with one term. Every harmonic fits a whole number of waves in the
	width, which is what lets the layer be tiled end to end without a seam.
	"""
	generator = random.Random(seed)
	octaves = ((1, 1.0), (2, 0.62), (3, 0.30), (5, 0.16), (8, 0.08), (13, 0.04))
	phases = [generator.random() * tau for _ in octaves]
	raw = [
		sum(weight * sin(x / width * tau * count * waves + phase) for (count, weight), phase in zip(octaves, phases))
		for x in range(0, width, PIXEL)
	]
	low, span = min(raw), (max(raw) - min(raw)) or 1.0
	# The power broadens the valleys and lifts the summits, gently enough that the faces stay walkable slopes
	return [((value - low) / span) ** 1.15 for value in raw]


def _bake_ridge(width: int, height: int, color: Color, waves: int, amplitude: float, ground: float, seed: int) -> tuple[Surface, int]:
	"""
	One mountain range, filled from its ridge line down to the bottom of the layer.

	The shading is banded by depth under the ridge rather than by slope: a lit crest fading into the base
	colour and then into shadow, every band parallel to the silhouette. Because a band's thickness is the
	same in every column, nothing in the fill can flip between neighbours, which is what would show up as
	vertical striping. A sparse dither on top keeps it from reading as flat paint.

	Returns the layer and its floor, the row below which it is opaque in every column, which is where the
	range in front of it stops needing to be drawn at all.
	"""
	profile = _ridge_profile(width, waves, seed)
	surface = Surface((width, height))
	surface.fill(COLORKEY)
	base = height * ground
	span = amplitude * height
	tops = [int(base - span * value) // PIXEL * PIXEL for value in profile]
	# Depth in art pixels under the ridge, and the colour from there down: each band overwrites the one above
	bands = (
		(0, tint(color, 0.30)), (1, tint(color, 0.13)), (3, tint(color, 0.05)),
		(6, color), (15, shade(color, 0.92)), (28, shade(color, 0.84)),
	)
	speck = shade(color, 0.88)
	generator = random.Random(seed ^ 0xBEEF)

	for index, x in enumerate(range(0, width, PIXEL)):
		top = tops[index]
		for depth, band in bands:
			y = top + depth * PIXEL
			if y < height:
				surface.fill(band, Rect(x, y, PIXEL, height - y))
		for y in range(top + 4 * PIXEL, height, PIXEL):
			if generator.random() < 0.045:
				surface.fill(speck, Rect(x, y, PIXEL, PIXEL))
	return _punched(surface), max(tops)


class Background:
	"""
	The parallax backdrop the level is drawn over.

	Four layers, each scrolled at its own fraction of the camera: the gradient barely moves, the stars drift,
	and the two ridges slide fast enough to read as distance. Every layer is one viewport-sized surface baked
	once, the scrolling ones tile horizontally and the key-coloured ones skip their empty rows, so a frame
	costs at most nine blits whatever the map size. Each layer is also clipped to the rows the one in front of
	it leaves uncovered, which on a low camera cuts the gradient and the stars out of the frame entirely.
	"""

	def __init__(self, level: tuple[int, int], view: tuple[int, int]) -> None:
		self._width, self._height = view
		self._scroll_y = max(1, level[1] - self._height)
		self._sky = _bake_sky(self._width, self._height + int(self._scroll_y * SKY_PARALLAX))
		self._stars = _bake_stars(self._width, self._height + int(self._scroll_y * STAR_PARALLAX))
		# Back to front, so a range can be clipped to the rows the one in front of it does not already cover
		self._ridges = tuple(
			(*_bake_ridge(self._width, self._height, color, waves, amplitude, ground, seed), factor)
			for color, waves, amplitude, ground, seed, factor in (
				(HILL_FAR, 1, 0.22, 0.82, 0x1234, HILL_FAR_PARALLAX),
				(HILL_MID, 1, 0.28, 0.90, 0x5678, HILL_MID_PARALLAX),
				(HILL_NEAR, 1, 0.34, 0.99, 0x9ABC, HILL_NEAR_PARALLAX),
			)
		)

	def draw(self, screen: Surface, left: int, top: int) -> None:
		"""`left` is the camera's x in the level and `top` its y on the level surface, so both are already positive."""
		# The ridges sit at the bottom of their layer, so they slide down the screen as the camera climbs
		tops = [int((self._scroll_y - top) * factor) for _, _, factor in self._ridges]
		# The back range is opaque under its own floor, so the two layers behind it are cut off there: with the
		# camera low in the level that is most of the viewport, and the gradient is the one full-screen copy a frame makes
		sky_height = min(self._height, max(0, tops[0] + self._ridges[0][1]))

		if sky_height:
			window = Rect(0, int(top * SKY_PARALLAX), self._width, sky_height)
			screen.blit(self._sky, (0, 0), window)

			# A layer is drawn twice, one copy left of the other, so whatever the offset the viewport is covered
			window.top = int(top * STAR_PARALLAX)
			offset = -int(left * STAR_PARALLAX) % self._width
			screen.blit(self._stars, (offset - self._width, 0), window)
			screen.blit(self._stars, (offset, 0), window)

		for index, (ridge, floor, factor) in enumerate(self._ridges):
			y = tops[index]
			if y >= self._height:
				continue
			# Everything under the next range's floor is hidden by it, so this one is cut off there
			bottom = min(self._height, tops[index + 1] + self._ridges[index + 1][1]) if index + 1 < len(tops) else self._height
			area = Rect(0, 0, self._width, max(0, bottom - y))
			if not area.height:
				continue
			offset = -int(left * factor) % self._width
			screen.blit(ridge, (offset - self._width, y), area)
			screen.blit(ridge, (offset, y), area)


def bake_level(kinds: NDArray[np.uint8], size: tuple[int, int], top: int) -> Surface:
	"""
	The whole static level on one surface: terrain, decoration and the goal, over a punched-out backdrop.

	Baking it once means a frame draws the level with a single blit whatever the map size, and because the
	air is a run-length encoded key colour rather than real transparency, that blit skips the empty sky
	instead of blending it. Coins and checkpoints are drawn per frame: the coins left on the map depend on
	the agent being followed, and a checkpoint is a translucent haze the key colour cannot carry.
	"""
	surface = Surface(size)
	surface.fill(COLORKEY)
	solid = (kinds == TileKind.SOLID).astype(np.uint8)
	masks = np.zeros(kinds.shape, dtype=np.uint8)
	masks[1:] |= solid[:-1] * UP
	masks[:-1] |= solid[1:] * DOWN
	masks[:, 1:] |= solid[:, :-1] * LEFT
	masks[:, :-1] |= solid[:, 1:] * RIGHT

	# One layout per tile from a fixed seed: scattered enough that no pattern shows, identical on every run
	variants = np.random.default_rng(0xA11CE).integers(0, VARIANTS, kinds.shape, dtype=np.uint8)
	skipped = (TileKind.AIR, TileKind.COIN, TileKind.SPAWN, TileKind.CHECKPOINT, TileKind.ENEMY)
	painted = np.argwhere(~np.isin(kinds, [int(kind) for kind in skipped]))
	blits: list[tuple[Surface, tuple[int, int]]] = []
	for row, column in painted.tolist():
		kind = TileKind(int(kinds[row, column]))
		if kind is TileKind.SOLID:
			sprite = terrain_sprite(int(masks[row, column]), int(variants[row, column]))
		elif kind is TileKind.DECOR:
			sprite = decor_sprite()
		else:
			sprite = flag_sprite(row == 0 or kinds[row - 1, column] != TileKind.FLAG)
		blits.append((sprite, (column * TILE_SIZE, top + row * TILE_SIZE)))

	surface.fblits(blits)
	return _punched(surface)

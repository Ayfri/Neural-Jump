from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Final

import numpy as np
import pygame
from numpy.typing import NDArray
from pygame import Rect, Surface
from pygame.font import Font

from game.settings import SCREEN_HEIGHT, SCREEN_WIDTH, SEMI_YELLOW, TILE_SIZE, WHITE
from game.tiles import TILES
from game.world import PLAYER_H, PLAYER_W, World

type Color = tuple[int, int, int]
type Blit = tuple[Surface, tuple[int, int]]

# Fitness ramp, walked from worst to best rank in the living population
FITNESS_RAMP: Final[tuple[Color, ...]] = ((198, 40, 62), (226, 118, 38), (226, 196, 46), (128, 200, 60), (36, 190, 168))
FITNESS_BUCKETS: Final[int] = 12
DEAD_COLOR: Final[Color] = (176, 178, 188)
WON_COLOR: Final[Color] = (120, 90, 220)
ELITE_RING: Final[Color] = (240, 190, 60)
RANDOM_RING: Final[Color] = (168, 92, 232)
FOCUS_RING: Final[Color] = (24, 24, 30)
JUMP_MARKER: Final[Color] = (32, 130, 240)

PANEL_BACKGROUND: Final[tuple[int, int, int, int]] = (54, 59, 74, 247)
PANEL_BORDER: Final[Color] = (108, 116, 140)
TITLE_COLOR: Final[Color] = (150, 220, 255)
LABEL_COLOR: Final[Color] = (182, 189, 206)
VALUE_COLOR: Final[Color] = (245, 247, 252)
ACCENT_COLOR: Final[Color] = (120, 232, 176)
GRID_COLOR: Final[Color] = (92, 100, 122)

MARGIN: Final[int] = 12
PADDING: Final[int] = 10
ROW_HEIGHT: Final[int] = 19
GAUGE_HEIGHT: Final[int] = 26  # A row with a bar under its value
TITLE_SIZE: Final[int] = 14
BODY_SIZE: Final[int] = 15
SMALL_SIZE: Final[int] = 13
LABEL_COLUMN: Final[int] = 116
HISTOGRAM_BINS: Final[int] = 18
HISTORY_LENGTH: Final[int] = 80
SPARKLINE_GUTTER: Final[int] = 34  # Room kept on the right of the curve for its scale labels
DEFAULT_FPS: Final[int] = 60


@dataclass(slots=True)
class Gauge:
	"""A row rendered as a label, a value and a filled bar showing `ratio`."""
	label: str
	value: str
	ratio: float
	color: Color = ACCENT_COLOR


@dataclass(slots=True)
class Panel:
	title: str
	rows: list[tuple[str, str] | Gauge] = field(default_factory=list)


@dataclass(slots=True)
class Hud:
	"""Everything the renderer cannot read off the World itself."""
	generation: int = 1
	tick: int = 0
	tick_rate: int = 60
	checkpoint: tuple[int, int] = (1, 1)
	best_ever: float = 0.0
	elite_count: int = 0
	random_count: int = 0
	speed: float = 0.0  # Simulation ticks per real second
	sim_speed: float = 1.0  # How many in-game seconds pass per real second
	training: list[tuple[str, str]] = field(default_factory=list)
	history: Sequence[float] = ()


def _shade(color: Color, factor: float) -> Color:
	return (min(255, int(color[0] * factor)), min(255, int(color[1] * factor)), min(255, int(color[2] * factor)))


@lru_cache(maxsize=FITNESS_BUCKETS)
def _bucket_color(bucket: int) -> Color:
	position = bucket / max(1, FITNESS_BUCKETS - 1) * (len(FITNESS_RAMP) - 1)
	low = min(int(position), len(FITNESS_RAMP) - 2)
	blend = position - low
	start, stop = FITNESS_RAMP[low], FITNESS_RAMP[low + 1]
	return (
		int(start[0] + (stop[0] - start[0]) * blend),
		int(start[1] + (stop[1] - start[1]) * blend),
		int(start[2] + (stop[2] - start[2]) * blend),
	)


@lru_cache(maxsize=8)
def _font(size: int) -> Font:
	return pygame.font.SysFont('consolas,dejavusansmono,couriernew,monospace', size)


@lru_cache(maxsize=2048)
def _text(size: int, text: str, color: Color) -> Surface:
	return _font(size).render(text, True, color)


@lru_cache(maxsize=16)
def _panel_background(width: int, height: int) -> Surface:
	surface = Surface((width, height), pygame.SRCALPHA)
	pygame.draw.rect(surface, PANEL_BACKGROUND, Rect(0, 0, width, height), border_radius=6)
	pygame.draw.rect(surface, PANEL_BORDER, Rect(0, 0, width, height), width=1, border_radius=6)
	return surface


@lru_cache(maxsize=FITNESS_BUCKETS * 9)  # Every (bucket, heading, state) sprite, so a full population never evicts one
def _body_sprite(bucket: int, direction: int, state: int) -> Surface:
	"""
	One player sprite: fill encodes the fitness rank, the stroke its state, the chevron its heading.

	`state` is 0 alive, 1 dead, 2 won. The whole set is tiny and fully cached, so a frame only ever blits
	pre-rendered surfaces instead of drawing shapes per player.
	"""
	fill = (DEAD_COLOR, WON_COLOR)[state - 1] if state else _bucket_color(bucket)
	surface = Surface((PLAYER_W, PLAYER_H), pygame.SRCALPHA)
	body = Rect(0, 0, PLAYER_W, PLAYER_H)
	pygame.draw.rect(surface, fill, body, border_radius=3)
	pygame.draw.rect(surface, _shade(fill, 0.45), body, width=2, border_radius=3)

	center_x, center_y = PLAYER_W // 2, PLAYER_H // 2
	mark = _shade(fill, 0.3) if state == 0 else _shade(fill, 0.55)
	if direction > 0:
		pygame.draw.polygon(surface, mark, [(center_x - 3, center_y - 6), (center_x + 5, center_y), (center_x - 3, center_y + 6)])
	elif direction < 0:
		pygame.draw.polygon(surface, mark, [(center_x + 3, center_y - 6), (center_x - 5, center_y), (center_x + 3, center_y + 6)])
	else:
		pygame.draw.rect(surface, mark, Rect(center_x - 3, center_y - 3, 6, 6), border_radius=1)
	return surface


@lru_cache(maxsize=8)
def _ring_sprite(color: Color, gap: int, thickness: int) -> Surface:
	size = (PLAYER_W + 2 * (gap + thickness), PLAYER_H + 2 * (gap + thickness))
	surface = Surface(size, pygame.SRCALPHA)
	pygame.draw.rect(surface, color, Rect(0, 0, *size), width=thickness, border_radius=5)
	return surface


@lru_cache(maxsize=2)
def _jump_sprite() -> Surface:
	surface = Surface((12, 9), pygame.SRCALPHA)
	pygame.draw.polygon(surface, JUMP_MARKER, [(6, 0), (11, 8), (0, 8)])
	return surface


class Renderer:
	"""
	Camera-culled renderer for a batched World.

	The static level is baked into one big Surface at startup and every player sprite is pre-rendered per
	(fitness bucket, heading, state), so a frame is one blit for the map plus a couple of batched `blits`
	calls for the population, whatever its size.
	"""

	def __init__(self, world: World, target_fps: int = 0, caption: str = 'Neural-Jump') -> None:
		pygame.init()
		pygame.font.init()
		self.world = world
		# Vsync only helps while the target matches the display; above it, it would cap the framerate itself
		desktop = self.desktop_fps()
		self.target_fps = target_fps or desktop
		self.vsync = self.target_fps >= desktop
		self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), vsync=int(self.vsync))
		if self.vsync:
			self.target_fps = desktop
		pygame.display.set_caption(caption)
		self.clock = pygame.time.Clock()
		self.camera = Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
		self.key_actions: dict[int, tuple[Callable[[], None], str]] = {}
		self.level_surface = self._bake_level(world)
		self._ranks = np.zeros(world.count, dtype=np.int64)

	@staticmethod
	def desktop_fps() -> int:
		"""The refresh rate of the display the window sits on, falling back to 60 when SDL cannot tell."""
		try:
			rates = [rate for rate in pygame.display.get_desktop_refresh_rates() if rate > 0]
		except (AttributeError, pygame.error):
			return DEFAULT_FPS
		return max(rates) if rates else DEFAULT_FPS

	def measured_fps(self) -> float:
		return self.clock.get_fps()

	@staticmethod
	def _bake_level(world: World) -> Surface:
		surface = Surface((world.width * TILE_SIZE, world.height * TILE_SIZE + max(0, world.offset_y)))
		surface.fill(WHITE)
		for y in range(world.height):
			for x in range(world.width):
				color = TILES.get(str(world.chars[y, x]), {}).get('color')
				if color is not None:
					surface.fill(color, Rect(x * TILE_SIZE, y * TILE_SIZE + world.offset_y, TILE_SIZE, TILE_SIZE))
		for checkpoint_x, checkpoint_y in world.checkpoints:
			checkpoint = Surface((TILE_SIZE, TILE_SIZE))
			checkpoint.fill(SEMI_YELLOW)
			checkpoint.set_alpha(128)
			surface.blit(checkpoint, (checkpoint_x, checkpoint_y))
		return surface

	def add_key_action(self, key: int, action: Callable[[], None], description: str = '') -> None:
		self.key_actions[key] = (action, description)

	def poll_events(self) -> None:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				raise SystemExit
			if event.type == pygame.KEYDOWN and event.key in self.key_actions:
				self.key_actions[event.key][0]()

	def draw(self, focus_index: int, fitness: NDArray[np.float64], hud: Hud) -> None:
		self._move_camera(focus_index)
		self.screen.blit(self.level_surface, (0, 0), self.camera)
		self._draw_players(focus_index, fitness, hud)
		self._draw_hud(focus_index, fitness, hud)
		pygame.display.flip()
		# Under vsync the flip already paces the loop; capping on top of it would make us miss every other frame
		self.clock.tick(0 if self.vsync else self.target_fps)

	def _move_camera(self, focus_index: int) -> None:
		self.camera.centerx = int(self.world.x[focus_index]) + PLAYER_W // 2
		self.camera.centery = int(self.world.y[focus_index]) + PLAYER_H // 2
		self.camera.left = max(0, min(self.camera.left, self.level_surface.get_width() - SCREEN_WIDTH))
		self.camera.top = max(0, min(self.camera.top, self.level_surface.get_height() - SCREEN_HEIGHT))

	def _fitness_buckets(self, fitness: NDArray[np.float64]) -> NDArray[np.int64]:
		"""Buckets agents by their rank rather than their raw fitness, so the colors stay readable."""
		count = len(fitness)
		if count < 2:
			return np.zeros(count, dtype=np.int64)
		self._ranks[np.argsort(fitness, kind='stable')] = np.arange(count)
		return self._ranks * (FITNESS_BUCKETS - 1) // (count - 1)

	def _draw_players(self, focus_index: int, fitness: NDArray[np.float64], hud: Hud) -> None:
		world = self.world
		left, top = self.camera.left, self.camera.top
		screen_x = world.x.astype(np.int64) - left
		screen_y = world.y.astype(np.int64) - top
		visible = (screen_x > -PLAYER_W) & (screen_x < SCREEN_WIDTH) & (screen_y > -PLAYER_H) & (screen_y < SCREEN_HEIGHT)
		if not visible.any():
			return

		# Read as python lists: pulling 300 values out of an array one index at a time costs more than the loop
		buckets = self._fitness_buckets(fitness).tolist()
		states = (world.dead.astype(np.int64) + world.win.astype(np.int64) * 2).tolist()
		directions = np.sign(world.change_x).astype(np.int64).tolist()
		rising = (world.change_y < 0).tolist()
		screen_x = screen_x.tolist()
		screen_y = screen_y.tolist()

		bodies: list[Blit] = []
		markers: list[Blit] = []
		jump = _jump_sprite()
		elite_ring = _ring_sprite(ELITE_RING, 2, 2)
		random_ring = _ring_sprite(RANDOM_RING, 2, 2)
		focus_ring = _ring_sprite(FOCUS_RING, 5, 2)
		random_start = world.count - hud.random_count

		for index in np.flatnonzero(visible).tolist():
			x, y = screen_x[index], screen_y[index]
			state = states[index]
			bodies.append((_body_sprite(buckets[index], directions[index], state), (x, y)))
			if state == 0 and rising[index]:
				markers.append((jump, (x + PLAYER_W // 2 - 6, y - 11)))
			if index < hud.elite_count:
				markers.append((elite_ring, (x - 4, y - 4)))
			elif index >= random_start:
				markers.append((random_ring, (x - 4, y - 4)))
			if index == focus_index:
				markers.append((focus_ring, (x - 7, y - 7)))

		self.screen.blits(bodies, doreturn=False)
		if markers:
			self.screen.blits(markers, doreturn=False)

	def _draw_hud(self, focus_index: int, fitness: NDArray[np.float64], hud: Hud) -> None:
		world = self.world
		alive = int(world.alive().sum())
		checkpoint, checkpoints = hud.checkpoint
		fps = self.measured_fps()

		run = Panel('Run', [
			('Generation', f'{hud.generation}'),
			('Time', f'{hud.tick / max(1, hud.tick_rate):.1f}s' + (f'  ckpt {checkpoint}/{checkpoints}' if checkpoints > 1 else '')),
			Gauge('Alive', f'{alive}/{world.count}', alive / max(1, world.count)),
			('Best', f'{float(fitness.max()) if fitness.size else 0.0:.1f}'),
			('Record', f'{hud.best_ever:.1f}'),
			('Ticks/s', f'{hud.speed:,.0f}  x{hud.sim_speed:.0f}'),
			Gauge('FPS', f'{fps:.0f}/{self.target_fps}', fps / max(1, self.target_fps)),
		])
		focus = Panel('Focus', [
			('Agent', f'#{focus_index}' + (' elite' if focus_index < hud.elite_count else '')),
			('Fitness', f'{float(fitness[focus_index]):.1f}'),
			('Position', f'{int(world.x[focus_index])}, {int(world.y[focus_index])}'),
		])

		width, right_width = 236, 236
		self._draw_panel(run, MARGIN, MARGIN, width)
		self._draw_panel(focus, MARGIN, MARGIN + self._panel_height(run) + MARGIN, width)
		self._draw_panel(Panel('Training', list(hud.training)), SCREEN_WIDTH - MARGIN - right_width, MARGIN, right_width)
		self._draw_fitness_panel(fitness, hud)
		self._draw_legend()

	@staticmethod
	def _panel_height(panel: Panel) -> int:
		rows = sum(GAUGE_HEIGHT if isinstance(row, Gauge) else ROW_HEIGHT for row in panel.rows)
		return PADDING * 2 + ROW_HEIGHT + 4 + rows

	def _draw_panel(self, panel: Panel, x: int, y: int, width: int) -> None:
		height = self._panel_height(panel)
		self.screen.blit(_panel_background(width, height), (x, y))
		self.screen.blit(_text(TITLE_SIZE, panel.title, TITLE_COLOR), (x + PADDING, y + PADDING))

		row_y = y + PADDING + ROW_HEIGHT + 4
		for row in panel.rows:
			label, value = (row.label, row.value) if isinstance(row, Gauge) else row
			self.screen.blit(_text(SMALL_SIZE, label, LABEL_COLOR), (x + PADDING, row_y + 2))
			self.screen.blit(_text(BODY_SIZE, value, VALUE_COLOR), (x + LABEL_COLUMN, row_y))
			if not isinstance(row, Gauge):
				row_y += ROW_HEIGHT
				continue

			bar = Rect(x + LABEL_COLUMN, row_y + ROW_HEIGHT + 1, width - LABEL_COLUMN - PADDING, 3)
			pygame.draw.rect(self.screen, GRID_COLOR, bar)
			bar.width = int(bar.width * min(1.0, max(0.0, row.ratio)))
			if bar.width:
				pygame.draw.rect(self.screen, row.color, bar)
			row_y += GAUGE_HEIGHT

	def _draw_fitness_panel(self, fitness: NDArray[np.float64], hud: Hud) -> None:
		width, height = 300, 196
		x = SCREEN_WIDTH - MARGIN - width
		y = SCREEN_HEIGHT - MARGIN - height
		self.screen.blit(_panel_background(width, height), (x, y))
		self.screen.blit(_text(TITLE_SIZE, 'Fitness', TITLE_COLOR), (x + PADDING, y + PADDING))
		self.screen.blit(_text(SMALL_SIZE, 'Population Spread', LABEL_COLOR), (x + PADDING, y + PADDING + ROW_HEIGHT))

		plot = Rect(x + PADDING, y + PADDING + 2 * ROW_HEIGHT + 2, width - 2 * PADDING, 56)
		self._draw_histogram(plot, fitness)

		self.screen.blit(_text(SMALL_SIZE, 'Best Per Generation', LABEL_COLOR), (x + PADDING, plot.bottom + ROW_HEIGHT))
		# The right gutter holds the scale labels, so they never sit on top of the curve
		sparkline = Rect(x + PADDING, plot.bottom + 2 * ROW_HEIGHT + 6, width - 2 * PADDING - SPARKLINE_GUTTER, 38)
		self._draw_sparkline(sparkline, hud.history)

	def _draw_histogram(self, plot: Rect, fitness: NDArray[np.float64]) -> None:
		"""Population fitness distribution, each bar tinted with the ramp used on the players themselves."""
		pygame.draw.line(self.screen, GRID_COLOR, (plot.left, plot.bottom), (plot.right, plot.bottom))
		if fitness.size == 0:
			return

		counts, edges = np.histogram(fitness, bins=HISTOGRAM_BINS)
		peak = max(1, int(counts.max()))
		bar_width = plot.width / HISTOGRAM_BINS
		for index, count in enumerate(counts):
			bar_height = int(count / peak * (plot.height - 2))
			if not bar_height:
				continue
			color = _bucket_color(index * (FITNESS_BUCKETS - 1) // (HISTOGRAM_BINS - 1))
			bar = Rect(int(plot.left + index * bar_width), plot.bottom - bar_height, max(1, int(bar_width) - 1), bar_height)
			pygame.draw.rect(self.screen, color, bar)

		self.screen.blit(_text(SMALL_SIZE, f'{edges[0]:.0f}', LABEL_COLOR), (plot.left, plot.bottom + 2))
		high = _text(SMALL_SIZE, f'{edges[-1]:.0f}', LABEL_COLOR)
		self.screen.blit(high, (plot.right - high.get_width(), plot.bottom + 2))

	def _draw_sparkline(self, plot: Rect, history: Sequence[float]) -> None:
		pygame.draw.line(self.screen, GRID_COLOR, (plot.left, plot.bottom), (plot.right, plot.bottom))
		recent = list(history)[-HISTORY_LENGTH:]
		if len(recent) < 2:
			return

		low, high = min(recent), max(recent)
		span = high - low or 1.0
		step = plot.width / (len(recent) - 1)
		points = [(plot.left + index * step, plot.bottom - (value - low) / span * plot.height) for index, value in enumerate(recent)]
		pygame.draw.aalines(self.screen, ACCENT_COLOR, False, points)
		pygame.draw.circle(self.screen, VALUE_COLOR, points[-1], 2)
		self.screen.blit(_text(SMALL_SIZE, f'{high:.0f}', VALUE_COLOR), (plot.right + 5, plot.top - 3))
		self.screen.blit(_text(SMALL_SIZE, f'{low:.0f}', LABEL_COLOR), (plot.right + 5, plot.bottom - 13))

	def _draw_legend(self) -> None:
		"""Bottom-left: what the shapes and colors on the players mean, plus the key bindings."""
		entries = [(ELITE_RING, 'Elite'), (RANDOM_RING, 'Random'), (FOCUS_RING, 'Focus'), (JUMP_MARKER, 'Rising'), (DEAD_COLOR, 'Dead'), (WON_COLOR, 'Won')]
		swatch_rows = (len(entries) + 2) // 3
		hints = sum(1 for _, description in self.key_actions.values() if description)
		width = 250
		height = PADDING * 2 + ROW_HEIGHT * (2 + swatch_rows + hints) + 12
		x, y = MARGIN, SCREEN_HEIGHT - MARGIN - height
		self.screen.blit(_panel_background(width, height), (x, y))
		self.screen.blit(_text(TITLE_SIZE, 'Legend', TITLE_COLOR), (x + PADDING, y + PADDING))

		ramp = Rect(x + PADDING, y + PADDING + ROW_HEIGHT + 4, width - 2 * PADDING, 6)
		slice_width = ramp.width / FITNESS_BUCKETS
		for bucket in range(FITNESS_BUCKETS):
			piece = Rect(int(ramp.left + bucket * slice_width), ramp.top, int(slice_width) + 1, ramp.height)
			pygame.draw.rect(self.screen, _bucket_color(bucket), piece)
		self.screen.blit(_text(SMALL_SIZE, 'Fitness Low', LABEL_COLOR), (x + PADDING, ramp.bottom + 2))
		high = _text(SMALL_SIZE, 'High', LABEL_COLOR)
		self.screen.blit(high, (ramp.right - high.get_width(), ramp.bottom + 2))

		row_y = ramp.bottom + ROW_HEIGHT + 2
		for index, (color, label) in enumerate(entries):
			column = x + PADDING + (index % 3) * 78
			if index and index % 3 == 0:
				row_y += ROW_HEIGHT
			pygame.draw.rect(self.screen, color, Rect(column, row_y + 3, 8, 8), border_radius=2)
			self.screen.blit(_text(SMALL_SIZE, label, LABEL_COLOR), (column + 13, row_y))

		row_y += ROW_HEIGHT
		for key, (_, description) in self.key_actions.items():
			if description:
				self.screen.blit(_text(SMALL_SIZE, f'{pygame.key.name(key).upper():<4} {description}', LABEL_COLOR), (x + PADDING, row_y))
				row_y += ROW_HEIGHT

	@staticmethod
	def quit() -> None:
		pygame.quit()

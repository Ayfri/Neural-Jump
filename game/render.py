from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Final

import numpy as np
import pygame
from numpy.typing import NDArray
from pygame import Rect, Surface
from pygame._sdl2.video import Window
from pygame.font import Font

from game.art import (
	CHECKPOINT_COLOR, COIN_COLOR, COLORKEY, ENEMY_COLOR, Background, bake_level, body_sprite, checkpoint_sprite,
	coin_sprite, enemy_sprite, jump_sprite, ring_sprite,
)
from game.settings import SCREEN_HEIGHT, SCREEN_WIDTH, TILE_SIZE
from game.world import ENEMY_H, ENEMY_W, PLAYER_H, PLAYER_W, World

type Color = tuple[int, int, int]
type Blit = tuple[Surface, tuple[int, int]]
type Row = tuple[str, str] | Gauge  # A plain label/value line, or one with a bar under it

# Fitness ramp, walked from worst to best rank in the living population
FITNESS_RAMP: Final[tuple[Color, ...]] = ((198, 40, 62), (226, 118, 38), (226, 196, 46), (128, 200, 60), (36, 190, 168))
FITNESS_BUCKETS: Final[int] = 12
DEAD_COLOR: Final[Color] = (176, 178, 188)
WON_COLOR: Final[Color] = (58, 120, 246)
HUMAN_COLOR: Final[Color] = (120, 232, 176)  # A human run has no rank to color by, so it gets its own fill
ELITE_RING: Final[Color] = (240, 190, 60)
RANDOM_RING: Final[Color] = (168, 92, 232)
FOCUS_RING: Final[Color] = (248, 250, 255)  # Bright, because the ring sits on the dark sky as often as on terrain
JUMP_MARKER: Final[Color] = (32, 130, 240)
DEAD_ALPHA: Final[int] = 80  # A dead agent stays visible as a ghost instead of crowding the living ones

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
LEGEND_COLUMNS: Final[int] = 3
HINT_COLUMNS: Final[int] = 2  # Key bindings are laid out side by side, one column would own the panel
BANNER_SIZE: Final[int] = 30
PANEL_WIDTH: Final[int] = 236
LEGEND_WIDTH: Final[int] = 250
HISTOGRAM_BINS: Final[int] = 18
HISTORY_LENGTH: Final[int] = 80
SPARKLINE_GUTTER: Final[int] = 34  # Room kept on the right of the curve for its scale labels
MENU_WIDTH: Final[int] = 420
MENU_ROWS: Final[int] = 16  # Items shown at once, the list scrolls to keep the selected one in the middle
DEFAULT_FPS: Final[int] = 60

# Zoom range: under x1 the world is drawn on a surface larger than the screen, and only the band the level
# actually spans is drawn on and scaled, which is what keeps the far end of the range affordable
MIN_ZOOM: Final[float] = 0.2
MAX_ZOOM: Final[float] = 4.0
ZOOM_STEP: Final[float] = 1.25


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
	rows: list[Row] = field(default_factory=list)
	width: int = PANEL_WIDTH


@dataclass(slots=True)
class Legend:
	"""Bottom-left panel: what the colors on the level mean, then the key bindings."""
	entries: Sequence[tuple[Color, str]] = ()
	hints: Sequence[tuple[str, str]] = ()
	ramp: bool = False  # Whether the fitness gradient is worth explaining, which only a population run is


@dataclass(slots=True)
class Fitness:
	"""Bottom-right panel: how the population is spread right now, next to its best score per generation."""
	values: NDArray[np.float64]
	history: Sequence[float] = ()


@dataclass(slots=True)
class Menu:
	"""A centred list with one row selected, drawn over everything else, panels hidden or not."""
	title: str
	items: Sequence[str]
	selected: int = 0


@dataclass(slots=True)
class Hud:
	"""
	Everything drawn over the level.

	The caller fills the panels because it is the only side that knows what its numbers mean; the renderer
	only lays them out, so a training run and a human run share one overlay.
	"""
	left: list[Panel] = field(default_factory=list)  # Stacked down from the top left corner
	right: list[Panel] = field(default_factory=list)  # Stacked down from the top right corner
	legend: Legend | None = None
	fitness: Fitness | None = None
	banner: str = ''  # Centred message, drawn even with the panels hidden
	menu: Menu | None = None
	elite_count: int = 0
	random_count: int = 0
	solo: bool = False  # One human body instead of a ranked population


TRAINING_LEGEND: Final[tuple[tuple[Color, str], ...]] = (
	(ELITE_RING, 'Elite'), (RANDOM_RING, 'Random'), (FOCUS_RING, 'Focus'),
	(JUMP_MARKER, 'Rising'), (DEAD_COLOR, 'Dead'), (WON_COLOR, 'Won'),
	(COIN_COLOR, 'Coin'), (CHECKPOINT_COLOR, 'Ckpt'), (ENEMY_COLOR, 'Enemy'),
)
PLAY_LEGEND: Final[tuple[tuple[Color, str], ...]] = (
	(HUMAN_COLOR, 'You'), (JUMP_MARKER, 'Rising'), (DEAD_COLOR, 'Dead'),
	(WON_COLOR, 'Won'), (COIN_COLOR, 'Coin'), (CHECKPOINT_COLOR, 'Ckpt'), (ENEMY_COLOR, 'Enemy'),
)


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
	return _font(size).render(text, True, color).convert_alpha()


@lru_cache(maxsize=32)
def _panel_background(width: int, height: int) -> Surface:
	surface = Surface((width, height), pygame.SRCALPHA)
	pygame.draw.rect(surface, PANEL_BACKGROUND, Rect(0, 0, width, height), border_radius=6)
	pygame.draw.rect(surface, PANEL_BORDER, Rect(0, 0, width, height), width=1, border_radius=6)
	return surface.convert_alpha()


@lru_cache(maxsize=FITNESS_BUCKETS * 24)  # Every (bucket, heading, state) sprite, so a full population never evicts one
def _player_sprite(bucket: int, direction: int, state: int) -> Surface:
	"""
	One opaque player sprite: the fill encodes its fitness rank, or its state once it is dead, won or human.

	`state` is 0 alive, 1 dead, 2 won, 3 the human player. The whole set is tiny and fully cached, so a frame
	only ever blits pre-rendered surfaces instead of painting a body per player.
	"""
	fill = (DEAD_COLOR, WON_COLOR, HUMAN_COLOR)[state - 1] if state else _bucket_color(bucket)
	return body_sprite(fill, (PLAYER_W, PLAYER_H), direction, state == 1)


@lru_cache(maxsize=64)  # One per heading and zoom, which is a handful
def _ghost_sprite(direction: int, size: tuple[int, int]) -> Surface:
	"""A dead agent, faded out. Sized by the zoom, because it is drawn on the screen rather than in the band."""
	return body_sprite(DEAD_COLOR, size, direction, True, DEAD_ALPHA)


def _player_ring(color: Color, gap: int, thickness: int) -> Surface:
	"""An outline around a player, `gap` pixels off its body, drawn from the same top-left corner minus the pad."""
	pad = gap + thickness
	return ring_sprite(color, (PLAYER_W + 2 * pad, PLAYER_H + 2 * pad), thickness)


class Renderer:
	"""
	Camera-culled renderer for a batched World.

	The static level is baked into one big Surface at startup and every player sprite is pre-rendered per
	(fitness bucket, heading, state), so a frame is one blit for the map plus a couple of batched `fblits`
	calls for the population, whatever its size.
	"""

	def __init__(self, world: World, target_fps: int = 0, caption: str = 'Neural-Jump') -> None:
		pygame.init()
		pygame.font.init()
		# Vsync only helps while the target matches the display; above it, it would cap the framerate itself
		desktop = self.desktop_fps()
		self.target_fps = target_fps or desktop
		self.vsync = self.target_fps >= desktop
		# SCALED stretches the fixed 1600x900 surface every layout is built against onto whatever the OS window becomes
		self.screen = pygame.display.set_mode(
			(SCREEN_WIDTH, SCREEN_HEIGHT), flags=pygame.RESIZABLE | pygame.SCALED, vsync=int(self.vsync),
		)
		if self.vsync:
			self.target_fps = desktop
		pygame.display.set_caption(caption)
		Window.from_display_module().maximize()
		self.clock = pygame.time.Clock()
		self.camera = Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
		self.key_actions: dict[int, tuple[Callable[[], None], str]] = {}
		self.show_hud = True
		self.camera_lock = False  # Camera only ever advances toward the furthest x reached, never snaps back
		self.surface = self.screen  # What the world is drawn on: the screen itself at x1, a camera-sized scratch otherwise
		self.zoom = 1.0
		self.freecam = False  # Camera driven by hand instead of by the followed agent
		self._free_x = 0.0
		self._free_y = 0.0
		self._dragging = False
		self._view: Surface | None = None  # Holds one zoomed frame of the world, sized to the band below
		self._scaled: Surface | None = None  # That band at screen scale, keyed so the backdrop shows through its sky
		# The part of the camera the level surface covers, which is all the world is ever drawn on. At x1 and
		# above it is the camera itself, since the camera is clamped inside the level there
		self._clip = Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
		self._draw_x = 0  # World coordinates of that band's top left corner, which every sprite is placed against
		self._draw_y = 0
		self._band_x = 0  # And where it sits in the camera, for the translucent sprites drawn on the screen
		self._band_y = 0
		self.set_world(world)
		for key, action, description in (
			(pygame.K_v, self.toggle_freecam, 'Freecam'),
			(pygame.K_PAGEUP, lambda: self.set_zoom(self.zoom * ZOOM_STEP), 'Zoom'),
			(pygame.K_PAGEDOWN, lambda: self.set_zoom(self.zoom / ZOOM_STEP), ''),
			(pygame.K_0, self.reset_view, 'Reset view'),
			(pygame.K_KP_0, self.reset_view, ''),
		):
			self.add_key_action(key, action, description)

	def set_world(self, world: World) -> None:
		"""Points the renderer at a world and bakes its level, which is everything a map change costs here."""
		self.world = world
		self._camera_lock_x = float('-inf')
		self._camera_lock_y = 0
		# World y and surface y differ by `_origin_y` on maps taller than the screen, where the map starts above it
		self._origin_y = min(0, world.offset_y)
		bottom = max(SCREEN_HEIGHT, world.offset_y + world.height * TILE_SIZE)
		size = (world.width * TILE_SIZE, bottom - self._origin_y)
		self.background = Background(size, (SCREEN_WIDTH, SCREEN_HEIGHT))
		self.level_surface = bake_level(world.kinds, size, self.to_surface(world.offset_y))
		self._ranks = np.zeros(world.count, dtype=np.int64)
		# Coins are drawn per frame instead of baked: which ones are left depends on the agent being followed.
		# Sorted by x, so a frame slices the column of them the camera covers instead of testing the whole map
		positions = np.array(world.coin_positions, dtype=np.int64).reshape(-1, 2)
		self._coin_ids = np.argsort(positions[:, 0], kind='stable')
		self._coin_x = positions[self._coin_ids, 0]
		self._coin_y = positions[self._coin_ids, 1]
		# Checkpoints are drawn per frame too: their haze is translucent, which the baked level cannot hold
		self._checkpoints = [(x, self.to_surface(y)) for x, y in world.checkpoints]

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

	def to_surface(self, world_y: int) -> int:
		"""World y to level surface y: the two only differ when the map reaches above the top of the screen."""
		return world_y - self._origin_y

	def add_key_action(self, key: int, action: Callable[[], None], description: str = '') -> None:
		self.key_actions[key] = (action, description)

	def key_hints(self) -> list[tuple[str, str]]:
		"""The described bindings, as the legend prints them: an empty description hides a binding."""
		return [(pygame.key.name(key).upper(), description) for key, (_, description) in self.key_actions.items() if description]

	def toggle_hud(self) -> None:
		self.show_hud = not self.show_hud

	def toggle_camera_lock(self) -> None:
		self.camera_lock = not self.camera_lock

	def toggle_freecam(self) -> None:
		"""Takes the camera off the followed agent, or hands it back, seeded where it already looks so nothing jumps."""
		self.freecam = not self.freecam
		self._free_x, self._free_y = float(self.camera.centerx), float(self.camera.centery)

	def reset_view(self) -> None:
		"""Back to x1 on the followed agent, which is the one key out of any corner of the level the freecam reached."""
		self.freecam = False
		self.zoom = 1.0

	def set_zoom(self, zoom: float, anchor: tuple[int, int] | None = None) -> None:
		"""
		Zooms the view, clamped to the range a per-frame scale can afford.

		`anchor` is a screen point the world pixel under it stays put at, which is what a wheel zoom over the
		cursor should do. The camera is centred on its target, so holding that pixel is a shift of the centre by
		what the point's distance from the middle of the screen is worth at either zoom. It only means anything
		while the camera is free: otherwise the next frame puts it back on its agent.
		"""
		zoom = min(MAX_ZOOM, max(MIN_ZOOM, zoom))
		if zoom == self.zoom:
			return
		if anchor is not None and self.freecam:
			self._free_x += (anchor[0] - SCREEN_WIDTH / 2) * (1 / self.zoom - 1 / zoom)
			self._free_y += (anchor[1] - SCREEN_HEIGHT / 2) * (1 / self.zoom - 1 / zoom)
		self.zoom = zoom

	def pan(self, dx: int, dy: int) -> None:
		"""Drags the view by a mouse move, taking the camera off its agent: the world follows the cursor 1:1."""
		if not self.freecam:
			self.toggle_freecam()
		self._free_x -= dx / self.zoom
		self._free_y -= dy / self.zoom

	def view_label(self) -> str:
		"""How the camera is set, for the panel the caller owns: its zoom, and whether it is off the followed agent."""
		return ('free ' if self.freecam else '') + f'x{self.zoom:.2f}'

	def poll_events(self) -> None:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				raise SystemExit
			if event.type == pygame.KEYDOWN and event.key in self.key_actions:
				self.key_actions[event.key][0]()
			elif event.type == pygame.MOUSEWHEEL:
				self.set_zoom(self.zoom * ZOOM_STEP ** event.y, pygame.mouse.get_pos())
			elif event.type == pygame.MOUSEBUTTONDOWN and event.button <= 3:
				self._dragging = True
			elif event.type == pygame.MOUSEBUTTONUP and event.button <= 3:
				self._dragging = False
			elif event.type == pygame.MOUSEMOTION and self._dragging:
				self.pan(*event.rel)

	def draw(self, focus_index: int, fitness: NDArray[np.float64], hud: Hud) -> None:
		self._move_camera(focus_index)
		view = Rect(self.camera.left, self.to_surface(self.camera.top), self.camera.width, self.camera.height)
		# Zoomed out past the map, the camera covers rows and columns the level does not have: nothing is ever
		# drawn on those, so the world is drawn on the band the level spans and the backdrop keeps the rest. At
		# x0.2 that is a quarter of the camera, and it is the difference between a 21 ms frame and a 9 ms one
		self._clip = view.clip(self.level_surface.get_rect())
		self._draw_x, self._draw_y = self._clip.left, self._clip.top + self._origin_y
		# Where that band sits in the camera, which is what turns a position in it into one on the screen
		self._band_x, self._band_y = self._clip.left - view.left, self._clip.top - view.top
		self.surface = self._view_surface()
		# The backdrop is drawn at screen size whatever the zoom: it is a parallax lie already, and keeping it out
		# of the scaled surface keeps a x1 frame exactly the handful of blits it has always been
		self.background.draw(self.screen, view.left, max(0, view.top))
		if self.surface is not self.screen:
			self.surface.fill(COLORKEY)
		self.surface.blit(self.level_surface, (0, 0), self._clip)
		self._draw_coins(focus_index)
		self._draw_enemies(focus_index)
		ghosts = self._draw_players(focus_index, fitness, hud)
		self._scale_view(view)
		# The two translucent things go on the screen once the band is down, because what alpha blends with has to
		# be the real frame: over the key colour the band's sky is, a haze comes out magenta and a ghost bright pink
		self._draw_checkpoints()
		if ghosts:
			self.screen.fblits(ghosts)
		if self.show_hud:
			self._draw_hud(hud)
		if hud.banner:
			self._draw_banner(hud.banner)
		if hud.menu is not None:
			self._draw_menu(hud.menu)
		pygame.display.flip()
		# Under vsync the flip already paces the loop; capping on top of it would make us miss every other frame
		self.clock.tick(0 if self.vsync else self.target_fps)

	def _view_surface(self) -> Surface:
		"""
		What the world is drawn on: the screen itself at x1, and a band-sized scratch under any other zoom.

		The band's size only depends on the zoom and the map, never on where the camera sits: the camera either
		covers an axis of the level whole or is clamped inside it, so panning never reallocates the surface.
		"""
		if self.zoom == 1.0:
			return self.screen
		if self._view is None or self._view.get_size() != self._clip.size:
			self._view = Surface(self._clip.size).convert()
		return self._view

	def _scale_view(self, view: Rect) -> None:
		"""Puts a zoomed frame on the screen: one nearest-neighbour scale, then a keyed blit over the backdrop."""
		if self.surface is self.screen:
			return
		size = (round(self._clip.width * self.zoom), round(self._clip.height * self.zoom))
		if self._scaled is None or self._scaled.get_size() != size:
			self._scaled = Surface(size).convert()
			self._scaled.set_colorkey(COLORKEY)
		# Nearest neighbour, so the key colour the level's air is left as survives the resize exactly
		pygame.transform.scale(self.surface, size, self._scaled)
		self.screen.blit(self._scaled, (round((self._clip.left - view.left) * self.zoom), round((self._clip.top - view.top) * self.zoom)))

	def _move_camera(self, focus_index: int) -> None:
		self.camera.size = (round(SCREEN_WIDTH / self.zoom), round(SCREEN_HEIGHT / self.zoom))
		if self.freecam:
			x, y = int(self._free_x), int(self._free_y)
		else:
			x = int(self.world.x[focus_index]) + PLAYER_W // 2
			y = int(self.world.y[focus_index]) + PLAYER_H // 2
			if self.camera_lock:
				# Only a new best past the current lock moves the camera; a generation restarting near spawn does not
				if x >= self._camera_lock_x:
					self._camera_lock_x, self._camera_lock_y = x, y
				x, y = self._camera_lock_x, self._camera_lock_y
		self.camera.centerx = x
		self.camera.centery = y
		# A map smaller than the view has no room to scroll in, so it is centred in it instead of clamped against it
		span = self.level_surface.get_width() - self.camera.width
		self.camera.left = max(0, min(self.camera.left, span)) if span > 0 else span // 2
		lowest = self._origin_y + self.level_surface.get_height() - self.camera.height
		self.camera.top = max(self._origin_y, min(self.camera.top, lowest)) if lowest > self._origin_y else (self._origin_y + lowest) // 2
		# Written back, so panning into the edge of the level does not build up an overshoot to undo on the way out
		if self.freecam:
			self._free_x, self._free_y = float(self.camera.centerx), float(self.camera.centery)

	def _draw_checkpoints(self) -> None:
		"""The checkpoints inside the camera, on the screen: a map holds a handful, so a plain culling loop is enough."""
		left, top = self.camera.left, self.to_surface(self.camera.top)
		size = max(1, round(TILE_SIZE * self.zoom))
		sprite = checkpoint_sprite(size)
		spots = [
			(round((x - left) * self.zoom), round((y - top) * self.zoom)) for x, y in self._checkpoints
			if -TILE_SIZE < x - left < self.camera.width and -TILE_SIZE < y - top < self.camera.height
		]
		if spots:
			self.screen.fblits([(sprite, spot) for spot in spots])

	def _draw_coins(self, focus_index: int) -> None:
		"""The coins the followed agent has not banked yet, culled to the camera."""
		left, top = self._draw_x, self._draw_y
		# Two binary searches cut the map down to the coins in the camera's column, whatever the map holds
		start = int(np.searchsorted(self._coin_x, left - TILE_SIZE, side='right'))
		stop = int(np.searchsorted(self._coin_x, left + self._clip.width))
		if start >= stop:
			return

		y = self._coin_y[start:stop] - top
		visible = (y > -TILE_SIZE) & (y < self._clip.height)
		visible &= ~self.world.collected[focus_index, self._coin_ids[start:stop]]
		if not visible.any():
			return

		coin = coin_sprite()
		x = self._coin_x[start:stop] - left
		self.surface.fblits([(coin, spot) for spot in zip(x[visible].tolist(), y[visible].tolist())])

	def _draw_enemies(self, focus_index: int) -> None:
		"""The enemies as the followed agent sees them: each agent wakes and stomps its own, so no two share a frame."""
		world = self.world
		if not world.enemy_count:
			return
		x, y, live, heading = world.enemy_lookup(world.enemy_step[focus_index])
		screen_x = x.astype(np.int64) - self._draw_x
		screen_y = y.astype(np.int64) - self._draw_y
		live &= ~world.stomped[focus_index]
		live &= (screen_x > -ENEMY_W) & (screen_x < self._clip.width) & (screen_y > -ENEMY_H) & (screen_y < self._clip.height)
		if live.any():
			self.surface.fblits([
				(enemy_sprite(int(side)), spot)
				for side, spot in zip(heading[live].tolist(), zip(screen_x[live].tolist(), screen_y[live].tolist()))
			])

	def _fitness_buckets(self, fitness: NDArray[np.float64]) -> NDArray[np.int64]:
		"""Buckets agents by their rank rather than their raw fitness, so the colors stay readable."""
		count = len(fitness)
		if count < 2:
			return np.zeros(count, dtype=np.int64)
		self._ranks[np.argsort(fitness, kind='stable')] = np.arange(count)
		return self._ranks * (FITNESS_BUCKETS - 1) // (count - 1)

	def _draw_players(self, focus_index: int, fitness: NDArray[np.float64], hud: Hud) -> list[Blit]:
		"""
		Draws the population into the band and hands back the ghosts, which belong on the screen instead.

		A ghost is the one player sprite that is translucent, so it cannot be blended against the key colour the
		band's sky is left as. Everything else is opaque or fully transparent per pixel, which a key handles.
		"""
		world = self.world
		screen_x = world.x.astype(np.int64) - self._draw_x
		screen_y = world.y.astype(np.int64) - self._draw_y
		visible = (screen_x > -PLAYER_W) & (screen_x < self._clip.width) & (screen_y > -PLAYER_H) & (screen_y < self._clip.height)
		if not visible.any():
			return []

		# Read as python lists: pulling 300 values out of an array one index at a time costs more than the loop
		buckets = self._fitness_buckets(fitness).tolist()
		states = (world.dead.astype(np.int64) + world.win.astype(np.int64) * 2).tolist()
		# A human body has no rank to be colored by, so its alive state points at the player fill instead
		fills = [state or 3 for state in states] if hud.solo else states
		directions = np.sign(world.change_x).astype(np.int64).tolist()
		rising = (world.change_y < 0).tolist()
		screen_x = screen_x.tolist()
		screen_y = screen_y.tolist()

		ghost = not hud.solo  # A human run has a single body, which would be unreadable faded out
		bodies: list[Blit] = []
		markers: list[Blit] = []
		ghosts: list[Blit] = []
		ghost_size = (max(1, round(PLAYER_W * self.zoom)), max(1, round(PLAYER_H * self.zoom)))
		jump = jump_sprite(JUMP_MARKER)
		elite_ring = _player_ring(ELITE_RING, 2, 2)
		random_ring = _player_ring(RANDOM_RING, 2, 2)
		focus_ring = _player_ring(FOCUS_RING, 5, 2)
		random_start = world.count - hud.random_count

		for index in np.flatnonzero(visible).tolist():
			x, y = screen_x[index], screen_y[index]
			state = states[index]
			if index == focus_index and not hud.solo:
				markers.append((focus_ring, (x - 7, y - 7)))
			# A ghost carries no rank marker either: those are only worth reading on an agent still in the run
			if ghost and state == 1:
				spot = (round((x + self._band_x) * self.zoom), round((y + self._band_y) * self.zoom))
				ghosts.append((_ghost_sprite(directions[index], ghost_size), spot))
				continue

			bodies.append((_player_sprite(buckets[index], directions[index], fills[index]), (x, y)))
			if state == 0 and rising[index]:
				markers.append((jump, (x + PLAYER_W // 2 - 6, y - 11)))
			if index < hud.elite_count:
				markers.append((elite_ring, (x - 4, y - 4)))
			elif index >= random_start:
				markers.append((random_ring, (x - 4, y - 4)))

		# fblits takes only (surface, dest) pairs, which is all these are, and draws them about 1.7x faster than blits
		if bodies:
			self.surface.fblits(bodies)
		if markers:
			self.surface.fblits(markers)
		return ghosts

	def _draw_hud(self, hud: Hud) -> None:
		"""Stacks the panels down both top corners, then places the two fixed-corner ones under them."""
		y = MARGIN
		for panel in hud.left:
			self._draw_panel(panel, MARGIN, y, panel.width)
			y += self._panel_height(panel) + MARGIN

		y = MARGIN
		for panel in hud.right:
			self._draw_panel(panel, SCREEN_WIDTH - MARGIN - panel.width, y, panel.width)
			y += self._panel_height(panel) + MARGIN

		if hud.fitness is not None:
			self._draw_fitness_panel(hud.fitness)
		if hud.legend is not None:
			self._draw_legend(hud.legend)

	def _draw_banner(self, text: str) -> None:
		"""A centred message near the top, used for the end of a run: it reads without pulling the eye off the player."""
		label = _text(BANNER_SIZE, text, VALUE_COLOR)
		width, height = label.get_width() + 4 * PADDING, label.get_height() + 2 * PADDING
		x, y = (SCREEN_WIDTH - width) // 2, MARGIN * 5
		self.screen.blit(_panel_background(width, height), (x, y))
		self.screen.blit(label, (x + 2 * PADDING, y + PADDING))

	def _draw_menu(self, menu: Menu) -> None:
		"""The list in the middle of the screen, scrolled so the selected row stays in view."""
		first = min(max(0, menu.selected - MENU_ROWS // 2), max(0, len(menu.items) - MENU_ROWS))
		shown = menu.items[first:first + MENU_ROWS]
		height = PADDING * 3 + ROW_HEIGHT * (len(shown) + 2) + 4
		x, y = (SCREEN_WIDTH - MENU_WIDTH) // 2, (SCREEN_HEIGHT - height) // 2
		self.screen.blit(_panel_background(MENU_WIDTH, height), (x, y))
		self.screen.blit(_text(TITLE_SIZE, f'{menu.title}  {menu.selected + 1}/{len(menu.items)}', TITLE_COLOR), (x + PADDING, y + PADDING))

		row_y = y + PADDING + ROW_HEIGHT + 4
		for index, item in enumerate(shown, first):
			chosen = index == menu.selected
			if chosen:
				pygame.draw.rect(self.screen, GRID_COLOR, Rect(x + 4, row_y - 1, MENU_WIDTH - 8, ROW_HEIGHT), border_radius=3)
			self.screen.blit(_text(BODY_SIZE, ('> ' if chosen else '  ') + item, ACCENT_COLOR if chosen else VALUE_COLOR), (x + PADDING, row_y))
			row_y += ROW_HEIGHT
		self.screen.blit(_text(SMALL_SIZE, 'UP/DOWN pick   ENTER load   M/ESC close', LABEL_COLOR), (x + PADDING, row_y + PADDING))

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

	def _draw_fitness_panel(self, fitness: Fitness) -> None:
		width, height = 300, 196
		x = SCREEN_WIDTH - MARGIN - width
		y = SCREEN_HEIGHT - MARGIN - height
		self.screen.blit(_panel_background(width, height), (x, y))
		self.screen.blit(_text(TITLE_SIZE, 'Fitness', TITLE_COLOR), (x + PADDING, y + PADDING))
		self.screen.blit(_text(SMALL_SIZE, 'Population Spread', LABEL_COLOR), (x + PADDING, y + PADDING + ROW_HEIGHT))

		plot = Rect(x + PADDING, y + PADDING + 2 * ROW_HEIGHT + 2, width - 2 * PADDING, 56)
		self._draw_histogram(plot, fitness.values)

		self.screen.blit(_text(SMALL_SIZE, 'Best Per Generation', LABEL_COLOR), (x + PADDING, plot.bottom + ROW_HEIGHT))
		# The right gutter holds the scale labels, so they never sit on top of the curve
		sparkline = Rect(x + PADDING, plot.bottom + 2 * ROW_HEIGHT + 6, width - 2 * PADDING - SPARKLINE_GUTTER, 38)
		self._draw_sparkline(sparkline, fitness.history)

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

	@staticmethod
	def _legend_height(legend: Legend) -> int:
		swatch_rows = (len(legend.entries) + LEGEND_COLUMNS - 1) // LEGEND_COLUMNS
		hint_rows = (len(legend.hints) + HINT_COLUMNS - 1) // HINT_COLUMNS
		rows = 1 + swatch_rows + hint_rows + legend.ramp
		return PADDING * 2 + ROW_HEIGHT * rows + (12 if legend.ramp else 4)

	def _draw_legend(self, legend: Legend) -> None:
		"""Bottom-left: what the colors on the level mean, plus the key bindings."""
		width = LEGEND_WIDTH
		height = self._legend_height(legend)
		x, y = MARGIN, SCREEN_HEIGHT - MARGIN - height
		self.screen.blit(_panel_background(width, height), (x, y))
		self.screen.blit(_text(TITLE_SIZE, 'Legend', TITLE_COLOR), (x + PADDING, y + PADDING))

		row_y = y + PADDING + ROW_HEIGHT + 4
		if legend.ramp:
			ramp = Rect(x + PADDING, row_y, width - 2 * PADDING, 6)
			slice_width = ramp.width / FITNESS_BUCKETS
			for bucket in range(FITNESS_BUCKETS):
				piece = Rect(int(ramp.left + bucket * slice_width), ramp.top, int(slice_width) + 1, ramp.height)
				pygame.draw.rect(self.screen, _bucket_color(bucket), piece)
			self.screen.blit(_text(SMALL_SIZE, 'Fitness Low', LABEL_COLOR), (x + PADDING, ramp.bottom + 2))
			high = _text(SMALL_SIZE, 'High', LABEL_COLOR)
			self.screen.blit(high, (ramp.right - high.get_width(), ramp.bottom + 2))
			row_y = ramp.bottom + ROW_HEIGHT + 2

		for index, (color, label) in enumerate(legend.entries):
			if index and index % LEGEND_COLUMNS == 0:
				row_y += ROW_HEIGHT
			column = x + PADDING + (index % LEGEND_COLUMNS) * (width - 2 * PADDING) // LEGEND_COLUMNS
			pygame.draw.rect(self.screen, color, Rect(column, row_y + 3, 8, 8), border_radius=2)
			self.screen.blit(_text(SMALL_SIZE, label, LABEL_COLOR), (column + 13, row_y))

		row_y += ROW_HEIGHT
		for index, (name, description) in enumerate(legend.hints):
			if index and index % HINT_COLUMNS == 0:
				row_y += ROW_HEIGHT
			column = x + PADDING + (index % HINT_COLUMNS) * (width - 2 * PADDING) // HINT_COLUMNS
			self.screen.blit(_text(SMALL_SIZE, f'{name:<5} {description}', LABEL_COLOR), (column, row_y))

	@staticmethod
	def quit() -> None:
		pygame.quit()

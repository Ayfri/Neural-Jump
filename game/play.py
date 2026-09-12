import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pygame

from game.art import COIN_COLOR
from game.constants import MOVE_IDLE, MOVE_JUMP, MOVE_LEFT, MOVE_RIGHT
from game.menu import Item, MenuStack, Page, Screen
from game.render import ZOOM_STEP, Gauge, Hud, Legend, PLAY_LEGEND, Panel, Renderer
from game.settings import PLAYER_SPEED, TILE_SIZE
from game.world import World, list_maps

DEFAULT_TICK_RATE: Final[int] = 90  # The rate the agents are trained at, so a human run is comparable to theirs
SPEED_STEP: Final[float] = 2.0  # Factor the slower and faster keys apply
MIN_SPEED: Final[float] = 0.1  # Slow motion, which is the only way to read a jump arc frame by frame
MAX_SPEED: Final[float] = 4.0
RESPAWN_DELAY: Final[float] = 1.2  # Seconds the end-of-run banner stays up before the next attempt starts

# Both hands and both layouts: Z and Q are where W and A sit on an AZERTY keyboard
LEFT_KEYS: Final[tuple[int, ...]] = (pygame.K_LEFT, pygame.K_a, pygame.K_q)
RIGHT_KEYS: Final[tuple[int, ...]] = (pygame.K_RIGHT, pygame.K_d)
JUMP_KEYS: Final[tuple[int, ...]] = (pygame.K_UP, pygame.K_w, pygame.K_z, pygame.K_SPACE)

MENU_FOOTER: Final[str] = 'Up/Down move   Enter pick   Esc back'
SETTINGS_FOOTER: Final[str] = 'Left/Right change   Enter toggle   Esc back'

CONTROLS: Final[tuple[tuple[str, str], ...]] = (
	('Arrows, WASD, ZQSD', 'Move'),
	('Space, Up, W', 'Jump, aimed by the direction held on the same tick'),
	('Esc', 'Open this menu, or step back out of it'),
	('P', 'Pause and resume'),
	('Tab', 'Hide the panels, the banner stays'),
	('R', 'Retry from the current spawn point'),
	('G', 'Start on the next spawn point'),
	('1, -, =', 'Speed x1, halve, double, also on the numpad'),
	('V, drag', 'Free the camera and pan it'),
	('Wheel, PgUp, PgDn', 'Zoom, x0.2 out to x4 in'),
	('0', 'Back to x1 on yourself'),
)


class PlaySession:
	"""
	A human-played run of the level.

	It runs on the same batched `World` as the training loop, with one slot instead of three hundred, and
	draws through the same renderer, so the physics under your feet and the panels around them are exactly
	the ones the agents are scored on. The menu stack is the state it is in: an empty one is the level being
	played, and anything on it freezes the run behind the page it is showing.
	"""

	def __init__(
		self,
		map_path: str = 'maps/level_1.txt',
		tick_rate: int = DEFAULT_TICK_RATE,
		fps: int = 0,
		spawn: int = 0,
		renderer: Renderer | None = None,
	) -> None:
		self.map_path = Path(map_path).as_posix()
		self.world = World(map_path, 1)
		self.tick_rate = tick_rate
		# A shell hands its window on rather than building a second one, which would re-initialise the display
		if renderer is None:
			self.renderer = Renderer(self.world, fps)
		else:
			self.renderer = renderer
			renderer.reset_key_actions()
			renderer.set_world(self.world)
		self.menu = MenuStack()
		self.spawn_points: list[tuple[int, int]] = []
		self.spawn_index = 0

		self.speed = 1.0
		self.paused = False
		self.running = True
		self.to_title = False  # Whether the session ended by asking for the title screen rather than by closing the window
		self.tick = 0
		self.attempt = 1
		self.deaths = 0
		self.wins = 0
		self.best_time = 0.0  # Fastest run to the flag in seconds, 0 while it has never been touched
		self.best_coins = 0
		self.banner = ''
		self._ended_at = 0.0  # Real time the run ended on, 0 while it is still live
		self._budget = 0.0
		self._actions = np.zeros(1, dtype=np.int64)

		# Only what is worth a key while playing: everything else is a row on the pause menu
		bindings: list[tuple[int, Callable[[], None], str]] = [
			(pygame.K_ESCAPE, self.open_menu, 'Menu'),
			(pygame.K_p, self.toggle_pause, 'Pause'),
			(pygame.K_TAB, self.renderer.toggle_hud, 'Panels'),
			(pygame.K_r, self.restart, 'Retry'),
			(pygame.K_g, self.next_spawn, ''),
			(pygame.K_1, lambda: self.set_speed(1.0), ''),
			(pygame.K_MINUS, lambda: self.set_speed(self.speed / SPEED_STEP), ''),
			(pygame.K_EQUALS, lambda: self.set_speed(self.speed * SPEED_STEP), ''),
			(pygame.K_KP_MINUS, lambda: self.set_speed(self.speed / SPEED_STEP), ''),
			(pygame.K_KP_PLUS, lambda: self.set_speed(self.speed * SPEED_STEP), ''),
		]
		for key, action, description in bindings:
			self.renderer.add_key_action(key, action, description)

		self._start_level(spawn)

	@property
	def level_name(self) -> str:
		return self.map_path.removeprefix('maps/').removesuffix('.txt')

	def _start_level(self, spawn: int) -> None:
		"""Starts the loaded level over: its spawn points, a clean record sheet and the first attempt."""
		self.spawn_points = [self.world.spawn_point, *self.world.checkpoints]
		self.spawn_index = min(max(0, spawn), len(self.spawn_points) - 1)
		self.attempt, self.deaths, self.wins = 1, 0, 0
		self.best_time, self.best_coins = 0.0, 0
		self.restart(count_attempt=False)

	# --- Menus -------------------------------------------------------------------------------------------

	def open_menu(self) -> None:
		self.menu.push(self.pause_page())

	def _resuming(self, action: Callable[[], None]) -> Callable[[], None]:
		"""Wraps a row so it does its work and hands the level straight back, which is what most of them want."""
		def run() -> None:
			action()
			self.menu.close()
		return run

	def pause_page(self) -> Page:
		return Page(Screen.PAUSED, 'Paused', [
			Item('Resume', self.menu.close),
			Item('Retry from spawn', self._resuming(self.restart)),
			Item('Next spawn point', self._resuming(self.next_spawn), value=lambda: f'{self.spawn_index + 1}/{len(self.spawn_points)}'),
			Item('Settings', lambda: self.menu.push(self.settings_page())),
			Item('Select level', lambda: self.menu.push(self.maps_page())),
			Item('Controls', lambda: self.menu.push(self.controls_page())),
			Item('Quit to title', self.quit_to_title),
		], subtitle=self.level_name, footer=MENU_FOOTER)

	def settings_page(self) -> Page:
		renderer = self.renderer
		return Page(Screen.SETTINGS, 'Settings', [
			Item('Simulation speed', adjust=lambda step: self.set_speed(self.speed * SPEED_STEP ** step), value=lambda: f'x{self.speed:g}'),
			Item('Spawn point', adjust=self.step_spawn, value=lambda: f'{self.spawn_index + 1}/{len(self.spawn_points)}'),
			Item('Zoom', adjust=lambda step: renderer.set_zoom(renderer.zoom * ZOOM_STEP ** step), value=lambda: f'x{renderer.zoom:.2f}'),
			Item('Free camera', renderer.toggle_freecam, value=lambda: 'on' if renderer.freecam else 'off'),
			Item('Panels', renderer.toggle_hud, value=lambda: 'shown' if renderer.show_hud else 'hidden'),
			Item('Reset the view', renderer.reset_view),
			Item('Back', self.menu.back),
		], subtitle='Slow motion reads a jump arc frame by frame', footer=SETTINGS_FOOTER)

	def maps_page(self) -> Page:
		"""The levels under `maps/`, listed again on every open so one imported mid-session shows up."""
		maps = list_maps()
		items = [Item(path.removeprefix('maps/').removesuffix('.txt'), self._loader(path)) for path in maps]
		page = Page(Screen.MAPS, 'Select level', items, subtitle=f'{len(maps)} levels, loading one starts the records over', footer=MENU_FOOTER)
		page.select(maps.index(self.map_path) if self.map_path in maps else 0)
		return page

	def _loader(self, map_path: str) -> Callable[[], None]:
		return lambda: self.load_map(map_path)

	def controls_page(self) -> Page:
		return Page(Screen.HELP, 'Controls', [
			*(Item(key, value=lambda text=description: text) for key, description in CONTROLS),
			Item('Back', self.menu.back),
		], footer='Esc back')

	def load_map(self, map_path: str) -> None:
		"""Swaps the level under the session, records included, since a time on one map says nothing about another."""
		self.map_path = Path(map_path).as_posix()
		self.world = World(map_path, 1)
		self.renderer.set_world(self.world)
		self._start_level(0)
		self.menu.close()

	def quit_to_title(self) -> None:
		"""Ends the session and hands the window back, which the shell turns into the title screen again."""
		self.menu.close()
		self.running = False
		self.to_title = True

	# --- The run -----------------------------------------------------------------------------------------

	@property
	def finished(self) -> bool:
		return bool(self.world.dead[0] or self.world.win[0])

	@property
	def frozen(self) -> bool:
		"""Whether the level is standing still, which a menu does as surely as the pause key."""
		return self.paused or self.menu.open

	def toggle_pause(self) -> None:
		self.paused = not self.paused

	def set_speed(self, speed: float) -> None:
		"""Slow motion down to a tenth, fast forward up to four times, applied to the simulation not the framerate."""
		self.speed = min(MAX_SPEED, max(MIN_SPEED, speed))
		self._budget = 0.0

	def step_spawn(self, step: int) -> None:
		"""Moves along the spawn points, wrapping both ways, so a late section can be practised on its own."""
		self.spawn_index = (self.spawn_index + step) % len(self.spawn_points)
		self.restart()

	def next_spawn(self) -> None:
		self.step_spawn(1)

	def restart(self, count_attempt: bool = True) -> None:
		self.world.reset(*self.spawn_points[self.spawn_index])
		self.tick = 0
		self.attempt += count_attempt
		self.banner = ''
		self._ended_at = 0.0
		self._budget = 0.0

	def run(self) -> bool:
		"""Plays until the session ends, returning whether the window is still open for whatever comes next."""
		try:
			while self.running:
				self.renderer.poll_events(self.menu)
				self.advance()
				self.renderer.draw(0, np.zeros(1), self.hud())
		except (KeyboardInterrupt, SystemExit):
			self.running, self.to_title = False, False
		best = f'{self.best_time:.2f}s' if self.best_time else 'never finished'
		print(f'{self.attempt} attempts, {self.wins} wins, {self.deaths} deaths, best time: {best}, best coins: {self.best_coins}')
		return self.to_title

	def advance(self) -> None:
		"""One frame's worth of simulation, so the tick rate stays fixed whatever the framerate does."""
		if self.frozen:
			# Nothing accumulates while frozen, so resuming does not fire a burst of catch up ticks
			self._budget = 0.0
			return

		if self.finished:
			if time.perf_counter() - self._ended_at >= RESPAWN_DELAY:
				self.restart()
			return

		self._budget += self.tick_rate * self.speed / self.renderer.target_fps
		steps = int(self._budget)
		self._budget -= steps

		held = pygame.key.get_pressed()
		for _ in range(steps):
			self._actions[0] = self.read_action(held)
			self.world.step(self._actions, self.tick)
			self.tick += 1
			if self.finished:
				self.end_run()
				break

	def read_action(self, held: Sequence[bool]) -> int:
		"""The held keys as one of the four moves, aiming a jump before it leaves the ground."""
		direction = any(held[key] for key in RIGHT_KEYS) - any(held[key] for key in LEFT_KEYS)
		if any(held[key] for key in JUMP_KEYS) and bool(self.world.grounded()[0]):
			# A jump keeps whatever horizontal speed it left with, so the direction has to be set on that same tick
			self.world.change_x[0] = direction * PLAYER_SPEED
			return MOVE_JUMP
		if direction:
			return MOVE_RIGHT if direction > 0 else MOVE_LEFT
		return MOVE_IDLE

	def end_run(self) -> None:
		"""Books the finished run and puts up the banner the next attempt waits behind."""
		self._ended_at = time.perf_counter()
		coins = int(self.world.coins[0])
		self.best_coins = max(self.best_coins, coins)
		if not self.world.win[0]:
			self.deaths += 1
			self.banner = f'Dead at tile {int(self.world.x[0]) // TILE_SIZE}'
			return

		self.wins += 1
		seconds = self.tick / self.tick_rate
		record = self.best_time == 0.0 or seconds < self.best_time
		self.best_time = seconds if record else self.best_time
		self.banner = f'Flag in {seconds:.2f}s, {coins} coins' + ('   NEW BEST' if record else '')

	def hud(self) -> Hud:
		world = self.world
		fps = self.renderer.measured_fps()
		coins = int(world.coins[0])
		progress = float(world.x[0]) / max(1, (world.width - 1) * TILE_SIZE)

		run = Panel('Run', [
			('Time', f'{self.tick / self.tick_rate:.2f}s'),
			('Best', f'{self.best_time:.2f}s' if self.best_time else '-'),
			Gauge('Progress', f'{progress * 100:.0f}%', progress),
			Gauge('Coins', f'{coins}/{world.coin_count}', coins / max(1, world.coin_count), COIN_COLOR),
			('Attempt', f'{self.attempt}'),
			('Wins', f'{self.wins}  deaths {self.deaths}'),
			Gauge('FPS', f'{fps:.0f}/{self.renderer.target_fps}', fps / max(1, self.renderer.target_fps)),
		])
		player = Panel('Player', [
			('Map', self.level_name),
			('Position', f'{int(world.x[0])}, {int(world.y[0])}'),
			('Tile', f'{int(world.x[0]) // TILE_SIZE}/{world.width}'),
			('Speed', f'{world.change_x[0]:+.0f}, {world.change_y[0]:+.0f}'),
			('Ground', 'yes' if bool(world.grounded()[0]) else 'no'),
			('Spawn', f'{self.spawn_index + 1}/{len(self.spawn_points)}'),
			('Sim speed', f'x{self.speed:g}'),
			('View', self.renderer.view_label()),
			('Best coins', f'{self.best_coins}/{world.coin_count}'),
		])
		hints = [('WASD', 'Move'), ('SPACE', 'Jump'), *self.renderer.key_hints()]

		return Hud(
			left=[run],
			right=[player],
			legend=Legend(PLAY_LEGEND, hints),
			banner='PAUSED' if self.paused else self.banner,
			menu=self.menu.page,
			solo=True,
		)

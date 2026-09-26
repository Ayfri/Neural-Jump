"""
The app around the two runs: a title screen, and whichever of them it starts.

This is the one module that sees both halves, which is why it sits at the root rather than under `game/`: the
world never imports the policy that drives it. It owns the window, so a run that ends hands it back instead of
tearing it down and the title screen comes up again on the same renderer, which is what makes switching between
playing a level and training on it cost nothing but the simulation each one needs.
"""
from collections.abc import Callable
from pathlib import Path
from typing import Final

import numpy as np

from ai.generation import DEFAULT_ENV_COUNT, DEFAULT_POPULATION_SIZE, TRAINERS, Generation
from game.menu import Item, MenuStack, Page, level_name
from game.play import DEFAULT_TICK_RATE, PlaySession
from game.render import Hud, Renderer
from game.world import World

# What the title screen needs to build a trainer: the level and the trainer it picked, on the window it owns.
# The caller closes over everything else, so a command line's knobs reach the run the title screen starts.
type TrainerFactory = Callable[[str, str, Renderer], Generation]

TRAINER_LABELS: Final[dict[str, str]] = {
	'ppo': 'PPO, one shared policy',
	'ga': 'Evolution, a population',
}
TITLE_FOOTER: Final[str] = 'Up/Down move   Left/Right change   Enter pick'


class Shell:
	"""
	The title screen and what it launches.

	A run is started from a menu row, but never from inside the event handler that fired it: the row records
	what was asked for and the loop acts on it once the frame is out, so a session's own loop never starts
	nested inside the poll of the one above it.
	"""

	def __init__(
		self,
		map_path: str = 'maps/level_1.txt',
		trainer: str = 'ppo',
		tick_rate: int = DEFAULT_TICK_RATE,
		fps: int = 0,
		generations: int = 0,
		build_trainer: TrainerFactory | None = None,
	) -> None:
		self.map_path = Path(map_path).as_posix()
		self.trainer = trainer if trainer in TRAINERS else TRAINERS[0]
		self.tick_rate = tick_rate
		self.generations = generations
		self.build_trainer = build_trainer or self._default_trainer
		self.world = World(self.map_path, 1)
		self.renderer = Renderer(self.world, fps)
		self.menu = MenuStack()
		self.menu.reset(self.title_page())
		self.running = True
		self._request = ''  # What a row asked for, acted on once the frame it was clicked in is out

	def title_page(self) -> Page:
		"""The root page: it is never popped, so Escape inside it goes nowhere and the app always has a screen."""
		return Page('Neural-Jump', [
			Item('Play the level yourself', self._asking('play')),
			Item('Train a policy on it', self._asking('train')),
			Item('Trainer', adjust=self.step_trainer, value=lambda: TRAINER_LABELS[self.trainer]),
			Item('Level', lambda: self.menu.push(self.menu.maps_page(self.map_path, self.pick_map, 'The level both rows above start on')),
				value=lambda: level_name(self.map_path)),
			Item('Quit', self.quit),
		], subtitle='A thousand networks learning one platformer', footer=TITLE_FOOTER, root=True)

	def pick_map(self, map_path: str) -> None:
		self.map_path = Path(map_path).as_posix()
		self.world = World(self.map_path, 1)
		self.renderer.set_world(self.world)
		self.menu.back()

	def _asking(self, request: str) -> Callable[[], None]:
		def ask() -> None:
			self._request = request
		return ask

	def step_trainer(self, step: int) -> None:
		self.trainer = TRAINERS[(TRAINERS.index(self.trainer) + step) % len(TRAINERS)]

	def quit(self) -> None:
		self.running = False

	def run(self) -> None:
		"""Shows the title screen, and whatever it starts, until the window closes or Quit is picked."""
		try:
			while self.running:
				self.renderer.poll_events(self.menu)
				self.renderer.draw(0, np.zeros(1), Hud(menu=self.menu.page, solo=True))
				if self._request:
					request, self._request = self._request, ''
					self._launch(request)
		except (KeyboardInterrupt, SystemExit):
			pass
		finally:
			self.renderer.quit()

	def _launch(self, request: str) -> None:
		"""Runs one session on the shell's window and takes it back, or stops the app if the window went with it."""
		alive = self._play() if request == 'play' else self._train()
		if not alive:
			self.running = False
			return
		# The session may have loaded another level, and the title screen shows the one actually last played
		self.world = World(self.map_path, 1)
		self.renderer.reset_key_actions()
		self.renderer.set_world(self.world)
		self.renderer.reset_view()
		self.menu.reset(self.title_page())

	def _play(self) -> bool:
		session = PlaySession(self.map_path, self.tick_rate, renderer=self.renderer)
		alive = session.run()
		self.map_path = session.map_path
		return alive

	def _default_trainer(self, map_path: str, trainer: str, renderer: Renderer) -> Generation:
		"""What the title screen builds when nothing richer was handed in, which is every default `run-ai` takes."""
		return Generation(
			DEFAULT_ENV_COUNT if trainer == 'ppo' else DEFAULT_POPULATION_SIZE,
			trainer=trainer, map_path=map_path, tick_rate=self.tick_rate, show_window=True, renderer=renderer,
		)

	def _train(self) -> bool:
		"""Builds the trainer behind a message, since compiling and capturing the simulation is seconds of silence."""
		self.renderer.draw_message('Building the simulation', 'compiling and capturing, a few seconds')
		generation = self.build_trainer(self.map_path, self.trainer, self.renderer)
		try:
			return generation.run(self.generations)
		finally:
			self.map_path = generation.map_path

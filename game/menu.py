"""
The menu system: a stack of pages, the last one being what the app is showing.

Nothing here knows about a world, a trainer or pygame. A page is a list of rows and the callbacks they fire, so
the same few dozen lines draw the title screen, a pause menu and a settings list, and an app only says which
pages exist and what their rows do. The stack is the single source of truth for the state the app is in: an
empty one is the simulation running, anything else is a menu over a frozen one.
"""
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final

from game.world import list_maps

MENU_FOOTER: Final[str] = 'Up/Down move   Enter pick   Esc back'
SETTINGS_FOOTER: Final[str] = 'Left/Right change   Enter toggle   Esc back'


def level_name(map_path: str) -> str:
	"""A map path as the menus and panels print it: `maps/smb/1-1.txt` is `smb/1-1`."""
	return map_path.removeprefix('maps/').removesuffix('.txt')


@dataclass(slots=True)
class Item:
	"""
	One row of a page.

	`action` fires on Enter or a click, `adjust` on Left/Right, and `value` is read every frame so a setting
	shows what it currently is rather than what it was when the page opened. A row with neither action nor
	adjust is a heading, which the cursor skips over.
	"""
	label: str
	action: Callable[[], None] | None = None
	adjust: Callable[[int], None] | None = None
	value: Callable[[], str] | None = None

	@property
	def usable(self) -> bool:
		return self.action is not None or self.adjust is not None


@dataclass(slots=True)
class Page:
	"""One screen of menu. `root` marks the one page the stack refuses to pop, which is the title screen."""
	title: str
	items: list[Item] = field(default_factory=list)
	subtitle: str = ''
	footer: str = ''
	selected: int = 0
	root: bool = False

	def __post_init__(self) -> None:
		if not self._at(self.selected):
			self.move(1)

	def _at(self, index: int) -> bool:
		return 0 <= index < len(self.items) and self.items[index].usable

	@property
	def current(self) -> Item | None:
		return self.items[self.selected] if self._at(self.selected) else None

	def move(self, step: int) -> None:
		"""Moves the cursor by `step`, wrapping around the page and stepping over the rows that are only headings."""
		index = self.selected
		for _ in range(len(self.items)):
			index = (index + step) % len(self.items)
			if self._at(index):
				self.selected = index
				return

	def select(self, index: int) -> None:
		"""Puts the cursor on the row the mouse is over, ignoring the headings it passes across."""
		if self._at(index):
			self.selected = index

	def activate(self) -> None:
		item = self.current
		if item is not None and item.action is not None:
			item.action()

	def adjust(self, step: int) -> None:
		item = self.current
		if item is not None and item.adjust is not None:
			item.adjust(step)


class MenuStack:
	"""
	The pages currently open, the last one being the one drawn.

	Pages are pushed rather than swapped, so `back` always has somewhere to return to, and the whole state of
	the app is `open` plus the top page. Nothing else needs to be kept in sync with it.
	"""

	def __init__(self) -> None:
		self.pages: list[Page] = []

	@property
	def page(self) -> Page | None:
		return self.pages[-1] if self.pages else None

	@property
	def open(self) -> bool:
		return bool(self.pages)

	def push(self, page: Page) -> None:
		self.pages.append(page)

	def reset(self, page: Page) -> None:
		"""Throws the stack away and starts again on one page, which is how a run hands the app back to the title."""
		self.pages = [page]

	def back(self) -> None:
		"""Closes the top page, unless it is the root one: a title screen has nothing behind it to fall back to."""
		if self.pages and not self.pages[-1].root:
			self.pages.pop()

	def close(self) -> None:
		"""Closes every page that can be closed, which is what resuming from a nested settings list does."""
		while self.pages and not self.pages[-1].root:
			self.pages.pop()

	def resuming(self, action: Callable[[], None]) -> Callable[[], None]:
		"""Wraps a row so it does its work and closes the menu, handing the run straight back."""
		def run() -> None:
			action()
			self.close()
		return run

	def maps_page(self, current: str, pick: Callable[[str], None], subtitle: str) -> Page:
		"""The levels under `maps/`, listed again on every open so one imported mid-run shows up, the cursor on `current`."""
		maps = list_maps()
		page = Page('Select level', [Item(level_name(path), lambda path=path: pick(path)) for path in maps], subtitle=subtitle, footer=MENU_FOOTER)
		page.select(maps.index(current) if current in maps else 0)
		return page

	def controls_page(self, controls: Sequence[tuple[str, str]]) -> Page:
		"""A key and what it does per row, read-only, with a way back out."""
		return Page('Controls', [
			*(Item(key, value=lambda text=description: text) for key, description in controls),
			Item('Back', self.back),
		], footer='Esc back')

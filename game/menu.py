"""
The menu system: a stack of pages, the last one being what the app is showing.

Nothing here knows about a world, a trainer or pygame. A page is a list of rows and the callbacks they fire, so
the same few dozen lines draw the title screen, a pause menu and a settings list, and an app only says which
pages exist and what their rows do. The stack is the single source of truth for the state the app is in: an
empty one is the simulation running, anything else is a menu over a frozen one.
"""
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto


class Screen(Enum):
	"""What the app is showing. `PLAYING` is the only state the simulation advances in."""
	PLAYING = auto()
	TITLE = auto()
	PAUSED = auto()
	MAPS = auto()
	SETTINGS = auto()
	HELP = auto()


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
	screen: Screen
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
	the app is `open` plus the top page's `screen`. Nothing else needs to be kept in sync with it.
	"""

	def __init__(self) -> None:
		self.pages: list[Page] = []

	@property
	def page(self) -> Page | None:
		return self.pages[-1] if self.pages else None

	@property
	def screen(self) -> Screen:
		page = self.page
		return page.screen if page is not None else Screen.PLAYING

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

	def toggle(self, page: Page) -> None:
		"""Opens `page`, or closes what is open: the one thing a key that both opens and dismisses a menu needs."""
		if self.open:
			self.close()
			return
		self.push(page)

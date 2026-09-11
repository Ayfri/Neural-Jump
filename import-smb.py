"""
Imports the Super Mario Bros levels of the Video Game Level Corpus (github.com/TheVGLC/TheVGLC) as maps.

The corpus stores a level as one character per tile, which is this game's own format under other letters:
every solid kind (ground, bricks, question blocks, pipes, cannons) becomes '#', enemies stay 'E' and coins 'o'.
It marks neither where Mario starts nor the flag, so the spawn goes on the floor near the left edge and a flag
gate fills the air over the last column that has a floor. Its floor is one row thick and SMB's is two, so the
bottom row is doubled. The levels are Nintendo's, so they are downloaded
on demand rather than committed.
"""
import argparse
from pathlib import Path
from typing import Final
from urllib.request import urlopen

SOURCE: Final[str] = 'https://raw.githubusercontent.com/TheVGLC/TheVGLC/master/Super%20Mario%20Bros/Processed/mario-{}.txt'
LEVELS: Final[tuple[str, ...]] = ('1-1', '1-2', '1-3', '2-1', '3-1', '3-3', '4-1', '4-2', '5-1', '5-3', '6-1', '6-2', '6-3', '7-1', '8-1')
SOLID: Final[str] = 'XS?Q<>[]Bb'
KEPT: Final[str] = 'oE'
SPAWN_COLUMN: Final[int] = 3  # Where Mario stands at the start of every level


def convert(lines: list[str]) -> list[str]:
	"""One corpus level as map rows: tiles translated, then a spawn point and a flag gate added."""
	width = max(len(line) for line in lines)
	grid = [['#' if char in SOLID else char if char in KEPT else '.' for char in line.ljust(width, '-')] for line in lines]
	# The corpus crops the floor to one row while a player dies two rows above the map bottom, so it gets SMB's second row back
	grid.append(grid[-1].copy())

	def floor(column: int) -> int:
		"""The row over the lowest solid tile of a column with air above it, or -1 over a pit."""
		return next((row - 1 for row in range(len(grid) - 1, 0, -1) if grid[row][column] == '#' and grid[row - 1][column] != '#'), -1)

	start = next(column for column in range(SPAWN_COLUMN, width) if floor(column) >= 0)
	grid[floor(start)][start] = 'P'
	end = next(column for column in range(width - 1, -1, -1) if floor(column) >= 0)
	for row in range(floor(end) + 1):
		if grid[row][end] != '#':
			grid[row][end] = 'F'
	return [''.join(row) for row in grid]


def main() -> None:
	parser = argparse.ArgumentParser(prog='import-smb', description='Download the Super Mario Bros levels of the VGLC and convert them to maps.')
	parser.add_argument('levels', nargs='*', choices=LEVELS, metavar='LEVEL', help='world-level pairs such as 1-1, every one the corpus holds when omitted')
	parser.add_argument('--out', type=Path, default=Path('maps/smb'), help='folder the maps are written to')
	args = parser.parse_args()

	folder: Path = args.out
	folder.mkdir(parents=True, exist_ok=True)
	levels: list[str] = args.levels or list(LEVELS)
	for level in levels:
		with urlopen(SOURCE.format(level)) as response:
			lines = [line for line in response.read().decode('ascii').splitlines() if line.strip()]
		rows = convert(lines)
		path = folder / f'{level}.txt'
		path.write_text('\n'.join(rows) + '\n')
		enemies, coins = sum(row.count('E') for row in rows), sum(row.count('o') for row in rows)
		print(f'{path}: {len(rows[0])}x{len(rows)} tiles, {enemies} enemies, {coins} coins')


if __name__ == '__main__':
	main()

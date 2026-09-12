from game.play import DEFAULT_TICK_RATE, PlaySession


def start_game(map_path: str = 'maps/level_1.txt', tick_rate: int = DEFAULT_TICK_RATE, fps: int = 0, spawn: int = 0) -> None:
	"""Plays the level on its own window, which is `run-game.py --no-menu` rather than the title screen's Play row."""
	session = PlaySession(map_path, tick_rate, fps, spawn)
	session.run()
	session.renderer.quit()


if __name__ == '__main__':
	start_game()

from game.game import Game


def start_game() -> None:
	game = Game(tick_rate=90)
	game.start()


if __name__ == "__main__":
	start_game()

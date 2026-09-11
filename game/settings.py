# Screen settings
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 900

# Tiles settings
TILE_SIZE = 40

# Player settings
PLAYER_WIDTH = TILE_SIZE * 0.8
PLAYER_HEIGHT = TILE_SIZE * 0.95
PLAYER_GRAVITY = 0.8
PLAYER_JUMP_STRENGTH = -17
PLAYER_SPEED = 8

# Enemy settings
ENEMY_WIDTH = TILE_SIZE * 0.8
ENEMY_HEIGHT = TILE_SIZE * 0.8
ENEMY_SPEED = 2
ENEMY_WAKE_DISTANCE = TILE_SIZE * 10  # How far ahead of a player an enemy starts walking, like one scrolling onto the screen
STOMP_BOUNCE = -10  # Vertical speed a player leaves a stomped enemy with

"""What a run pays out, in one place: the per-tick terms, the end of episode terms and the tile payouts."""
from typing import Final

from game.tiles import TILE_REWARDS, TileKind

# Per tick, accumulated while the episode runs
FORWARD_MOVEMENT_REWARD: Final[float] = 0.02
NEW_MAX_POSITION_BONUS: Final[float] = 0.1
BACKWARD_MOVEMENT_PENALTY: Final[float] = -0.1
STATIONARY_PENALTY: Final[float] = -0.05
STATIONARY_THRESHOLD: Final[int] = 5  # Ticks before penalty kicks in
FALLING_PENALTY: Final[float] = -0.02
FALLING_THRESHOLD: Final[int] = 5  # Y distance before penalty

# End of episode
COIN_REWARD: Final[float] = TILE_REWARDS[TileKind.COIN]  # Every tile's payout lives in one table
DEATH_PENALTY: Final[float] = -20.0
WIN_BASE_BONUS: Final[float] = 200.0  # Paid for touching the flag at all, whatever the time taken
WIN_SPEED_BONUS: Final[float] = 1200.0  # Paid on top, scaled by how much of the episode was still left
WIN_SPEED_EXPONENT: Final[float] = 2.0  # Bends the scale, so shaving ticks off an already fast run pays the most
PROGRESS_SPEED_BONUS: Final[float] = 100.0  # Same idea for agents that never reach the flag, on how fast they got as far as they did
DISTANCE_REWARD_DIVISOR: Final[float] = 10.0
PROGRESS_REWARD_DIVISOR: Final[float] = 20.0
MIN_REWARD: Final[float] = -30.0

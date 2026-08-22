# Neural-Jump

A platformer where a whole population of neural networks plays the level at once. No gradients, no reward
backprop: 300 agents run the level in parallel, the best few breed, their children get mutated, repeat until
someone touches the flag.

The interesting part is that everything is batched. The world is numpy arrays with one slot per agent, the
population is one set of `(agents, in, out)` weight tensors, and a tick steps all of them together. On a
4060 Ti that is a million agent-steps per second.

## Quick start

```bash
uv sync
uv run run-ai.py                              # headless training, as fast as the machine allows
uv run run-ai.py --show-window --speed max    # watch it, as fast as the framerate survives
uv run run-game.py                            # play the level yourself
uv run run-game.py --spawn 3                  # start on the third checkpoint
```

Python 3.13+, pygame-ce for the window, and `uv sync` pulls torch with CUDA 13 on Windows and Linux, plus
the triton wheel torch does not ship on Windows, which is what compiles the simulation. CPU works too, it is
just slower: the whole tick then runs in numpy on the host instead.

The first generation of a run is slower than the rest, because that is where the simulation is compiled.

## Options

*evolution* - how a generation is selected and bred

- `--population-size N`: agents per generation (default: 300). A tick at 300 agents costs about 1.4x a tick
  at 100 while trying three times as many mutations, and those extra mutations are what break a plateau
- `--elite-count N`: agents carried over untouched and used as parents (default: 6)
- `--mutation-rate R`: probability that a child's weight tensor is mutated at all (default: 0.8)
- `--mutation-strength S`: scale of the noise added to a mutated tensor (default: 0.02)

*network* - shape and placement of the policy

- `--hidden-sizes N1 N2 N3`: sizes of the three hidden layers (default: 256 128 64), smaller is faster and dumber
- `--device auto|cpu|cuda`: where the population runs (default: auto)
- `--threads N`: torch CPU threads (default: 4)

*simulation* - the level and the episode played on it

- `--map PATH`: level file to train on (default: maps/level_1.txt)
- `--episode-seconds S`: in-game time budget per spawn point (default: 60)
- `--tick-rate N`: simulation ticks per in-game second (default: 90)
- `--action-repeat N`: physics ticks a chosen action is held for (default: 2)
- `--checkpoints`: use the level checkpoints as extra spawn points

*run* - where the run starts and when it stops

- `--generations N`: stop after N generations (default: 0, runs forever)
- `--seed N`: seed python, numpy and torch so a run replays exactly
- `--load-latest-generation-weights`: start from the most recent weight file

*display* - only meaningful together with `--show-window`

- `--show-window`: render the run instead of training headless
- `--speed S`: simulation speed multiplier, or `max` to run as fast as the framerate survives (default: 1)
- `--fps N`: target framerate (default: 0, uses the display refresh rate)

```bash
uv run run-ai.py --population-size 1000 --hidden-sizes 64 32 16   # a big dumb crowd
uv run run-ai.py --seed 1                                         # replays exactly
uv run run-ai.py --show-window --speed 4                          # watchable
```

## Reading the window

Every agent's state is in its sprite, so one glance reads the whole population:

- **Body color**: fitness rank, red (worst) to teal (best). **Blue body**: reached the flag
- **Faded grey ghost with shut eyes**: dead, dimmed out of the way of the agents still running, and carrying no outline
- **Heading**: the face slides forward, both pupils sit against the front of their whites and the light moves to the
  leading edge; standing still, the face is centred and symmetric. **Blue arrow above the head**: rising, so it jumped
- **Gold outline**: an elite carried over untouched. **Violet outline**: a re-randomised agent, kept for diversity
- **White outline**: the agent the camera follows, the best one still alive

The panels cover the run, the followed agent, the hyper-parameters, and the fitness distribution next to the
best score of every generation so far.

## Keys

| Key | What it does |
| --- | --- |
| `Space` | Pause and resume, the window stays live |
| `Tab` | Hide the panels, leaving the level and the agents |
| `1` | Back to speed x1 |
| `M` | Speed `max`, which tunes itself to the framerate |
| `-` / `=` | Halve or double the speed, also on the numpad. From `max` it starts at the multiplier it had reached |
| `G` | Skip to the next spawn point |
| `S` | End the generation now and breed from what it scored |
| `R` | Start the whole run over: random weights, generation 1, records cleared |

They are listed in the legend at the bottom left, and the speed shows in the Training panel.

## Playing it yourself

`run-game.py` puts you on the same `World` the agents train on, drawn through the same renderer, so the
physics under your feet and the panels around them are the ones they are scored on. A run ends on the flag or
on a death, a banner says what it was worth, and the next attempt starts a second later.

- `--map PATH`: level file to play (default: maps/level_1.txt)
- `--tick-rate N`: simulation ticks per in-game second (default: 90)
- `--fps N`: target framerate (default: 0, uses the display refresh rate)
- `--spawn N`: spawn point to start on (default: 0, the start, the rest are the checkpoints)

| Key | What it does |
| --- | --- |
| `Arrows`, `WASD`, `ZQSD` | Move |
| `Space`, `Up`, `W` | Jump, aimed by the direction held on the same tick |
| `P` | Pause and resume |
| `Tab` | Hide the panels, the banner stays |
| `R` | Retry from the current spawn point |
| `G` | Start on the next spawn point, wrapping back to the beginning of the level |
| `1` | Back to speed x1 |
| `-` / `=` | Halve or double the simulation speed, down to x0.1 and up to x4, also on the numpad |
| `Escape` | Quit, printing what the session scored |

The Run panel tracks the session: time, best time, progress through the map, coins, attempts, wins and
deaths. The Player panel is the debug view: pixel position, tile, both speed components, ground contact,
spawn point and simulation speed. Slow motion is the useful one there, a jump arc lasts about 42 ticks and at
x0.1 it can be read frame by frame.

## The network

A policy over a 7x7 tile view plus the player's own state:

- **Input**: 58 features. One solid flag per tile, then the closest flag tile in view as `in view, dx, dy`,
  then the closest coin the same way, then horizontal speed, vertical speed and ground contact
- **Hidden**: 256, 128, 64 (LayerNorm on the first two, leaky ReLU)
- **Output**: 3 logits (jump, left, right), played as an argmax

Terrain gets a number per tile, the flag and the coins do not. The map holds 20 flag tiles and 217 coins out of
22,920, so a channel per tile for them would spend most of the observation saying "still nothing here", while
one offset to the closest one points at the flag directly rather than leaving the network to read a position
out of a one-hot grid. The coin block is baked with the terrain, so it still points at a coin the agent has
already banked.

Every agent shares that shape, so the population lives in one `(agents, in, out)` tensor per layer and a
forward pass for everyone is a single `baddbmm` per layer. Crossover, mutation and elitism are plain tensor
ops on those same weights. Weight files hold one agent in ordinary `nn.Linear` / `nn.LayerNorm` layout.

Agents are ranked by fitness, elites are copied over untouched, most of the population is a mutated crossover
of two random elites, and a few slots are re-randomised for diversity.

## Rewards

- **Forward**: +0.02 per step, **+0.1** on a new distance record
- **Backward**: -0.1, **stationary**: -0.05 after 5 ticks, **falling**: -0.02 past 5 pixels
- **Death**: -20, **progress**: max distance / 20, plus up to 100 for how early the record was set, floored at -30
- **Win**: distance / 10, +200 for the flag, plus up to 1200 on the square of the episode time left
- **Coins**: +5 each, added to whatever the run scored, so the 217 coins of the level are worth 1085 to an
  agent that could sweep them all, and a detour for one is always worth something

Time is part of the fitness on both paths: an agent that touches the flag halfway through the episode scores
300 of the 1200, one that touches it in the first tenth scores 970. An episode runs until every agent is dead
or has finished, so the winners of a generation are ranked against each other by the tick they arrived on.

Agents that stand still for 2 seconds, or end up behind where they were 6 seconds earlier, are killed so the
generation ends sooner.

## The level

`maps/level_1.txt` is 764 tiles wide and 30 tall, read one character per tile:

| Char | Tile |
| --- | --- |
| `#` | Solid terrain |
| `.` | Air |
| `P` | Spawn point |
| `@` | Checkpoint, an extra spawn point under `--checkpoints`, drawn as a violet banner |
| `o` | Coin, worth fitness and nothing else, drawn as a gold coin |
| `F` | The flag: touching it wins the episode |
| `*` | Decoration, no collision |

Geometry follows the jump: a jump rises 4.5 tiles and its arc covers 8, so steps stay within 3 tiles and gaps
within 6. Every pit has a launch step before it and a coin road above it, which pays more than the flat
crossing. Eleven zones run from tutorial steps to a comb of single pillars, a ceiling too low to jump under,
a serpentine of blocks and ceilings, and a wall that has to be climbed. The fastest route through it is about
44 seconds, against a 60 second episode.

The map is 30 rows tall against a 22.5 row screen, so the camera follows agents off the top of it. Two zones
are built around that height:

- **The tower** (tiles 290-409) zigzags from the floor to the top row of the map, runs a high road of narrow
  platforms across it, and comes back down. The ground under it is clear, so the fast line is to ignore it
  and the paying line is to climb.
- **The fork** (tiles 410-469) splits into a flat, empty low road and a gallery of 25 coins above it. The
  gallery's staircase climbs to the left before it climbs to the right, which is the move an agent that only
  ever holds right and jump never makes, so which of the two an agent takes is visible at a glance.

## Why the argmax

The policy always plays the argmax of its logits. Sampling from it instead makes fitness a lottery: the same
weights replayed a hundred times score anywhere from 40 to 730, mean 207, standard deviation 130. Selection
then picks whoever drew the luckiest samples, that agent regresses to its mean next generation, and the best
fitness saws up and down instead of climbing. The argmax has no variance: elites re-score exactly what earned
them their rank, and the best fitness becomes a monotonic staircase.

Nothing is lost by dropping the noise. Exploration here comes from mutating weights, not from mutating
actions, and there is no policy gradient to feed a sampled action back into. Only the ordering of the logits
is ever read, so their spread is never shaped into a distribution worth sampling from.

## Picking the mutation strength

Swept at 300 agents over 200 generations on five seeds each, on `maps/level_1.txt`, scored by the furthest
tile the population reaches out of the level's 764.

| `--mutation-strength` | mean tile | per seed |
| --- | --- | --- |
| 0.008 | 497 | 504, 514, 199, 501, 767 |
| 0.015 | 606 | 697, 628, 499, 697, 509 |
| **0.02** | **709** | **758, 697, 697, 697, 697** |
| 0.03 | 580 | 494, 504, 509, 697, 697 |

The per-seed column matters more than the mean: below 0.02 a seed can spend its whole run stuck on the tile
it plateaued at, because the population has converged onto one lineage and the noise is no longer wide enough
to find the jump it is missing. Weights start with a standard deviation near 0.04, so 0.03 is close to a full
sigma and leaves most children as damaged copies of their parent, which is what makes the elite run away
alone on screen.

Population size buys the same thing, and a run that plateaus is short of both: over 300 generations at 0.02,
100 agents reach tile 511 on average and never see the flag, 300 agents reach 718, and 1000 agents reach the
flag on two seeds out of five.

## Performance

On CUDA the whole tick runs on the device and a population of 300 plays about four million agent-steps a
second. The same tick in numpy, which is what a CPU run falls back to, is an order of magnitude slower at
that size. Measured on a 4060 Ti.

| Setup | Ticks/s | Agent-steps/s | Same tick in numpy |
| --- | --- | --- | --- |
| 100 agents | ~21,000 | ~2,100,000 | ~1,200 |
| 300 agents | ~13,600 | ~4,100,000 | ~1,160 |
| 600 agents | ~5,500 | ~3,300,000 | ~810 |
| 1000 agents | ~3,300 | ~3,300,000 | ~620 |
| 3000 agents | ~1,200 | ~3,600,000 | ~370 |

At 90 ticks per in-game second, 300 agents playing a 60 second episode take about 0.4 seconds of wall clock.
Agent-steps flatten out around four million because past a few hundred agents a tick is no longer physics at
all, it is the action pass reading every agent's weights.

**A tick never leaves the device.** Physics, the per-tick rewards and the next decision are one compiled,
captured graph replayed once per action window, so a window is a single launch and nothing crosses back to
the host inside it. Positions are truncated to whole pixels every tick, which keeps the device simulation
exactly equal to the numpy one rather than merely close: same collisions, same deaths, same coins, agent for
agent. What the host still reads, it reads between windows. The alive mask is sampled once an in-game second,
and the whole state is copied back only when a frame is drawn or a checkpoint ends.

**Fusing is what makes it fast, not the device.** Written as plain tensor calls a tick is 245 tiny kernels,
each paying a fixed cost whatever the population size, and it measures no faster than numpy at 300 agents.
Inductor fuses those into 14, which is the order of magnitude, and the graph capture removes what is left of
the launch cost. Compiling the action pass on top is worth about 3%, because that one moves real bytes, so
only the simulation is compiled.

**Physics in numpy is bound by call overhead, not by data.** A 300-element `np.add` costs 0.69 us against 0.61 us
for a one-element one, so nine tenths of a call is dispatch and the vector unit is idle waiting on Python.
Widening the arrays is free and narrowing them buys nothing: float32 measures the same as float64 at this size.
The only lever is making fewer calls, so the collision passes are written around that. The four tiles a player
box touches come out of one `(4, count)` block instead of four separate divisions, and the far side of that box
is one strided add off the near side. Both passes that run after the vertical snap read the same tiles, so the
box is resolved once and handed to them rather than rebuilt inside each. Grids are gathered with the `take`
method rather than `np.take`, which is the same gather without two frames of dispatch in front of it, and the
chain of `where` calls that snaps a blocked player collapses into one select per direction. The coin pass takes
its four corners in a single gather and drops out before its dedupe loop unless somebody is standing on a coin;
reward tiles are rare, so one lookup in a baked "any reward in these four tiles" map skips the reward gather on
almost every tick.

Per-tick rewards are written the same way: the stationary counter is stepped and cleared by a whole-array add
and multiply instead of two masked writes and an invert, and the terms accumulate into the running fitness
through one masked add rather than a mask multiply and a separate `+=`.

**Vision is baked.** The window of every tile in the map, terrain and reward vector both, is built once at
load time, so an observation is a single gather of one row. It is written in half precision straight into a
page-locked buffer the device copies from, so a tick crosses PCIe once each way with no staging copy in
between.

**The forward pass is one graph replay.** A pass is a few dozen tiny kernels, so it is bound by launch
latency: the action pass is captured as a CUDA graph, which is about 3x faster than launching its kernels one
by one, and the capture falls back to eager mode if the device cannot do it. The network
runs in half precision, which nearly halves the pass and costs nothing when only the argmax is read.

**The device works while the CPU does.** Waiting on the device is most of what a decision costs, so a pass is
started at the end of the tick before the one that plays it, and that tick's physics runs while the device
decides. The observation is taken at the same point either way, so a run is identical to a serial one, down
to the last float.

What is left is the pass itself, bound by reading every agent's weights: 300 agents of 57k half precision
parameters is 34 MB, near the memory bandwidth of the card. That is also why the observation is worth
keeping small, and why `--hidden-sizes` is the last lever on it.

**A frame is a handful of blits and two lists.** Behind everything sits a parallax backdrop: a sky gradient, a
star field and three mountain ranges, each one a viewport-sized surface baked at startup and scrolled at its own
fraction of the camera, from 0.05 for the stars to 0.36 for the nearest range. The scrolling layers tile
horizontally, so each is drawn twice side by side whatever the offset, and a range is clipped to the rows the one
in front of it does not already cover. The gradient is the one full-screen copy a frame makes, so it is clipped
the same way: below the back range's floor that range is opaque in every column, and the sky and the stars stop
there. With the camera low in the level that is most of the viewport.

The terrain and the goal never move, so the whole level is painted once into a single surface in the display's
own format, with the air left as a run-length encoded key colour. That makes the level blit skip the empty sky
instead of blending it, which is worth about a hundred times the cost of the same copy with a real alpha channel.
Over it go two batched `fblits` calls, one for the bodies and one for the markers. `fblits` gives up the source
rect and the per-blit flags that `blits` carries, which none of these need, and draws the same 300 bodies about
1.7x faster. Both lists are filled only with what the camera actually covers: the players are tested against the screen as one numpy mask, and the coins are held sorted by x
so two binary searches cut the map down to the column in view. Coins and checkpoints are the two things drawn per
frame rather than baked, the coins because which are left depends on the agent being followed, the checkpoints
because their haze is translucent and a key colour cannot carry that.

**Nothing is painted twice.** Every sprite is a character grid blown up with nearest-neighbour scaling, cached
under the values that shaped it, and there are few enough of those to cache the lot: sixteen terrain blocks
per speckle layout, one sprite per fitness bucket, heading, state and fade. A frame never draws a shape, it only
copies surfaces.

## Layout

`ai/` holds the learning code: `generation.py` is the training loop and weight files, `population.py` is the
batched network, `device_runner.py` plays whole action windows on the device, `rewards.py` is what a run pays
out. `game/` holds the engine: `world.py` is the batched numpy simulation and `world_cuda.py` the same
physics as tensors, `render.py` the camera-culled renderer and the panels drawn over it, `art.py` every
sprite it draws with, `play.py` the human-played session on top of both. Both entry points fill the same `Hud` and the renderer only lays it out.
Levels are text files in `maps/`, `tiles.py` maps their characters to a `TileKind`, weights land in `weights/`,
screen and physics constants live in `game/settings.py`.

## License

GNU General Public License v3.0, see [LICENSE](LICENSE).

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
```

Python 3.13+, pygame-ce for the window, and `uv sync` pulls torch with CUDA 13 on Windows and Linux. CPU
works too, it is just slower.

## Options

*evolution* - how a generation is selected and bred

- `--population-size N`: agents per generation (default: 300). A tick at 300 agents costs about 1.4x a tick
  at 100 while trying three times as many mutations, and those extra mutations are what break a plateau
- `--elite-count N`: agents carried over untouched and used as parents (default: 4)
- `--mutation-rate R`: probability that a child's weight tensor is mutated at all (default: 0.8)
- `--mutation-strength S`: scale of the noise added to a mutated tensor (default: 0.008)

*network* - shape and placement of the policy

- `--hidden-sizes N1 N2 N3`: sizes of the three hidden layers (default: 256 128 64), smaller is faster and dumber
- `--device auto|cpu|cuda`: where the population runs (default: auto)
- `--threads N`: torch CPU threads (default: 4)

*simulation* - the level and the episode played on it

- `--map PATH`: level file to train on (default: maps/level_1.txt)
- `--episode-seconds S`: in-game time budget per spawn point (default: 30)
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

- **Fill color**: fitness rank, red (worst) to teal (best). **Grey**: dead. **Violet fill**: reached the flag
- **Chevron**: heading, a square when standing still. **Blue arrow above the head**: rising, so it jumped
- **Gold outline**: an elite carried over untouched. **Violet outline**: a re-randomised agent, kept for diversity
- **Dark outline**: the agent the camera follows, the best one still alive

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

## The network

A policy over a 7x7 tile view plus the player's own state:

- **Input**: 56 features. One solid flag per tile, then the closest reward tile in view as `in view, dx, dy,
  is the flag`, then horizontal speed, vertical speed and ground contact
- **Hidden**: 256, 128, 64 (LayerNorm on the first two, leaky ReLU)
- **Output**: 3 logits (jump, left, right), played as an argmax

Terrain gets a number per tile, rewards do not. The map holds 7 reward tiles out of 9,913, so a channel per
tile for them would spend most of the observation saying "still nothing here", while one offset to the
closest one points at the flag directly rather than leaving the network to read a position out of a one-hot
grid.

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

Time is part of the fitness on both paths: an agent that touches the flag halfway through the episode scores
300 of the 1200, one that touches it in the first tenth scores 970. An episode runs until every agent is dead
or has finished, so the winners of a generation are ranked against each other by the tick they arrived on.

Agents that stand still for 2 seconds, or end up behind where they were 6 seconds earlier, are killed so the
generation ends sooner.

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

Swept at 300 agents over 40 generations on five seeds each, counting how often the run reaches the flag and
how far into the run it gets there. `pack` is the population's mean fitness over the last five generations,
which says how closely the rest follows its elites.

| `--mutation-strength` | reached the flag | generation it took | pack |
| --- | --- | --- | --- |
| 0.002 | 2/5 | 6, 31 | 384 |
| 0.004 | 3/5 | 14, 14, 18 | 345 |
| **0.008** | **5/5** | **3, 8, 9, 12, 16** | 224 |
| 0.015 | 4/5 | 7, 11, 28, 34 | 129 |
| 0.03 | 2/3 | 23, 36 | 86 |
| 0.06 | 2/3 | 30, 39 | 67 |

Weights start with a standard deviation near 0.04, so 0.03 is close to a full sigma and leaves most children
as damaged copies of their parent. That is what makes the elite run away alone on screen. Too small and the
population never finds the jump it is missing.

## Performance

A tick grows far slower than the population it simulates: ten times the agents cost about two and a half
times the tick. Measured on a 4060 Ti.

| Setup | Ticks/s | Agent-steps/s |
| --- | --- | --- |
| 100 agents, CUDA | ~2,450 | ~245,000 |
| 300 agents, CUDA | ~1,800 | ~540,000 |
| 600 agents, CUDA | ~1,330 | ~800,000 |
| 1000 agents, CUDA | ~1,000 | ~1,000,000 |

At 90 ticks per in-game second, 100 agents playing a 30 second episode take about 1.1 seconds of wall clock.

**Physics is bound by numpy call overhead, not by data.** On 300-element arrays a numpy call costs far more
than the arithmetic inside it, so the four collision passes are written to make as few calls as possible. The
four tiles a player box touches come out of one `(4, count)` block instead of four separate divisions,
collisions read a flat grid through `take` rather than a broadcast fancy index, and the chain of `where`
calls that snaps a blocked player collapses into one select per direction. Reward tiles are rare, so one
lookup in a baked "any reward in these four tiles" map skips the reward gather on almost every tick.

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

## Layout

`ai/` holds the learning code: `generation.py` is the training loop, rewards and weight files, `population.py`
is the batched network. `game/` holds the engine: `world.py` is the batched numpy simulation used for
training, `render.py` the camera-culled renderer, and the rest is the manual game. Levels are text files in
`maps/`, weights land in `weights/`, physics constants live in `game/settings.py`.

## License

GNU General Public License v3.0, see [LICENSE](LICENSE).

# Neural-Jump

A platformer where a thousand neural networks play the level at once, and two ways of getting better at it:

- **PPO** (default) trains one policy on every environment in parallel, with a reverse curriculum that starts
  episodes near the flag and walks them back to the level's own spawn point as the policy learns.
- **Evolution** (`--trainer ga`) runs a population of separate networks, scores them, breeds the best few and
  mutates their children. No gradients anywhere. It plateaus part way through the level.

The interesting part is that everything is batched. The world is numpy arrays with one slot per agent, the
networks are one set of `(agents, in, out)` weight tensors, and a tick steps all of them together. On a
4060 Ti that is seven million agent-steps per second.

## Quick start

```bash
uv sync
uv run run-ai.py                              # headless PPO, as fast as the machine allows
uv run run-ai.py --show-window --speed max    # watch it, as fast as the framerate survives
uv run run-ai.py --trainer ga                 # the genetic algorithm instead
uv run run-game.py                            # play the level yourself
uv run run-game.py --spawn 3                  # start on the third checkpoint
```

Python 3.13+, pygame-ce for the window, and `uv sync` pulls torch with CUDA 13 on Windows and Linux, plus
the triton wheel torch does not ship on Windows, which is what compiles the simulation. CPU works too, it is
just slower: the whole tick then runs in numpy on the host instead.

The first generation of a run is slower than the rest, because that is where the simulation is compiled.

## Options

- `--trainer ppo|ga`: `ppo` trains one policy on every environment at once, `ga` selects and breeds a
  population of separate networks (default: ppo)
- `--population-size N`: parallel environments under PPO, agents per generation under evolution. Defaults to
  1024 and 300, because one shared policy makes the action pass cost almost nothing per environment

*ppo* - the policy gradient, ignored under `--trainer ga`

- `--rollout-steps N`: decisions per environment between two updates (default: 256), so the default update
  sees 262,144 transitions
- `--learning-rate R` (default: 3e-4), `--gamma G` (default: 0.999, per decision, not per tick)
- `--gae-lambda L`: bias against variance in the advantage estimate (default: 0.95)
- `--clip-range C`: how far one update may move the policy (default: 0.2)
- `--epochs N`: passes over each rollout (default: 4), `--minibatches N`: what each pass is split into (default: 8)
- `--target-kl K`: divergence from the rollout that stops the extra passes early (default: 0.02)
- `--entropy-coef E`: exploration bonus (default: 0.01), annealed to 0.001 over the first 400 updates
- `--spawn curriculum|uniform|start`: where an episode restarts (default: curriculum)
- `--spawn-spacing T`: tiles between two rungs of the curriculum ladder (default: 20), which is built from
  the level floor rather than from the hand-placed checkpoints

*evolution* - how a generation is selected and bred, ignored under `--trainer ppo`

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
uv run run-ai.py --spawn start                                    # PPO on the whole level, no curriculum
uv run run-ai.py --population-size 4096 --rollout-steps 128       # wider rollout, same transitions per update
uv run run-ai.py --trainer ga --population-size 1000 --hidden-sizes 64 32 16   # a big dumb crowd
uv run run-ai.py --seed 1                                         # replays exactly
uv run run-ai.py --show-window --speed 4                          # watchable
```

## Reading the window

Every agent's state is in its sprite, so one glance reads the whole population:

- **Body color**: fitness rank, red (worst) to teal (best). **Blue body**: reached the flag
- **Faded grey ghost with shut eyes**: dead, dimmed out of the way of the agents still running, and carrying no outline
- **Heading**: the face slides forward, both pupils sit against the front of their whites and the light moves to the
  leading edge; standing still, the face is centred and symmetric. **Blue arrow above the head**: rising, so it jumped
- **Gold outline**: an elite carried over untouched. **Violet outline**: a re-randomised agent, kept for
  diversity. Neither appears under PPO, where every environment plays the same policy
- **White outline**: the agent the camera follows, the best one still alive

The panels cover the run, the followed agent, the hyper-parameters, and the fitness distribution next to the
best score of every generation so far.

Under PPO the Run panel counts rollouts rather than generations, and its time row is how full the current one
is. There is no reset between two of them: an environment carries its episode across the update, and the ones
that restart do so on their own, whenever they die or reach the flag. The Training panel shows which rung the
curriculum is on, which is near the flag at the start of a run and not at the level's own spawn point. To
watch a run from the beginning of the level instead, use `--spawn start`.

Between two rollouts the row reads **LEARNING** and the level stops moving, because it is: an update over a
quarter of a million transitions takes about two seconds, and the simulation is not being stepped through any
of it. The window stays live, it just has nothing new to draw. What that costs to watch depends on `--speed`,
since it is a fixed two seconds against however long the rollout in front of it took.

## Keys

| Key | What it does |
| --- | --- |
| `Space` | Pause and resume, the window stays live |
| `Tab` | Hide the panels, leaving the level and the agents |
| `1` | Back to speed x1 |
| `F` | Speed `max`, which tunes itself to the framerate |
| `M` | Open the map list: `Up` / `Down` to pick, `Enter` to load it under the run, `M` to close |
| `-` / `=` | Halve or double the speed, also on the numpad. From `max` it starts at the multiplier it had reached |
| `G` | Skip to the next spawn point, or under PPO move the curriculum on a rung by hand |
| `S` | End the generation now and breed from what it scored, or under PPO update on the rollout so far |
| `R` | Start the whole run over: random weights, generation 1, records cleared |
| `V` | Free the camera from the followed agent, or give it back |
| `Page Up` / `Page Down` | Zoom, x0.2 out to x4 in |
| `0` | Back to x1 on the followed agent, also on the numpad |
| Drag, wheel | Pan the camera, and zoom around the cursor |

They are listed in the legend at the bottom left, and the speed shows in the Training panel.

## Moving the camera

The camera follows the best agent still alive, and `V` takes it off it. A free camera is dragged with any mouse
button and zoomed with the wheel, which holds the world pixel under the cursor in place; dragging frees the
camera on its own, so a drag is all it takes. `0` puts it back on its agent at x1. The Focus panel says which of
the two it is and what the zoom is, and the same keys work in `run-game.py`, where the camera follows you.

Zoom out reads the whole spread of a population, which is what a rollout looks like in one glance, and zoom in
reads a single jump arc against the tile it lands on. Out is the one that costs: at x0.2 the camera covers
8000x4500 world pixels to fill a 1600x900 window. Only the band the level actually spans is drawn on and scaled,
the backdrop keeping the rest, which is what makes the far end of the range affordable at all: over 300 agents on
`level_1` a frame is 5.3 ms at x1 and 8.7 ms at x0.2, against 21 ms with the whole camera drawn. At x1 nothing is
scaled and the frame is exactly what it has always been.

## Switching maps mid-run

`M` opens the map list over a held simulation. Picking a level drops whatever rollout or generation is in
flight, since it was played on the old map, and the swap happens between two of them rather than inside one.

What is rebuilt is everything the map is baked into: both worlds, the runner holding the compiled window and its
captured graph, and the curriculum ladder read off the new floor. That is a compile and a capture, so the window
sits still for a few seconds. Every record and tracker starts over too, because a score on one level says
nothing about another.

What carries over is the policy and its optimiser, which is the reason to switch at all: the observation is the
same 62 features whatever the map, so a network that has learned to run and jump on one level starts the next
one already knowing how.

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
| `M` | Open the map list over a frozen run: `Up` / `Down` to pick, `Enter` to load, `M` or `Escape` to close |
| `V`, drag, wheel | Free the camera, pan it, zoom it: see [Moving the camera](#moving-the-camera) |
| `0` | Back to x1 on yourself |
| `Escape` | Quit, printing what the session scored |

The Run panel tracks the session: time, best time, progress through the map, coins, attempts, wins and
deaths. The Player panel is the debug view: pixel position, tile, both speed components, ground contact,
spawn point and simulation speed. Slow motion is the useful one there, a jump arc lasts about 42 ticks and at
x0.1 it can be read frame by frame.

The map list holds every `.txt` under `maps/`, read again each time it opens, so a level imported while the game
runs shows up, and loading one starts the session's records over. `run-ai.py` has the same list under the same
key, where a swap costs rather more: see [Switching maps mid-run](#switching-maps-mid-run).

## The network

A policy over a 7x7 tile view plus the player's own state:

- **Input**: 62 features. One solid flag per tile, then the closest flag tile in view as `in view, dx, dy`,
  then the closest coin the same way, then the closest enemy as `in view, dx, dy, heading`, then horizontal
  speed, vertical speed and ground contact. A map without enemies leaves those four at zero, and a weight file
  saved against another input size is refused, the run starting from random weights instead
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

Under evolution every agent owns a copy of that shape, so the population lives in one `(agents, in, out)`
tensor per layer. Agents are ranked by fitness, elites are copied over untouched, most of the population is a
mutated crossover of two random elites, and a few slots are re-randomised for diversity.

Under PPO the same tensors hold a single network with a value head bolted onto the trunk, asked about one
observation per environment instead of `agents` networks asked about one each. It is the same `baddbmm` with
the two dimensions swapped, and it reads 1024 times fewer weight bytes per decision.

## PPO

A rollout is `--rollout-steps` decisions on every environment, each one an action window of `--action-repeat`
ticks. Environments run their own episodes: one that dies or reaches the flag restarts on the next step while
the rest carry on with theirs, so nothing waits for anybody and a rollout is always exactly as long as it says.
Then GAE(lambda) over the buffer, and `--epochs` passes of clipped minibatch ascent, stopped early when the
policy has moved `--target-kl` away from the data that produced it.

The whole rollout is one graph replay per step, buffer writes included: the write cursor is a device tensor,
episode ends and resets are whole-population `where` calls, and the spawn point of a restart is drawn on the
device from a cumulative table. The host reads nothing between the start of a rollout and its end.

**The curriculum is what gets through the level.** The fast route is 44 seconds long, and a policy dropped at
the start ends every one of its first thousand episodes in the same opening seconds, so nothing past them is
ever seen. So episodes start near the flag instead and work backwards, one rung at a time.

The ladder is built from the level's geometry, not from its four hand-placed checkpoints. A column's landing
is its lowest solid tile with two clear rows over it, which is the main floor where there is one and the
platform bridging a pit where there is not. A rung is taken every `--spawn-spacing` tiles, and also on the
near edge of every surface, which is what puts one on each stepping stone of a crossing. That is 58 rungs on
`maps/level_1.txt` against 4 checkpoints whose last one is still 169 tiles from the flag.

How the ladder is built is the whole thing working or not, and both refinements were paid for by a stall:

- On the **checkpoints**, `--seed 1` spent 150 updates on the first rung without a single win, its entropy
  collapsing the whole way, while other seeds got through in 9. On a built ladder the same seed wins in its
  first rollout.
- On the **main floor alone**, the ladder has 76-tile holes where the level runs over pits, and the front
  stalls on the first one.
- Every `--spawn-spacing` tiles and nothing else, the front descended 30 rungs and then sat for 280 updates
  on tile 147, which is the near lip of a 13-tile pit crossed by two stepping platforms. The rung straddled
  the entire crossing. **Taking the edges too** puts a rung on each platform, so the ladder breaks that jump
  into the three it is made of.

The entropy bonus anneals against updates spent on the current rung rather than updates in the run, for the
same reason: a curriculum can sit on one stretch for hundreds of updates, and a bonus annealed against the
run is at its floor by the time the hard rungs come up. A rung that clears nothing at all for 25 rollouts is
called stuck, and doubles what the bonus restarts on, up to a ceiling of 0.08.

That is aimed at one failure in particular. The hardest jump on `maps/level_1.txt` is a 6-tile gap taken two
rows uphill off a 4-tile platform, at tile 639: one random agent in 16,384 lands it, against about 50 in
16,384 for the rungs either side. A policy that has already settled explores a good deal less than a random
one, so the way past a rung like that is to put the noise back rather than to wait. Over a run that clears
all 57 rungs it fires exactly once, on that one.

The front moves one rung back each time 60% of the episodes started there reach the rung in front of it.
Reaching the *next rung* rather than the flag is what keeps every promotion asking for the same thing, one
stretch of level; asking for a full run instead makes each promotion harder than the last and stalls the
ladder somewhere in the middle of the map. Rungs already behind the front keep 40% of the episodes, which is
what stops the earlier ones being forgotten.

`--spawn uniform` draws any rung with equal probability and `--spawn start` always uses the level's own spawn
point, which is the honest baseline the curriculum is measured against.

**What it comes to.** `--seed 1` at the defaults, 1024 environments and 262,144 transitions an update: the
front leaves the last rung on update 3, reaches the level's own spawn point on update 429, and by update 457
finishes the level in 156 of 160 episodes, its fastest full run 48.49 seconds against a 44 second route and a
60 second budget. That is about two hours on a 4060 Ti. The genetic path on the same level plateaus around
tile 700 of 764 and reaches the flag on two seeds out of five at a thousand agents.

**Rewards are paid where the decision is.** Evolution only needs one number per agent to rank on, so it pays
distance, coins and the win bonus at the end of the episode. A policy gradient needs to know which decision
earned what, so under PPO the coins are paid on the tick they are taken and the end of episode reward is the
flag bonus or the death penalty and nothing else. Distance is already paid tick by tick as it is covered;
paying it again at the end would put most of the return in one terminal spike.

What is left of the scale problem is handled by dividing every reward by the running spread of its own
discounted return, which is what keeps a 1400 point flag and a 0.02 point step in the same critic.

## Rewards

- **Forward**: +0.02 per step, **+0.1** on a new distance record
- **Backward**: -0.1, **stationary**: -0.05 after 5 ticks, **falling**: -0.02 past 5 pixels
- **Death**: -20, **progress**: max distance / 20, plus up to 100 for how early the record was set, floored at -30
- **Win**: distance / 10, +200 for the flag, plus up to 1200 on the square of the episode time left
- **Coins**: +5 each, added to whatever the run scored, so the 217 coins of the level are worth 1085 to an
  agent that could sweep them all, and a detour for one is always worth something

Time is part of the score: an agent that touches the flag halfway through the episode scores 300 of the 1200,
one that touches it in the first tenth scores 970.

Under evolution an episode runs until every agent is dead or has finished, so the winners of a generation are
ranked against each other by the tick they arrived on. Under PPO the progress and coin lines move: coins are
paid the tick they are taken, and the end of an episode pays the flag bonus or the death penalty alone, since
distance has already been paid as it was covered.

Agents that stand still for 2 seconds, or end up behind where they were 6 seconds earlier, are killed. Under
PPO an environment whose episode is younger than the window is exempt, since it is behind where it was only
because it restarted.

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
| `E` | An enemy's starting cell, air once it walks off |
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

## Enemies

`maps/level_1.txt` has none, the Super Mario Bros levels below have plenty. An enemy walks toward the start of
the level at 2 pixels a tick, a quarter of the player's speed, turns around on a wall and falls off ledges. It
wakes only once a player comes within 10 tiles, the way an SMB enemy starts moving as it scrolls onto the
screen, so every agent meets it in the same state whenever it gets there. Landing on one from above, feet over
its middle on the tick before, stomps it and bounces the player back up; any other contact kills.

An enemy never reacts to a player, so where it stands only depends on how long it has walked. Every path is
baked once at load: walked tick by tick until the enemy dies or repeats a state, then its loop is tiled over
8192 ticks. Stopping on the repeat is what makes that cheap, since an enemy pacing between two pipes never dies:
walking all 8192 ticks of every path costs about 2 seconds a map, tiling the loops 10 to 160 ms. An agent's whole enemy state is then a step counter and a stomped bit per enemy, two `(agents,
enemies)` arrays like the coins, and both simulation paths read the same tables. Each agent wakes and stomps its
own copy of every enemy, and the window draws the copies of the agent it follows. On a map without enemies the
pass is skipped outright.

## Super Mario Bros levels

`uv run import-smb.py` downloads the 15 Super Mario Bros levels of the
[Video Game Level Corpus](https://github.com/TheVGLC/TheVGLC) into `maps/smb/`, and `uv run import-smb.py 1-1 4-2`
only those. The corpus stores a level as one character per tile, so importing one is a translation: ground,
bricks, question blocks, pipes and cannons become `#`, enemies `E`, coins `o`. It marks neither Mario nor the
flag, so the spawn goes on the floor at column 3 and a flag gate fills the air over the last column with a floor.
Its floor is one row thick where SMB's is two, and a player dies two rows above the bottom of a map, so the
bottom row is doubled. Every enemy kind is the same walker and cannons never fire. The levels are Nintendo's,
which is why they are downloaded on demand and ignored by git.

Train or play one with `--map maps/smb/1-1.txt`. They are 15 rows tall and 149 to 373 tiles wide, a fifth of
`maps/level_1.txt`, so the default 60 second episode leaves plenty of room.

## Why evolution plays the argmax

Under evolution the policy always plays the argmax of its logits. Sampling from it instead makes fitness a lottery: the same
weights replayed a hundred times score anywhere from 40 to 730, mean 207, standard deviation 130. Selection
then picks whoever drew the luckiest samples, that agent regresses to its mean next generation, and the best
fitness saws up and down instead of climbing. The argmax has no variance: elites re-score exactly what earned
them their rank, and the best fitness becomes a monotonic staircase.

Nothing is lost by dropping the noise. Exploration there comes from mutating weights, not from mutating
actions, and there is no policy gradient to feed a sampled action back into. Only the ordering of the logits
is ever read, so their spread is never shaped into a distribution worth sampling from.

PPO is the other case exactly: the ratio it clips is defined against the probability of the action that was
played, so the rollout has to sample, and the entropy bonus is there to keep that distribution wide early on.
Fitness being noisy costs nothing, because nothing is selected on it.

## Picking the mutation strength

Evolution only. Swept at 300 agents over 200 generations on five seeds each, on `maps/level_1.txt`, scored by the furthest
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

On CUDA the whole tick runs on the device. Measured on a 4060 Ti, on `maps/level_1.txt`.

| Trainer | Setup | Ticks/s collecting | Agent-steps/s | Ticks/s counting the update |
| --- | --- | --- | --- | --- |
| PPO | 1000 environments | ~7,000 | ~7,000,000 | ~1,700 |
| Evolution | 300 agents | ~13,600 | ~4,100,000 | ~13,600 |

PPO steps fewer ticks a second than evolution at three times the environments and still moves more
agent-steps, because its action pass reads one network instead of a thousand. What it spends the difference on
is the rest of a rollout step: the value head, sampling, the buffer writes and the resets.

**The update is three quarters of the wall clock**, and that is what PPO is. A rollout of 256 decisions on
1000 environments takes 85 ms to collect and 250 ms to learn from, because learning walks the same 256,000
transitions four more times, forwards and backwards. Evolution has no such phase: it never learns from what it
played, it only ranks it. The `Speed:` line reports both rates so the gap is visible rather than surprising.

Getting that 250 ms down was mostly not about the gradient. GAE and the reward scaler are sequential scans
over the rollout, and written a step at a time they were 127 ms of pure kernel launches for arithmetic on
1000 floats: everything in them that does not depend on the step before it is now lifted out into one
whole-rollout call each, which is 40 ms. Compiling the policy's evaluation is worth another third of a
minibatch, since the trunk is bound by passes over its activations rather than by its matmuls. Half precision
buys nothing here and neither does TF32; both are the wrong lever on something already at bandwidth.

**Watching costs almost nothing.** The window draws exactly one frame per update, not one per minibatch, and
that frame does not read the device state back. Every one of those reads would otherwise block on the
gradient kernels already queued in front of it, which serialises the entire update behind the window. At
`--speed max` the update still dominates, which is the honest picture: there is nothing to watch at that
speed anyway. Watch at `--speed 2` or `--speed 4`, where a rollout takes seconds and the update is a fifth of
the cycle, or trade sample efficiency for smoothness with `--epochs 2`.

The table below is the genetic path alone, where the action pass is the whole cost. The same tick in numpy,
which is what a CPU run falls back to, is an order of magnitude slower at that size.

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

Zoom is the one thing that surface cannot be the screen for: the world goes on a scratch surface instead, which
is then scaled to the window in one nearest-neighbour pass. Nearest is what keeps it cheap and also what keeps it
correct, since the key colour the air is left as has to survive the resize exactly. That scratch is not the
camera but the part of it the level covers, because zoomed out past the map the camera is mostly rows the level
does not have: clearing and scaling those is three quarters of a x0.2 frame and none of it can ever be drawn on.
Its size depends only on the zoom and the map, never on where the camera sits, so panning never reallocates it.
The backdrop stays at screen size behind it, both because it is a parallax lie already and because it keeps a x1
frame, where the scratch is the screen itself, exactly the handful of blits it was.

A key colour is all or nothing, which is what decides where the two translucent things in the frame are drawn. A
checkpoint's haze and a dead agent's fade blend with whatever is under them, and under them on that scratch is
the key colour, not the sky: the haze comes out magenta instead of violet and a ghost comes out bright pink,
neither of them a key pixel any more, so nothing downstream can take them back out. Both are drawn on the screen
once the band is down, sized by the zoom, where what they blend with is the real frame. Everything else is opaque
or fully transparent per pixel, which is what a key handles, so it stays in the band.

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
Levels are text files in `maps/`, `import-smb.py` fetches the Super Mario Bros ones into `maps/smb/`, `tiles.py`
maps their characters to a `TileKind`, weights land in `weights/`,
screen and physics constants live in `game/settings.py`.

## License

GNU General Public License v3.0, see [LICENSE](LICENSE).

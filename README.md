# Neural-Jump

A platformer game where AI agents learn to play through neuroevolution and genetic algorithms.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [License](#license)

## Introduction

Neural-Jump is an interactive platformer game featuring AI agents that learn to navigate levels through evolutionary algorithms. Agents control a player character that must jump over obstacles and reach the end of each level while maximizing their score. The AI uses deep neural networks trained via neuroevolution—a genetic algorithm that evolves the best-performing agents across generations.

## Features

- **Platformer Gameplay**: Classic side-scrolling platformer mechanics with jumping, movement, and collision detection
- **Neuroevolution**: Agents improve by genetic selection between generations, elites carried over untouched
- **Deterministic evaluation**: Agents play their argmax, so a generation's scores are repeatable and selection measures skill rather than luck
- **Batched Simulation**: The whole population is simulated as numpy arrays, so hundreds of agents run in parallel far faster than real time
- **Batched Networks**: The population is one set of `(agents, in, out)` weight tensors evaluated in a single batched matmul per layer
- **Reward Shaping**: Sophisticated reward system that encourages forward progress, penalizes backward movement, and rewards level completion
- **CUDA Support**: Automatic GPU acceleration when available
- **Persistent Training**: Save and load agent weights across generations
- **Manual Play**: Optional manual player mode to understand level design
- **Customizable Parameters**: Adjustable population size, network size, mutation rates, and game settings
- **Level Design**: Text-based level files for easy customization

## Requirements

- Python >= 3.13
- PyTorch 2.13 with CUDA 13.0 support (for GPU acceleration)
- pygame-ce for rendering
- NumPy

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ayfri/Neural-Jump.git
cd Neural-Jump
```

### 2. Install dependencies using uv

[uv](https://github.com/astral-sh/uv) is a fast Python package installer. Install the project dependencies:

```bash
uv sync
```

> **Note**: The first time you run this, uv will install PyTorch with CUDA 13.0 support (on Windows and Linux). This may take a few minutes.

### 3. (Optional) Activate the virtual environment

To manually activate the virtual environment created by uv:

```bash
# On Windows
.venv\Scripts\Activate.ps1

# On macOS/Linux
source .venv/bin/activate
```

## Usage

### Train AI Agents

Run the AI training script to watch agents learn to play the game:

```bash
uv run run-ai.py
```

Training is headless by default and runs as fast as the machine allows. `--show-window` paces it with `--speed` so the run stays watchable, and draws a HUD that reads the population's state straight off the screen.

**Command-line options:**

- `--population-size N`: Number of agents per generation (default: 300). A tick costs almost the same at 300
  agents as at 100, because the simulation is bound by numpy call overhead rather than by the data, and the
  extra mutations per generation are what break a plateau
- `--elite-count N`: Agents carried over untouched and used as parents (default: 4)
- `--mutation-rate R`: Probability of mutating a given weight tensor (default: 0.8, range: 0.0-1.0)
- `--mutation-strength S`: Scale of mutations (default: 0.03)
- `--hidden-sizes N N N`: Sizes of the three shared hidden layers (default: 256 128 64), smaller is faster and dumber
- `--sampled`: Sample actions from the policy instead of playing its argmax, which makes a generation's scores a lottery
- `--seed N`: Seed python, numpy and torch so a run replays exactly
- `--action-repeat N`: Physics ticks a chosen action is held for (default: 2)
- `--device auto|cpu|cuda`: Where the population runs (default: auto)
- `--threads N`: Torch CPU threads (default: 4)
- `--tick-rate N`: Simulation ticks per in-game second (default: 90)
- `--episode-seconds S`: In-game time budget per spawn point (default: 30)
- `--generations N`: Stop after N generations (default: 0, runs forever)
- `--speed S`: Simulation speed multiplier while rendering, or `max` to run as fast as the target framerate survives (default: 1)
- `--fps N`: Target framerate, only with `--show-window` (default: 0, uses the display refresh rate)
- `--map PATH`: Level file to train on (default: maps/level_1.txt)
- `--load-latest-generation-weights`: Load weights from the latest saved generation
- `--show-window`: Display the game window during training
- `--checkpoints`: Use checkpoint platforms as spawn points

**Examples:**

```bash
# Fast headless training
uv run run-ai.py --population-size 500

# Smaller and dumber network, even faster
uv run run-ai.py --population-size 1000 --hidden-sizes 64 32 16

# Watch a run at 4x speed
uv run run-ai.py --show-window --speed 4

# Watch it go as fast as the display can keep up with
uv run run-ai.py --show-window --speed max
```

### Reading The Window

The renderer encodes each agent's state in its sprite, so a glance at the screen is enough to tell the
population apart:

- **Fill color**: the agent's fitness rank in the population, from red (worst) to teal (best)
- **Chevron**: the direction it is moving, a square when it is standing still
- **Blue arrow above the head**: the agent is rising, so it jumped
- **Gold outline**: an elite carried over untouched from the previous generation
- **Violet outline**: a re-randomised agent, kept for diversity
- **Dark outline**: the agent the camera follows, the best one still alive
- **Grey**: dead, **violet fill**: reached the flag

The panels cover the run (generation, time, living agents, best fitness, throughput and framerate), the
followed agent, the hyper-parameters the run uses, and the fitness distribution of the population next to
the best score of every generation so far.

### Performance

The simulation and the population are both batched, so the cost of a tick is nearly flat in the number of
agents: going from 100 to 1000 agents multiplies the throughput per second, not the wall clock.

| Setup | Ticks/s | Agent-steps/s |
| --- | --- | --- |
| 100 agents, CUDA | ~1,380 | ~138,000 |
| 300 agents, CUDA | ~1,000 | ~300,000 |
| 600 agents, CUDA | ~730 | ~438,000 |

Two things carry that number. The 7x7x4 vision window of every tile is baked once at load time, so an
observation is a single gather instead of a broadcast fancy index rebuilt per tick. And `--action-repeat`
holds each decision for two physics ticks, which halves the network calls.

At 90 ticks per in-game second, 100 agents playing a 30 second episode take about 2.5 seconds of wall clock.

Observations are half precision, so the host-to-device transfer every tick moves half the bytes.

On CUDA both action passes, the argmax and the sampled one, are captured as CUDA graphs: a tick is a few
dozen tiny kernels, so it is bound by launch latency, and replaying one captured graph is about 3x faster
than launching them one by one. The capture falls back to eager mode on its own if the device does not
support it. The network itself runs in half precision on CUDA, which nearly halves the forward pass and
costs nothing when only the argmax of the logits is read. Weight files are widened back to float32.

### Play Manually

To play the game yourself:

```bash
uv run run-game.py
```

## Project Structure

```
Neural-Jump/
├── ai/                    # Learning code
│   ├── generation.py     # Training loop, rewards and weight files
│   ├── population.py     # The whole population as one batched policy network
│   └── __init__.py
├── game/                  # Game engine and mechanics
│   ├── world.py          # Batched numpy simulation used for training
│   ├── render.py         # Camera-culled renderer for the batched world
│   ├── game.py           # Main game loop (manual play)
│   ├── player.py         # Player character logic (manual play)
│   ├── level.py          # Level management (manual play)
│   ├── tiles.py          # Tile and collision system
│   ├── platform.py       # Platform objects
│   ├── constants.py      # Game constants
│   ├── settings.py       # Game configuration
│   ├── main.py           # Game entry point
│   └── __init__.py
├── maps/                  # Level definitions
│   └── level_1.txt       # First level layout
├── weights/              # Saved neural network weights
│   └── generation_*.pth  # Weights for each generation
├── pyproject.toml        # Project configuration and dependencies
├── run-ai.py             # AI training script
├── run-game.py           # Manual gameplay script
├── LICENSE               # GNU General Public License v3.0
└── README.md             # This file
```

## Configuration

### Neural Network Architecture

The agent's network is a policy over a 7×7 grid view plus the player's own state:

- **Input**: 199 features, 4 channels per tile (solid, flag, reward, empty) plus horizontal speed, vertical speed and ground contact
- **Hidden Layers**: 256, 128 then 64 neurons (LayerNorm on the first two, leaky ReLU), configurable with `--hidden-sizes`
- **Output**: 3 action logits (jump, move left, move right), played as an argmax

Every agent shares this shape, so the population is stored as one `(agents, in, out)` tensor per layer and a
forward pass for all agents is a single `baddbmm` per layer. The genetic operators are plain tensor ops on
those same weights. Weight files hold one agent in plain `nn.Linear` / `nn.LayerNorm` layout
(`fc1.weight`, `norm1.bias`, `actor.weight`, ...).

### How a generation evolves

Agents are ranked by fitness, the elites are copied over untouched, most of the population is a mutated
crossover of two random elites, and a few slots are re-randomised for diversity. A weight tensor of a child
is mutated with probability `--mutation-rate`, by gaussian noise scaled by `--mutation-strength`.

### Why the action selection is an argmax

Sampling from the policy makes an agent's measured fitness a lottery: the same weights replayed a hundred
times score anywhere from 40 to 730, mean 207, standard deviation 130. Selection then picks whichever agent
drew the luckiest samples, and next generation that agent regresses to its mean, so the best fitness saws up
and down instead of climbing. Playing the argmax removes the variance entirely, elites re-score exactly what
earned them their rank, and the best fitness becomes a monotonic staircase. `--sampled` restores the lottery.

### Reward System

Agents receive rewards/penalties based on:

- **Forward Movement**: +0.02 per step forward
- **Max Position Bonus**: +0.1 when reaching new distance records
- **Backward Movement**: -0.1 penalty
- **Stationary Penalty**: -0.05 after 5 ticks without movement
- **Falling Penalty**: -0.02 when falling more than 5 pixels
- **Death Penalty**: -20.0 when dying
- **Progress**: max distance reached divided by 20, floored at -30.0
- **Win Bonus**: distance / 10, plus 100 per second saved under the 10 second target

Agents that do not move for 2 seconds, or that end up behind where they were 6 seconds earlier, are killed
so the generation ends sooner.

### Game Settings

Edit `game/settings.py` to customize:

- Gravity and physics parameters
- Player acceleration and max velocity
- Camera settings
- Tile sizes and collision detection

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for more details.
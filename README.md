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
- **A2C + Neuroevolution**: Agents learn by advantage actor-critic within a generation and by genetic selection between generations
- **Batched A2C**: One Adam trains the entire population at once, gradients clipped per agent
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

Training is headless by default and runs as fast as the machine allows. `--show-window` caps it to `--tick-rate` so the run stays watchable.

**Command-line options:**

- `--population-size N`: Number of agents per generation (default: 100)
- `--elite-count N`: Agents carried over untouched and used as parents (default: 4)
- `--mutation-rate R`: Probability of mutating a given weight tensor (default: 0.8, range: 0.0-1.0)
- `--mutation-strength S`: Scale of mutations (default: 0.015)
- `--hidden-sizes N N N`: Sizes of the three shared hidden layers (default: 256 128 64), smaller is faster and dumber
- `--no-use-a2c`: Disable A2C and evolve with the genetic algorithm only
- `--device auto|cpu|cuda`: Where the population runs (default: auto)
- `--threads N`: Torch CPU threads (default: 4)
- `--tick-rate N`: Simulation ticks per in-game second, also the FPS cap while rendering (default: 90)
- `--episode-seconds S`: In-game time budget per spawn point (default: 20)
- `--generations N`: Stop after N generations (default: 0, runs forever)
- `--render-every N`: Draw one frame every N ticks, only with `--show-window` (default: 1)
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
uv run run-ai.py --show-window --render-every 4
```

### Performance

The simulation and the population are both batched, so the cost of a tick is nearly flat in the number of
agents: going from 100 to 1000 agents multiplies the throughput per second, not the wall clock.

| Setup | Ticks/s | Agent-steps/s |
| --- | --- | --- |
| 100 agents, A2C + genetic, CUDA | ~620 | ~62,000 |
| 100 agents, genetic only, CUDA | ~820 | ~41,000 |
| 1000 agents, A2C + genetic, CUDA | ~155 | ~155,000 |

At 90 ticks per in-game second, 100 agents playing a 20 second episode take about 3 seconds of wall clock.

On CUDA the sampling pass is captured as a CUDA graph: a tick is a few dozen tiny kernels, so it is bound
by launch latency, and replaying one captured graph is about 3x faster than launching them one by one. The
capture falls back to eager mode on its own if the device does not support it.

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
│   ├── population.py     # The whole population as one batched actor-critic network
│   ├── a2c_trainer.py    # Rollout buffer and batched A2C update
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

The agent's network is an actor-critic over a 7×7 grid view plus the player's own state:

- **Input**: 199 features, 4 channels per tile (solid, flag, reward, empty) plus horizontal speed, vertical speed and ground contact
- **Hidden Layers**: 256, 128 then 64 neurons (LayerNorm on the first two, leaky ReLU), configurable with `--hidden-sizes`
- **Actor**: 3 action logits (jump, move left, move right), sampled with the Gumbel-max trick
- **Critic**: a 32 neuron head reducing to one state value

Every agent shares this shape, so the population is stored as one `(agents, in, out)` tensor per layer and a
forward pass for all agents is a single `baddbmm` per layer. Because an agent's weights only ever touch its
own outputs, one Adam over those tensors trains every agent independently. Weight files hold one agent in
plain `nn.Linear` / `nn.LayerNorm` layout (`fc1.weight`, `norm1.bias`, `actor.weight`, ...).

### How the two learning signals combine

- **Within a generation**: every transition goes into a rollout buffer. At the end of the generation the
  trainer replays it in chunks of 256 timesteps, so the autograd graph never covers the whole episode, and
  applies one A2C update (normalised discounted returns, entropy bonus, gradients clipped per agent).
- **Between generations**: agents are ranked by fitness, the elites are copied untouched, most of the
  population is a mutated crossover of two elites, and a few are re-randomised. The Adam moments are reset
  afterwards because the agents behind each slot have changed.

`--no-use-a2c` keeps only the second half.

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
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
- **Neuroevolution**: AI agents improve through genetic algorithms across generations
- **Deep Neural Networks**: Custom neural network architecture for agent decision-making
- **Reward Shaping**: Sophisticated reward system that encourages forward progress, penalizes backward movement, and rewards level completion
- **CUDA Support**: Automatic GPU acceleration when available
- **Persistent Training**: Save and load agent weights across generations
- **Manual Play**: Optional manual player mode to understand level design
- **Customizable Parameters**: Adjustable population size, mutation rates, and game settings
- **Level Design**: Text-based level files for easy customization

## Requirements

- Python >= 3.13
- PyTorch with CUDA 12.8 support (for GPU acceleration)
- Pygame for rendering
- NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn

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

> **Note**: The first time you run this, uv will install PyTorch with CUDA 12.8 support (on Windows and Linux). This may take a few minutes.

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

**Command-line options:**

- `--population-size N`: Number of agents per generation (default: 60)
- `--mutation-rate R`: Probability of mutation (default: 0.3, range: 0.0-1.0)
- `--mutation-strength S`: Scale of mutations (default: 0.02)
- `--load-latest-generation-weights`: Load weights from the latest saved generation
- `--show-window`: Display the game window during training
- `--checkpoints`: Use checkpoint platforms as spawn points

**Example:**

```bash
uv run run-ai.py --population-size 80 --mutation-rate 0.25 --show-window
```

### Play Manually

To play the game yourself:

```bash
uv run run-game.py
```

## Project Structure

```
Neural-Jump/
├── ai/                    # AI agent and neural network code
│   ├── agent.py          # Agent class with decision-making logic
│   ├── generation.py     # Population management and evolution
│   ├── neural_network.py # Neural network architecture
│   └── __init__.py
├── game/                  # Game engine and mechanics
│   ├── game.py           # Main game loop
│   ├── player.py         # Player character logic
│   ├── level.py          # Level management
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

The agent's neural network processes a 7×7 grid view with 2 channels (obstacles and rewards):

- **Input**: 98 features (7×7×2)
- **Hidden Layer 1**: 256 neurons (ReLU activation)
- **Hidden Layer 2**: 128 neurons (ReLU activation)
- **Hidden Layer 3**: 64 neurons (ReLU activation)
- **Output**: 3 actions (jump, move left, move right)

### Reward System

Agents receive rewards/penalties based on:

- **Forward Movement**: +0.02 per step forward
- **Max Position Bonus**: +0.1 when reaching new distance records
- **Backward Movement**: -0.1 penalty
- **Stationary Penalty**: -0.05 after 5 ticks without movement
- **Falling Penalty**: -0.02 when falling more than 5 pixels
- **Death Penalty**: -20.0 when dying
- **Win Bonus**: Up to +10.0 based on completion time

### Game Settings

Edit `game/settings.py` to customize:

- Gravity and physics parameters
- Player acceleration and max velocity
- Camera settings
- Tile sizes and collision detection

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for more details.
# envpack

A collection of game environments for Gymnasium. Currently includes the game 2048.

[![Python package](https://github.com/rax85/gym_2048/actions/workflows/python-package.yml/badge.svg)](https://github.com/rax85/gym_2048/actions/workflows/python-package.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## Description

This package implements classic games as Gymnasium environments. Currently, it supports:
1. **2048**: Played on a 4x4 grid. The goal is to slide tiles on the grid to combine tiles of the same value and create a tile with the value 2048.

## Installation

To install the environments, you can use pip:

```bash
pip install git+https://github.com/rax85/envpack.git
```

## Usage

To use the environment, you can use the following code:

```python
import gymnasium as gym
import envpack

env = gym.make('envpack/2048-v0')

observation, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

## Action Space

The action space is a `Discrete(4)` space, where the actions are:

*   `0`: Up
*   `1`: Down
*   `2`: Left
*   `3`: Right

## Observation Space

The observation space is a `Dict` with three keys:

*   `'observation'`: A `Box` of shape `(4, 4)` representing the integer values of the game board.
*   `'valid_mask'`: A `Box` of shape `(4,)` representing a binary mask of valid actions.
*   `'total_score'`: A `Box` of shape `(1,)` representing the total score of the game.

## Rewards

The reward is the sum of the values of the merged tiles in a step. If an invalid move is made, the reward is -32.

## Screenshots

### Initial State

![Initial State](screenshot_initial.png)

### Mid-game State

![Mid-game State](screenshot_mid_game.png)

### Game Over State

![Game Over State](screenshot_game_over.png)

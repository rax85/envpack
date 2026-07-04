# envpack

A collection of classic game environments for Gymnasium.

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Installation

To install the environments, you can use pip:

```bash
pip install git+https://github.com/rax85/envpack.git
```

## Usage

```python
import gymnasium as gym
import envpack

# To run 2048
env = gym.make('envpack/2048-v0')

# To run Snake
# env = gym.make('envpack/Snake-v0')

# To run Tetris
# env = gym.make('envpack/Tetris-v0')

# To run Sudoku
# env = gym.make('envpack/Sudoku-v0')

observation, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Take a random action
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

---

## Game Environments

### 1. 2048 (`envpack/2048-v0`)

A Gymnasium environment for the classic 2048 tile-merging game played on a 4x4 grid.

*   **Action Space**: `Discrete(4)`:
    *   `0`: Up, `1`: Down, `2`: Left, `3`: Right
*   **Observation Space**: `Dict` containing:
    *   `'observation'`: `Box(4, 4)` representing tile values.
    *   `'valid_mask'`: `Box(4,)` binary mask of valid moves.
    *   `'total_score'`: `Box(1,)` representing the accumulated score.
*   **Rewards**: Sum of merged tile values. Invalid moves yield `-32`.
*   **Screenshots**:
    *   *Initial State*: ![2048 Initial State](screenshots/screenshot_initial.png)
    *   *Mid-game*: ![2048 Mid-game State](screenshots/screenshot_mid_game.png)
    *   *Game Over*: ![2048 Game Over State](screenshots/screenshot_game_over.png)

### 2. Snake (`envpack/Snake-v0`)

A Gymnasium environment for the classic Snake game played on a 10x10 grid.

*   **Action Space**: `Discrete(4)`:
    *   `0`: Up, `1`: Down, `2`: Left, `3`: Right
*   **Observation Space**: `Dict` containing:
    *   `'observation'`: `Box(10, 10)` representing the board (0: empty, 1: food, 2: snake head, 3: snake body).
    *   `'valid_mask'`: `Box(4,)` binary mask of valid moves (direct backward folding is masked out).
    *   `'total_score'`: `Box(1,)` representing the number of food items eaten.
*   **Rewards**: `+1.0` for eating food, `-0.01` step penalty, and `-1.0` for wall/self collision.
*   **Screenshots**:
    *   *Initial State*: ![Snake Initial State](screenshots/snake_screenshot_initial.png)
    *   *Mid-game*: ![Snake Mid-game State](screenshots/snake_screenshot_mid_game.png)
    *   *Game Over*: ![Snake Game Over State](screenshots/snake_screenshot_game_over.png)

### 3. Tetris (`envpack/Tetris-v0`)

A Gymnasium environment for the classic Tetris block-falling puzzle game played on a 10x20 grid.

*   **Action Space**: `Discrete(5)`:
    *   `0`: Move Left, `1`: Move Right, `2`: Rotate Clockwise, `3`: Soft Drop (Down 1), `4`: Hard Drop (Instant drop & lock)
*   **Observation Space**: `Dict` containing:
    *   `'observation'`: `Box(20, 10)` representing the board (0: empty, 1..7: landed tetromino blocks, 8: active falling piece blocks).
    *   `'valid_mask'`: `Box(5,)` binary mask of valid actions.
    *   `'total_score'`: `Box(1,)` representing the accumulated score.
*   **Rewards**: Small survival reward of `+0.01` per step. Clearing lines yields: `0.1` (1 line), `0.3` (2 lines), `0.5` (3 lines), `1.0` (4 lines). Game over yields `-1.0`.
*   **Screenshots**:
    *   *Initial State*: ![Tetris Initial State](screenshots/tetris_screenshot_initial.png)
    *   *Mid-game*: ![Tetris Mid-game State](screenshots/tetris_screenshot_mid_game.png)
    *   *Game Over*: ![Tetris Game Over State](screenshots/tetris_screenshot_game_over.png)

### 4. Sudoku (`envpack/Sudoku-v0`)

A Gymnasium environment for solving standard 9x9 Sudoku puzzles.

*   **Action Space**: `MultiDiscrete([9, 9, 10])`:
    *   `row` in `[0..8]`: Target row coordinate to edit.
    *   `col` in `[0..8]`: Target column coordinate to edit.
    *   `value` in `[0..9]`: Digit to place (`1..9`), or `0` to clear/delete the digit.
*   **Observation Space**: `Dict` containing:
    *   `'observation'`: `Box(9, 9)` representing current cell digits.
    *   `'given_mask'`: `Box(9, 9)` representing fixed clues (1 if given clue, 0 if editable).
    *   `'valid_mask'`: `Box(9, 9, 10)` representing safe (conflict-free) digits that can be placed in each cell.
    *   `'total_score'`: `Box(1,)` representing number of cells matching target solution.
*   **Rewards**: `+1.0` for placing a correct digit, `-1.0` for removing/replacing a correct digit, `-0.1` for constraint conflict violations, and `-0.01` step penalty. Completion yields a `+10.0` bonus.
*   **Screenshots**:
    *   *Initial State*: ![Sudoku Initial State](screenshots/sudoku_screenshot_initial.png)
    *   *Mid-game*: ![Sudoku Mid-game State](screenshots/sudoku_screenshot_mid_game.png)
    *   *Solved State*: ![Sudoku Solved State](screenshots/sudoku_screenshot_solved.png)

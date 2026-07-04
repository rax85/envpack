import os
import numpy as np
from PIL import Image
from envpack.envs.game_2048.env import Gym2048Env
from envpack.envs.game_snake.env import GymSnakeEnv
from envpack.envs.game_tetris.env import GymTetrisEnv
from envpack.envs.game_sudoku.env import GymSudokuEnv
from envpack.envs.game_raptor.env import GymRaptorEnv


def save_screenshot(env, name):
    rgb_data = env.render()
    image = Image.fromarray(rgb_data)
    os.makedirs("screenshots", exist_ok=True)
    path = os.path.join("screenshots", f"{name}.png")
    image.save(path)
    print(f"Saved {path}")


def generate_game_screenshots(env, name_prefix):
    # Initial
    env.reset(seed=42)
    save_screenshot(env, f"{name_prefix}_initial")

    # Mid Game
    env.action_space.seed(42)
    for _ in range(50):
        action = env.action_space.sample()
        _, _, done, truncated, _ = env.step(action)
        if done or truncated:
            env.reset()
    save_screenshot(env, f"{name_prefix}_mid_game")

    # Game Over
    done = False
    truncated = False
    max_steps = 10000
    steps = 0
    while not (done or truncated) and steps < max_steps:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
    save_screenshot(env, f"{name_prefix}_game_over")


def generate_sudoku_screenshots():
    env = GymSudokuEnv()
    
    # Initial
    env.reset(seed=42)
    save_screenshot(env, "sudoku_screenshot_initial")
    
    # Mid Game
    env.reset(seed=42)
    # Inject some move entries to showcase all text colors
    env._grid[0, 4] = 1
    env._grid[0, 1] = 2
    env._grid[0, 2] = 2
    env._grid[1, 3] = 3
    env._move_history.append(((0, 1, 2), False))
    env._move_history.append(((0, 2, 2), False))
    env._move_history.append(((1, 3, 3), True))
    save_screenshot(env, "sudoku_screenshot_mid_game")
    
    # Solved
    env_solved = GymSudokuEnv(clues=80)
    env_solved.reset(seed=42)
    edit_r, edit_c = np.argwhere(env_solved._given_mask == 0)[0]
    correct_val = env_solved._solved_grid[edit_r, edit_c]
    env_solved.step(np.array([edit_r, edit_c, correct_val], dtype=np.int32))
    save_screenshot(env_solved, "sudoku_screenshot_solved")


def generate_raptor_screenshots():
    env = GymRaptorEnv()
    
    # Initial
    env.reset(seed=42)
    save_screenshot(env, "raptor_screenshot_initial")
    
    # Mid-game
    env.reset(seed=42)
    # Inject some items, enemies, bullets, lasers, and moves to showcase gameplay
    env._enemies.append(([2, 2], 3)) # ENEMY_BASIC
    env._enemies.append(([5, 7], 4)) # ENEMY_SHOOTER
    env._bullets.append([7, 7])
    env._coins.append([9, 10])
    env._lasers.append([12, 5])
    env.step(1) # LEFT
    env.step(3) # UP
    save_screenshot(env, "raptor_screenshot_mid_game")
    
    # Game Over
    env.reset(seed=42)
    env._shield = 0
    save_screenshot(env, "raptor_screenshot_game_over")


def main():
    print("Generating 2048 screenshots...")
    env_2048 = Gym2048Env()
    generate_game_screenshots(env_2048, "screenshot")

    print("Generating Snake screenshots...")
    env_snake = GymSnakeEnv()
    generate_game_screenshots(env_snake, "snake_screenshot")

    print("Generating Tetris screenshots...")
    env_tetris = GymTetrisEnv()
    generate_game_screenshots(env_tetris, "tetris_screenshot")

    print("Generating Sudoku screenshots...")
    generate_sudoku_screenshots()

    print("Generating Raptor screenshots...")
    generate_raptor_screenshots()


if __name__ == "__main__":
    main()

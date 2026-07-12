import os
import numpy as np
from PIL import Image
from envpack.envs.game_2048.env import Gym2048Env
from envpack.envs.game_snake.env import GymSnakeEnv
from envpack.envs.game_tetris.env import GymTetrisEnv
from envpack.envs.game_sudoku.env import GymSudokuEnv
from envpack.envs.game_raptor.env import GymRaptorEnv
from envpack.envs.game_checkers.env import GymCheckersEnv
from envpack.envs.game_tron.env import GymTronEnv
from envpack.envs.game_air_hockey.env import GymAirHockeyEnv
from envpack.envs.game_racing.env import GymRacingEnv
from envpack.envs.game_doom.env import GymDoomEnv
from envpack.envs.game_paratrooper.env import GymParatrooperEnv
from envpack.envs.game_street_fighter.env import GymStreetFighterEnv



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


def generate_checkers_screenshots():
    env = GymCheckersEnv()
    
    # Initial
    env.reset(seed=42)
    save_screenshot(env, "checkers_screenshot_initial")
    
    # Mid-game
    env.reset(seed=42)
    env.step(np.array([5, 0, 4, 1], dtype=np.int32))
    env.step(np.array([2, 1, 3, 0], dtype=np.int32))
    env._grid[0, 1] = 2 # P1_KING
    save_screenshot(env, "checkers_screenshot_mid_game")
    
    # Game Over (win state)
    env.reset(seed=42)
    env._grid = np.zeros((8, 8), dtype=np.int32)
    env._grid[4, 4] = 1 # P1_NORMAL
    env._grid[3, 3] = 3 # P2_NORMAL
    env.step(np.array([4, 4, 2, 2], dtype=np.int32)) # P1 captures P2, P2 pieces count = 0 -> P1 wins!
    save_screenshot(env, "checkers_screenshot_game_over")


def generate_tron_screenshots():
    env = GymTronEnv()
    
    # Initial
    env.reset(seed=42)
    save_screenshot(env, "tron_screenshot_initial")
    
    # Mid-game
    env.reset(seed=42)
    # Move forward a few steps
    env.step(np.array([3, 2], dtype=np.int32)) # RIGHT, LEFT
    env.step(np.array([0, 1], dtype=np.int32)) # UP, DOWN
    save_screenshot(env, "tron_screenshot_mid_game")
    
    # Game Over (crash)
    env.reset(seed=42)
    # Force them to crash head-on
    env._grid = np.zeros((30, 30), dtype=np.int32)
    env._p1_pos = (15, 14)
    env._p2_pos = (15, 16)
    env.step(np.array([3, 2], dtype=np.int32)) # RIGHT, LEFT -> crash head-on
    save_screenshot(env, "tron_screenshot_game_over")


def generate_air_hockey_screenshots():
    env = GymAirHockeyEnv()
    
    # Initial
    env.reset(seed=42)
    save_screenshot(env, "air_hockey_screenshot_initial")
    
    # Mid-game
    env.reset(seed=42)
    # Move mallets and push puck
    env.step(np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.float32))
    save_screenshot(env, "air_hockey_screenshot_mid_game")
    
    # Game Over
    env.reset(seed=42)
    env._scores = np.array([7, 2], dtype=np.int32)
    save_screenshot(env, "air_hockey_screenshot_game_over")


def generate_racing_screenshots():
    env = GymRacingEnv()
    
    # Initial
    env.reset(seed=42)
    save_screenshot(env, "racing_screenshot_initial")
    
    # Mid-game
    env.reset(seed=42)
    # Add skidmarks, drift, and redline to display the engine HUD and tire dynamics
    env._p1_x = 220.0
    env._p1_y = 120.0
    env._p1_theta = -0.5
    env._p1_rpm = 7100.0
    env._p1_gear = 2
    env._p1_skids = [(210.0, 110.0), (205.0, 108.0)]
    env._p2_x = 260.0
    env._p2_y = 100.0
    env._p2_theta = 0.5
    env._p2_rpm = 5500.0
    env._p2_gear = 3
    env._p2_skids = [(270.0, 90.0), (275.0, 88.0)]
    save_screenshot(env, "racing_screenshot_mid_game")
    
    # Game Over
    env.reset(seed=42)
    env._scores = np.array([1.0, 0.0], dtype=np.float32)
    # Project winner P1 to end spline
    end_idx = len(env._track_spline) - 1
    env._p1_x, env._p1_y = env._track_spline[end_idx]
    env._p1_progress = 1.0
    save_screenshot(env, "racing_screenshot_game_over")


def generate_street_fighter_screenshots():
    env = GymStreetFighterEnv()
    
    # Initial
    env.reset(seed=42)
    save_screenshot(env, "street_fighter_screenshot_initial")
    
    # Mid-game
    env.reset(seed=42)
    env.x = [150.0, 200.0]
    env.state = ["punch", "hitstun"]
    env.hitstun[1] = 5
    env.sparks.append({"x": 175.0, "y": 200.0, "lifetime": 4})
    env.fireballs.append({
        "x": 280.0,
        "y": 200.0,
        "dir": -1,
        "owner": 1,
        "speed": 5.0,
        "active": True
    })
    save_screenshot(env, "street_fighter_screenshot_mid_game")
    
    # Game Over
    env.reset(seed=42)
    env.health = [40, 0]
    env.state = ["idle", "knockdown"]
    env.knockdown[1] = 15
    save_screenshot(env, "street_fighter_screenshot_game_over")


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

    print("Generating Checkers screenshots...")
    generate_checkers_screenshots()

    print("Generating Tron screenshots...")
    generate_tron_screenshots()

    print("Generating Air Hockey screenshots...")
    generate_air_hockey_screenshots()

    print("Generating Racing screenshots...")
    generate_racing_screenshots()

    print("Generating Doom screenshots...")
    env_doom = GymDoomEnv()
    generate_game_screenshots(env_doom, "doom_screenshot")

    print("Generating Paratrooper screenshots...")
    env_paratrooper = GymParatrooperEnv()
    generate_game_screenshots(env_paratrooper, "paratrooper_screenshot")

    print("Generating Street Fighter screenshots...")
    generate_street_fighter_screenshots()


if __name__ == "__main__":
    main()

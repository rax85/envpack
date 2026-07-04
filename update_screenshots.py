import os
import numpy as np
from PIL import Image
from envpack.envs.game_2048.env import Gym2048Env
from envpack.envs.game_snake.env import GymSnakeEnv
import random


def save_screenshot(env, name):
    rgb_data = env.render()
    image = Image.fromarray(rgb_data)
    image.save(f"{name}.png")
    print(f"Saved {name}.png")


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


def main():
    print("Generating 2048 screenshots...")
    env_2048 = Gym2048Env()
    generate_game_screenshots(env_2048, "screenshot")

    print("Generating Snake screenshots...")
    env_snake = GymSnakeEnv()
    generate_game_screenshots(env_snake, "snake_screenshot")


if __name__ == "__main__":
    main()

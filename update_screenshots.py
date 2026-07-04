import os
import numpy as np
from PIL import Image
from envpack.envs.game_2048.env import Gym2048Env
import random


def save_screenshot(env, name):
    rgb_data = env.render()
    image = Image.fromarray(rgb_data)
    image.save(f"{name}.png")
    print(f"Saved {name}.png")


def main():
    env = Gym2048Env()

    # Initial
    env.reset(seed=42)
    save_screenshot(env, "screenshot_initial")

    # Mid Game
    # Play some steps
    # Seed random for reproducibility
    env.action_space.seed(42)

    for _ in range(50):  # 50 steps for mid game
        action = env.action_space.sample()
        env.step(action)
    save_screenshot(env, "screenshot_mid_game")

    # Game Over
    # Play until done
    done = False
    truncated = False
    max_steps = 10000
    steps = 0
    while not (done or truncated) and steps < max_steps:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        steps += 1

    print(f"Game over after {steps} additional steps.")
    save_screenshot(env, "screenshot_game_over")


if __name__ == "__main__":
    main()

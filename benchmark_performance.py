"""
Benchmark script for the Gym2048Env environment.
Measures the average step time over a large number of steps.
"""

import time
import numpy as np
from envpack.envs.game_2048.env import Gym2048Env


def benchmark():
    env = Gym2048Env()
    env.reset()

    # Warmup
    print("Warming up...")
    for _ in range(100):
        action = env.action_space.sample()
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            env.reset()

    print("Starting benchmark...")
    num_steps = 10000
    start_time = time.time()

    for _ in range(num_steps):
        action = env.action_space.sample()
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            env.reset()

    end_time = time.time()
    total_time = end_time - start_time
    avg_step_time = total_time / num_steps

    print(f"Total time for {num_steps} steps: {total_time:.4f} seconds")
    print(f"Average step time: {avg_step_time:.6f} seconds")
    print(f"Steps per second: {1/avg_step_time:.2f}")


if __name__ == "__main__":
    benchmark()

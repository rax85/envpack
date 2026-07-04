"""Tests for the GymSnakeEnv environment."""

import copy
import unittest
import numpy as np

from envpack.envs.game_snake import env as snake_env
from envpack.envs.game_snake.env import GymSnakeEnv


class TestGymSnakeEnv(unittest.TestCase):
    """Tests for the GymSnakeEnv environment."""

    def test_initial_state(self):
        """Test that the initial state of the environment is set up correctly."""
        env = GymSnakeEnv()
        obs, _ = env.reset()

        # Snake should have length 1 and be placed at center (5, 5)
        self.assertEqual(len(env.snake), 1)
        self.assertEqual(env.snake[0], (5, 5))
        self.assertEqual(env.direction, snake_env.RIGHT)
        self.assertEqual(env.score, 0)

        # Check observation structure
        self.assertIn("observation", obs)
        self.assertIn("valid_mask", obs)
        self.assertIn("total_score", obs)
        
        # Grid checks
        grid = obs["observation"]
        self.assertEqual(grid[5, 5], 2)  # 2 is the head
        self.assertEqual(np.count_nonzero(grid == 2), 1)
        self.assertEqual(np.count_nonzero(grid == 1), 1)  # 1 food spawned
        self.assertEqual(obs["total_score"][0], 0)

        # Since length is 1, all directions should be valid (mask is all ones)
        np.testing.assert_array_equal(obs["valid_mask"], [1, 1, 1, 1])

    def test_movement_without_eating(self):
        """Test that moving the snake shifts positions and leaves tail behind."""
        env = GymSnakeEnv()
        env.reset()
        # Force food to be far away so we don't eat it
        env.food = (0, 0)
        
        obs, reward, terminated, truncated, info = env.step(snake_env.RIGHT)
        
        # Head should be at (5, 6), previous position should be empty
        self.assertEqual(env.snake, [(5, 6)])
        self.assertEqual(reward, -0.01)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        
        grid = obs["observation"]
        self.assertEqual(grid[5, 6], 2)
        self.assertEqual(grid[5, 5], 0)

    def test_eating_food(self):
        """Test that eating food increases score, snake grows, and spawns new food."""
        env = GymSnakeEnv()
        env.reset()
        
        # Force food to be one step to the right of the snake
        env.food = (5, 6)
        
        obs, reward, terminated, truncated, info = env.step(snake_env.RIGHT)
        
        self.assertEqual(env.score, 1)
        self.assertEqual(reward, 1.0)
        # Snake grows: head is (5, 6), body is (5, 5)
        self.assertEqual(env.snake, [(5, 6), (5, 5)])
        self.assertFalse(terminated)
        
        # Check grid observation
        grid = obs["observation"]
        self.assertEqual(grid[5, 6], 2)  # Head
        self.assertEqual(grid[5, 5], 3)  # Body
        
        # Food should have spawned in a new location (not on snake body or head)
        new_food = env.food
        self.assertNotEqual(new_food, (5, 6))
        self.assertNotEqual(new_food, (5, 5))
        self.assertEqual(grid[new_food[0], new_food[1]], 1)

    def test_mask_and_invalid_direction(self):
        """Test that moving backwards when length > 1 is masked and ignored."""
        env = GymSnakeEnv()
        env.reset()
        env.food = (5, 6)
        
        # Step RIGHT to eat food and grow
        env.step(snake_env.RIGHT)
        self.assertEqual(len(env.snake), 2)
        self.assertEqual(env.direction, snake_env.RIGHT)
        
        # Verify valid moves mask (opposite LEFT is invalid)
        obs, _ = env._create_observation()
        # UP, DOWN, LEFT, RIGHT -> 0, 1, 2, 3. LEFT (2) should be 0.
        np.testing.assert_array_equal(obs["valid_mask"], [1, 1, 0, 1])

        # Attempt to move LEFT (invalid) - should ignore and continue RIGHT
        obs, reward, terminated, _, _ = env.step(snake_env.LEFT)
        self.assertEqual(env.direction, snake_env.RIGHT)
        self.assertEqual(env.snake[0], (5, 7))  # Moved RIGHT instead of LEFT
        self.assertFalse(terminated)

    def test_wall_collision(self):
        """Test that hitting the wall terminates the game with a penalty."""
        env = GymSnakeEnv()
        env.reset()
        
        # Move RIGHT until we hit the right boundary
        # Head starts at (5, 5). 4 moves right gets head to (5, 9). 5th move hits the wall.
        for _ in range(4):
            _, _, terminated, _, _ = env.step(snake_env.RIGHT)
            self.assertFalse(terminated)
            
        obs, reward, terminated, _, _ = env.step(snake_env.RIGHT)
        self.assertTrue(terminated)
        self.assertEqual(reward, -1.0)

    def test_self_collision(self):
        """Test that hitting its own body terminates the game."""
        env = GymSnakeEnv()
        env.reset()
        
        # Set snake manually to a state where it's long and coiled
        # Head: (5, 5) -> (5, 6) -> (6, 6) -> (6, 5) -> (5, 5) (collision)
        env.snake = [(5, 5), (5, 6), (6, 6), (6, 5)]
        env.direction = snake_env.LEFT
        
        obs, reward, terminated, _, _ = env.step(snake_env.DOWN)
        self.assertTrue(terminated)
        self.assertEqual(reward, -1.0)

    def test_steps_limit_truncation(self):
        """Test that exceeding the step limit without eating results in truncation."""
        env = GymSnakeEnv()
        env.reset()
        env.food = (0, 0)
        
        # Manually set steps since eating to 199
        env.steps_since_eating = 199
        
        # Step once more (reaches 200)
        _, _, _, truncated, _ = env.step(snake_env.RIGHT)
        self.assertTrue(truncated)

    def test_state_saving_and_restoring(self):
        """Test saving and restoring state works correctly."""
        env = GymSnakeEnv()
        env.reset()
        
        # Perform some steps
        env.step(snake_env.RIGHT)
        env.step(snake_env.DOWN)
        _, _, _, _, info = env.step(snake_env.LEFT)
        
        saved_state = info["state"]
        
        new_env = GymSnakeEnv()
        new_env.reset(options={"state": saved_state})
        
        self.assertEqual(new_env.snake, env.snake)
        self.assertEqual(new_env.direction, env.direction)
        self.assertEqual(new_env.food, env.food)
        self.assertEqual(new_env.score, env.score)
        self.assertEqual(new_env.total_moves, env.total_moves)
        self.assertEqual(new_env.steps_since_eating, env.steps_since_eating)
        self.assertEqual(new_env.move_history, env.move_history)


if __name__ == "__main__":
    unittest.main()

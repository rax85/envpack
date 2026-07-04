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
        obs = env._create_observation()
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

    def test_tail_chasing_no_collision(self):
        """Test that moving into the tail position (which is vacated) does not cause a collision."""
        env = GymSnakeEnv()
        env.reset()
        # Food far away
        env.food = (0, 0)
        # Head at (5, 5), tail at (6, 5). Moving DOWN makes head (6, 5).
        env.snake = [(5, 5), (5, 6), (6, 6), (6, 5)]
        env.direction = snake_env.LEFT

        obs, reward, terminated, _, _ = env.step(snake_env.DOWN)
        self.assertFalse(terminated)
        self.assertEqual(env.snake[0], (6, 5))  # Successfully moved to tail position
        self.assertEqual(len(env.snake), 4)

    def test_self_collision(self):
        """Test that hitting its own body (excluding the tail) terminates the game."""
        env = GymSnakeEnv()
        env.reset()
        # Food far away
        env.food = (0, 0)
        # Head at (5, 5), body segments (5, 6), (6, 6), (6, 5), and tail at (6, 4).
        # Moving DOWN makes head (6, 5), which is a non-tail body segment.
        env.snake = [(5, 5), (5, 6), (6, 6), (6, 5), (6, 4)]
        env.direction = snake_env.LEFT

        obs, reward, terminated, _, _ = env.step(snake_env.DOWN)
        self.assertTrue(terminated)
        self.assertEqual(reward, -1.0)

    def test_gymnasium_compliance(self):
        """Test that the environment complies with Gymnasium specs."""
        from gymnasium.utils.env_checker import check_env
        env = GymSnakeEnv()
        check_env(env, skip_render_check=True)

    def test_seeding_and_determinism(self):
        """Test that resetting with a seed produces deterministic food spawns and steps."""
        env1 = GymSnakeEnv()
        env2 = GymSnakeEnv()
        
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        # Verify initial observations match
        np.testing.assert_array_equal(obs1["observation"], obs2["observation"])
        self.assertEqual(env1.food, env2.food)
        
        # Step through both and verify they remain identical
        for _ in range(5):
            action = env1.action_space.sample()
            o1, r1, term1, trunc1, _ = env1.step(action)
            o2, r2, term2, trunc2, _ = env2.step(action)
            np.testing.assert_array_equal(o1["observation"], o2["observation"])
            self.assertEqual(r1, r2)
            self.assertEqual(term1, term2)
            self.assertEqual(trunc1, trunc2)
            self.assertEqual(env1.food, env2.food)

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

    def test_rendering_extra_coverage(self):
        """Test rendering, arrow drawing, and close in GymSnakeEnv."""
        env = GymSnakeEnv()
        env.reset()
        
        # Step 10 times in all directions to build history and trigger pop(0)
        # Start at (5, 5). Moving sequence of UP, LEFT, DOWN, RIGHT etc.
        actions = [
            snake_env.UP, snake_env.LEFT, snake_env.DOWN, snake_env.RIGHT,
            snake_env.UP, snake_env.LEFT, snake_env.DOWN, snake_env.RIGHT,
            snake_env.UP, snake_env.LEFT
        ]
        for a in actions:
            env.step(a)
            
        # Verify history popped
        self.assertEqual(len(env.move_history), 8)
        
        # Render observation
        img = env.render()
        self.assertIsNotNone(img)
        
        # Invalid arrow action
        from PIL import ImageDraw, Image
        canvas = Image.new("RGB", (100, 100))
        draw = ImageDraw.Draw(canvas)
        env._draw_arrow(draw, 50, 50, 99, (255, 255, 255))
        
        env.close()

    def test_font_loading_fallback(self):
        """Test font loading fallback when matplotlib findfont fails."""
        from unittest.mock import patch
        with patch("matplotlib.font_manager.findfont", side_effect=Exception("Font error")):
            env = GymSnakeEnv()
            self.assertIsNotNone(env._score_font)

    def test_perfect_game_and_food_spawn_failure(self):
        """Test when the snake occupies the entire grid and food cannot spawn."""
        env = GymSnakeEnv()
        env.reset()
        
        # Snake head at (0, 1), body covering all other cells except (0, 0)
        env.snake = [(0, 1)] + [(y, x) for y in range(10) for x in range(10) if (y, x) not in [(0, 0), (0, 1)]]
        env.food = (0, 0)
        env.direction = snake_env.LEFT
        
        obs, reward, terminated, truncated, info = env.step(snake_env.LEFT)
        self.assertTrue(terminated)
        self.assertEqual(env.food, (-1, -1))
        self.assertGreaterEqual(reward, 5.0)

    def test_invalid_action_value_error(self):
        """Test that invalid action values raise a ValueError."""
        env = GymSnakeEnv()
        env.reset()
        with self.assertRaises(ValueError):
            env.step(5)


if __name__ == "__main__":
    unittest.main()

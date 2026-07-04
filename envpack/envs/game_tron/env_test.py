"""Tests for GymTronEnv."""

import copy
import unittest
import numpy as np

from envpack.envs.game_tron import env as tron_env
from envpack.envs.game_tron.env import GymTronEnv, CANVAS_SIZE, GRID_SIZE, EMPTY, P1_HEAD, P1_TRAIL, P2_HEAD, P2_TRAIL, UP, DOWN, LEFT, RIGHT


class TestGymTronEnv(unittest.TestCase):
    """Tests for the GymTronEnv environment."""

    def test_initial_state(self):
        """Test that the initial state is set up correctly."""
        env = GymTronEnv()
        obs, _ = env.reset()

        self.assertEqual(env._p1_pos, (GRID_SIZE // 2, 4))
        self.assertEqual(env._p2_pos, (GRID_SIZE // 2, GRID_SIZE - 5))
        self.assertEqual(env._p1_dir, RIGHT)
        self.assertEqual(env._p2_dir, LEFT)

        self.assertEqual(env._grid[env._p1_pos], P1_HEAD)
        self.assertEqual(env._grid[env._p2_pos], P2_HEAD)

        # Check observations
        self.assertIn("observation", obs)
        self.assertIn("valid_mask", obs)
        self.assertIn("total_score", obs)

        # Opposite directions are blocked
        np.testing.assert_array_equal(obs["valid_mask"], [[1, 1, 0, 1], [1, 1, 1, 0]])

    def test_step_movement(self):
        """Test moving straight for both players."""
        env = GymTronEnv()
        env.reset()

        action = np.array([RIGHT, LEFT], dtype=np.int32)
        obs, reward, terminated, truncated, info = env.step(action)

        self.assertEqual(env._p1_pos, (GRID_SIZE // 2, 5))
        self.assertEqual(env._p2_pos, (GRID_SIZE // 2, GRID_SIZE - 6))
        
        self.assertEqual(env._grid[GRID_SIZE // 2, 4], P1_TRAIL)
        self.assertEqual(env._grid[GRID_SIZE // 2, GRID_SIZE - 5], P2_TRAIL)
        self.assertEqual(env._grid[env._p1_pos], P1_HEAD)
        self.assertEqual(env._grid[env._p2_pos], P2_HEAD)

        self.assertEqual(reward, 0.01) # Survival incentive
        self.assertFalse(terminated)

    def test_suicide_prevention(self):
        """Test that picking opposite direction forces straight direction."""
        env = GymTronEnv()
        env.reset()

        # Opposite of RIGHT is LEFT, opposite of LEFT is RIGHT
        action = np.array([LEFT, RIGHT], dtype=np.int32)
        obs, reward, terminated, truncated, info = env.step(action)

        # Directions should remain unchanged
        self.assertEqual(env._p1_dir, RIGHT)
        self.assertEqual(env._p2_dir, LEFT)
        self.assertEqual(env._p1_pos, (GRID_SIZE // 2, 5))

    def test_collision_boundaries(self):
        """Test crash when hitting grid boundaries."""
        # 1. P1 hits left wall
        env = GymTronEnv()
        env.reset()
        env._p1_pos = (15, 0)
        env._p1_dir = LEFT

        obs, reward, terminated, truncated, info = env.step(np.array([LEFT, LEFT], dtype=np.int32))
        self.assertTrue(terminated)
        self.assertEqual(reward, -10.0) # P2 wins

        # 2. P2 hits right wall
        env = GymTronEnv()
        env.reset()
        env._p2_pos = (15, GRID_SIZE - 1)
        env._p2_dir = RIGHT

        obs, reward, terminated, truncated, info = env.step(np.array([RIGHT, RIGHT], dtype=np.int32))
        self.assertTrue(terminated)
        self.assertEqual(reward, 10.0) # P1 wins

        # 3. P2 hits P1's trail (within bounds)
        env = GymTronEnv()
        env.reset()
        env._grid[15, GRID_SIZE - 6] = P1_TRAIL
        obs, reward, terminated, truncated, info = env.step(np.array([RIGHT, LEFT], dtype=np.int32))
        self.assertTrue(terminated)
        self.assertEqual(reward, 10.0) # P1 wins

    def test_collision_self_trail(self):
        """Test crash when hitting own trail."""
        env = GymTronEnv()
        env.reset()

        # P1 loops to hit own trail (UP, RIGHT, DOWN, LEFT)
        env.step(np.array([UP, LEFT], dtype=np.int32))
        env.step(np.array([RIGHT, LEFT], dtype=np.int32))
        env.step(np.array([DOWN, LEFT], dtype=np.int32))
        obs, reward, terminated, truncated, info = env.step(np.array([LEFT, LEFT], dtype=np.int32))

        self.assertTrue(terminated)
        self.assertEqual(reward, -10.0) # P1 crashes

    def test_collision_opponent_trail(self):
        """Test crash when hitting opponent's trail."""
        env = GymTronEnv()
        env.reset()

        # Force P2 to leave trail in front of P1 (at 15, 6)
        # P1 starts at (15, 4) moving RIGHT.
        env._grid[15, 6] = P2_TRAIL
        
        # Step 1: P1 moves to (15, 5)
        env.step(np.array([RIGHT, LEFT], dtype=np.int32))
        # Step 2: P1 moves into P2's trail at (15, 6) and crashes
        obs, reward, terminated, truncated, info = env.step(np.array([RIGHT, LEFT], dtype=np.int32))
        self.assertTrue(terminated)
        self.assertEqual(reward, -10.0) # P1 crashes

    def test_head_on_collision(self):
        """Test head-on collision where both crash at the same target spot."""
        env = GymTronEnv()
        env.reset()

        # Force them to be 1 step away from each other
        env._grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        env._p1_pos = (15, 14)
        env._grid[15, 14] = P1_HEAD
        env._p2_pos = (15, 16)
        env._grid[15, 16] = P2_HEAD

        # Move to 15,15 together
        obs, reward, terminated, truncated, info = env.step(np.array([RIGHT, LEFT], dtype=np.int32))
        self.assertTrue(terminated)
        self.assertEqual(reward, 0.0) # Draw
        self.assertIn((15, 15), env._crash_pos)

    def test_rendering_and_close(self):
        """Test canvas rendering with crashes and entities."""
        env = GymTronEnv()
        env.reset(seed=42)

        # Step until crash to populate trails and explosions
        terminated = False
        while not terminated:
            _, _, terminated, _, _ = env.step(np.array([RIGHT, LEFT], dtype=np.int32))

        img = env.render()
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (CANVAS_SIZE[1], CANVAS_SIZE[0], 3))
        
        env.close()

    def test_font_loading_fallback(self):
        """Test font loading fallback."""
        from unittest.mock import patch
        with patch("matplotlib.font_manager.findfont", side_effect=Exception("Font error")):
            env = GymTronEnv()
            self.assertIsNotNone(env._title_font)

    def test_invalid_action_value_error(self):
        """Test invalid action bounds raise ValueError."""
        env = GymTronEnv()
        env.reset()
        with self.assertRaises(ValueError):
            env.step(np.array([-1, 0], dtype=np.int32))
        with self.assertRaises(ValueError):
            env.step(np.array([0, 4], dtype=np.int32))

    def test_gymnasium_compliance(self):
        """Test compliance with Gymnasium standard checks."""
        from gymnasium.utils.env_checker import check_env
        env = GymTronEnv()
        check_env(env, skip_render_check=True)

    def test_seeding_and_determinism(self):
        """Test seeding determinism."""
        env1 = GymTronEnv()
        env2 = GymTronEnv()
        
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        
        np.testing.assert_array_equal(obs1["observation"], obs2["observation"])

    def test_state_saving_and_restoring(self):
        """Test state save/restore options."""
        env = GymTronEnv()
        env.reset(seed=42)
        
        env.step(np.array([RIGHT, LEFT], dtype=np.int32))
        _, _, _, _, info = env.step(np.array([UP, DOWN], dtype=np.int32))
        saved_state = info["state"]
        
        new_env = GymTronEnv()
        new_env.reset(options={"state": saved_state})
        
        np.testing.assert_array_equal(new_env._grid, env._grid)
        self.assertEqual(new_env._p1_pos, env._p1_pos)
        self.assertEqual(new_env._p2_pos, env._p2_pos)
        self.assertEqual(new_env._p1_dir, env._p1_dir)
        self.assertEqual(new_env._p2_dir, env._p2_dir)
        self.assertEqual(new_env._steps, env._steps)
        np.testing.assert_array_equal(new_env._scores, env._scores)
        self.assertEqual(new_env._crash_pos, env._crash_pos)


if __name__ == "__main__":
    unittest.main()

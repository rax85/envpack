"""Tests for GymPlatformerEnv."""

import unittest
import numpy as np

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from envpack.envs.game_platformer.env import GymPlatformerEnv, IDLE, LEFT, RIGHT, JUMP


class TestGymPlatformerEnv(unittest.TestCase):
    """Tests for the GymPlatformerEnv Gymnasium environment."""

    def test_gym_compliance(self):
        """Test Gymnasium compliance using check_env."""
        env = GymPlatformerEnv()
        check_env(env, skip_render_check=True)

    def test_initial_state(self):
        """Test that the initial state is correct after reset."""
        env = GymPlatformerEnv()
        obs, info = env.reset()

        # Check observation structure
        self.assertIn("observation", obs)
        self.assertIn("valid_mask", obs)
        self.assertIn("level_progress", obs)

        # Check observation shapes and dtypes
        self.assertEqual(obs["observation"].shape, (300, 400, 3))
        self.assertEqual(obs["observation"].dtype, np.uint8)
        self.assertEqual(obs["valid_mask"].shape, (4,))
        self.assertEqual(obs["valid_mask"].dtype, np.int8)
        self.assertEqual(obs["level_progress"].shape, (1,))
        self.assertEqual(obs["level_progress"].dtype, np.float32)

        # Defaults
        self.assertEqual(env.score, 0)
        self.assertEqual(env.px, 50.0)
        self.assertEqual(env.py, 230.0)
        self.assertEqual(env.vx, 0.0)
        self.assertEqual(env.vy, 0.0)

    def test_movements_and_inertia(self):
        """Test horizontal movement and inertia drag."""
        env = GymPlatformerEnv()
        env.reset()

        # Step RIGHT: vx should increase
        env.step(RIGHT)
        self.assertGreater(env.vx, 0.0)
        x_before = env.px

        # Step IDLE: friction/drag should slow down vx
        env.step(IDLE)
        self.assertLess(env.vx, 0.5)  # friction slowed it down
        self.assertGreater(env.px, x_before)

        # Step LEFT: vx should decrease/become negative
        env.step(LEFT)
        self.assertLess(env.vx, 0.0)

    def test_jumping_and_falling(self):
        """Test gravity and jumping constraints."""
        env = GymPlatformerEnv()
        env.reset()

        # Player starts at y=230, which is on the ground (ground y is 260, player ph=24, so y=236).
        # Wait, since player y is 230 and gravity pulls down, player should fall onto ground.
        # Let's advance a few steps to let them land
        for _ in range(10):
            env.step(IDLE)
        self.assertTrue(env.on_ground)
        self.assertEqual(env.py, 260.0 - 24.0)  # y = 236

        # Now jump: vy should be negative (upwards)
        env.step(JUMP)
        self.assertLess(env.vy, 0.0)
        self.assertFalse(env.on_ground)

        # In mid-air, jump action should be masked out
        mask = env._get_valid_mask()
        self.assertEqual(mask[JUMP], 0)

        # Advance steps: gravity should pull player back down to ground
        for _ in range(60):
            env.step(IDLE)
        self.assertTrue(env.on_ground)

    def test_custom_state_inject(self):
        """Test custom state restoration."""
        env = GymPlatformerEnv()
        state = {
            "player_pos": (200.0, 100.0),
            "player_vel": (2.0, -3.0),
            "score": 40,
            "coins": [(240, 170), (700, 230)],
            "on_ground": False,
        }
        obs, info = env.reset(options={"state": state})
        self.assertEqual(env.px, 200.0)
        self.assertEqual(env.py, 100.0)
        self.assertEqual(env.vx, 2.0)
        self.assertEqual(env.vy, -3.0)
        self.assertEqual(env.score, 40)
        self.assertEqual(env.coins, {(240, 170), (700, 230)})
        self.assertFalse(env.on_ground)

    def test_solid_collisions(self):
        """Test collisions against platforms."""
        env = GymPlatformerEnv()
        # Place player right above Platform A: x=200 to 280, y=200.
        # Player is pw=16, ph=24.
        state = {
            "player_pos": (220.0, 172.0),
            "player_vel": (0.0, 5.0), # moving down fast
            "score": 0,
            "coins": [],
        }
        env.reset(options={"state": state})

        env.step(IDLE)
        # Should collide vertically with Platform A (y=200). Player y becomes 200 - 24 = 176.
        self.assertEqual(env.py, 176.0)
        self.assertTrue(env.on_ground)
        self.assertEqual(env.vy, 0.0)

    def test_hazard_spikes(self):
        """Test spikes collision causing death and reset."""
        env = GymPlatformerEnv()
        # Place player right above spike pit: x=300 to 380, y=280.
        state = {
            "player_pos": (320.0, 270.0),
            "player_vel": (0.0, 0.0),
            "score": 50,
            "coins": [],
        }
        env.reset(options={"state": state})

        obs, reward, term, trunc, info = env.step(IDLE)
        # Should collide with spikes, die (penalty -50), and reset to start
        self.assertEqual(reward, -50.0 - 0.05)
        self.assertEqual(env.px, 50.0)
        self.assertEqual(env.py, 230.0)

    def test_coin_collection(self):
        """Test gold coin collection."""
        env = GymPlatformerEnv()
        # Place player right near a coin: Coin is at (240, 170).
        # Platform A is at y=200, so player standing on Platform A is at y=176.
        state = {
            "player_pos": (230.0, 176.0),
            "player_vel": (0.0, 0.0),
            "score": 10,
            "coins": [(240, 170)],
        }
        env.reset(options={"state": state})

        # Step closer to the coin
        obs, reward, term, trunc, info = env.step(RIGHT)
        # Coin center is (240, 170), player center is (230 + 8, 176 + 12) = (238, 188) before move.
        # After move, player is at px ~ 230 + vx.
        # Let's verify coin was eaten
        self.assertEqual(env.score, 20)
        self.assertEqual(len(env.coins), 0)

    def test_win_condition(self):
        """Test flag reached to win level."""
        env = GymPlatformerEnv()
        # Place player near the flag (x = 750)
        state = {
            "player_pos": (745.0, 236.0),
            "player_vel": (0.0, 0.0),
            "score": 100,
            "coins": [],
        }
        env.reset(options={"state": state})

        # Move right multiple times to cross Flag at x = 750
        for _ in range(15):
            obs, reward, term, trunc, info = env.step(RIGHT)
            if term:
                break
        self.assertTrue(term)
        self.assertGreater(reward, 90.0)

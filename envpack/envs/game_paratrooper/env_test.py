"""Tests for GymParatrooperEnv."""

import math
import unittest
import numpy as np

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from envpack.envs.game_paratrooper.env import GymParatrooperEnv


class TestGymParatrooperEnv(unittest.TestCase):
    """Tests for the GymParatrooperEnv Gymnasium environment."""

    def test_gym_compliance(self):
        """Test Gymnasium compliance using check_env."""
        env = GymParatrooperEnv()
        check_env(env, skip_render_check=True)

    def test_initial_state(self):
        """Test that the initial state is correct after reset."""
        env = GymParatrooperEnv()
        obs, info = env.reset()

        # Check observation structure
        self.assertIn("observation", obs)
        self.assertIn("valid_mask", obs)
        self.assertIn("score", obs)
        self.assertIn("landed_left", obs)
        self.assertIn("landed_right", obs)

        # Check observation shapes and dtypes
        self.assertEqual(obs["observation"].shape, (300, 400, 3))
        self.assertEqual(obs["observation"].dtype, np.uint8)
        self.assertEqual(obs["valid_mask"].shape, (4,))
        self.assertEqual(obs["valid_mask"].dtype, np.int8)
        self.assertEqual(obs["score"].shape, (1,))
        self.assertEqual(obs["score"].dtype, np.int32)
        self.assertEqual(obs["landed_left"].shape, (1,))
        self.assertEqual(obs["landed_left"].dtype, np.int32)
        self.assertEqual(obs["landed_right"].shape, (1,))
        self.assertEqual(obs["landed_right"].dtype, np.int32)

        # Check defaults
        self.assertEqual(env._score, 0)
        self.assertEqual(env._landed_left, 0)
        self.assertEqual(env._landed_right, 0)
        self.assertAlmostEqual(env._turret_angle, math.pi / 2.0)
        self.assertEqual(len(env._helicopters), 1)  # Spawns 1 initially

    def test_movements(self):
        """Test turret movements and constraints [0.1 * pi, 0.9 * pi]."""
        env = GymParatrooperEnv()
        env.reset()

        initial_angle = env._turret_angle

        # Action 0: Turn Left (counter-clockwise, angle increases)
        obs, reward, term, trunc, info = env.step(0)
        self.assertGreater(env._turret_angle, initial_angle)

        # Hold Turn Left to hit the ceiling limit of 0.9 * pi
        for _ in range(50):
            env.step(0)
        self.assertAlmostEqual(env._turret_angle, 0.9 * math.pi)

        # Action 1: Turn Right (clockwise, angle decreases)
        env.step(1)
        self.assertLess(env._turret_angle, 0.9 * math.pi)

        # Hold Turn Right to hit the floor limit of 0.1 * pi
        for _ in range(50):
            env.step(1)
        self.assertAlmostEqual(env._turret_angle, 0.1 * math.pi)

    def test_shooting_and_cooldown(self):
        """Test shooting mechanism and firing cooldown."""
        env = GymParatrooperEnv()
        # Custom state reset to avoid spawning random helicopters
        env.reset(options={"state": {"helicopters": []}})

        self.assertEqual(len(env._bullets), 0)

        # Action 2: Shoot
        env.step(2)
        self.assertEqual(len(env._bullets), 1)

        # Shoot again immediately (should be blocked by cooldown)
        env.step(2)
        self.assertEqual(len(env._bullets), 1)

        # Advance steps to cool down
        for _ in range(5):
            env.step(3)  # Stay

        # Shoot again (cooldown expired, should work)
        env.step(2)
        self.assertEqual(len(env._bullets), 2)

    def test_custom_state_inject(self):
        """Test that custom state injection sets state correctly."""
        env = GymParatrooperEnv()
        state = {
            "turret_angle": 0.4,
            "score": 100,
            "landed_left": 2,
            "landed_right": 1,
            "helicopters": [{"x": 100.0, "y": 50.0, "vx": 3.0}],
            "paratroopers": [{"x": 150.0, "y": 120.0, "vy": 1.0, "parachute_state": "open"}],
            "bullets": [{"x": 200.0, "y": 200.0, "vx": 0.0, "vy": -5.0}],
            "bombs": [{"x": 250.0, "y": 80.0, "vy": 3.0}],
        }
        obs, info = env.reset(options={"state": state})

        self.assertAlmostEqual(env._turret_angle, 0.4)
        self.assertEqual(env._score, 100)
        self.assertEqual(env._landed_left, 2)
        self.assertEqual(env._landed_right, 1)

        self.assertEqual(len(env._helicopters), 1)
        self.assertEqual(env._helicopters[0]["x"], 100.0)

        self.assertEqual(len(env._paratroopers), 1)
        self.assertEqual(env._paratroopers[0]["parachute_state"], "open")

        self.assertEqual(len(env._bullets), 1)
        self.assertEqual(env._bullets[0]["y"], 200.0)

        self.assertEqual(len(env._bombs), 1)
        self.assertEqual(env._bombs[0]["x"], 250.0)

    def test_collisions(self):
        """Test bullet collisions with other entities."""
        # 1. Bullet vs Helicopter collision
        env = GymParatrooperEnv()
        state = {
            "bullets": [{"x": 100.0, "y": 50.0, "vx": 0.0, "vy": 0.0}],
            "helicopters": [{"x": 100.0, "y": 50.0, "vx": 0.0}],
        }
        env.reset(options={"state": state})
        obs, reward, term, trunc, info = env.step(3)  # Stay
        self.assertEqual(len(env._helicopters), 0)
        self.assertEqual(len(env._bullets), 0)
        self.assertEqual(env._score, 10)
        self.assertEqual(reward, 10.0)

        # 2. Bullet vs Bomb collision
        state = {
            "bullets": [{"x": 120.0, "y": 80.0, "vx": 0.0, "vy": 0.0}],
            "bombs": [{"x": 120.0, "y": 80.0, "vy": 0.0}],
        }
        env.reset(options={"state": state})
        obs, reward, term, trunc, info = env.step(3)
        self.assertEqual(len(env._bombs), 0)
        self.assertEqual(len(env._bullets), 0)
        self.assertEqual(env._score, 15)
        self.assertEqual(reward, 15.0)

        # 3. Bullet vs Paratrooper body collision
        state = {
            "bullets": [{"x": 150.0, "y": 100.0, "vx": 0.0, "vy": 0.0}],
            "paratroopers": [{"x": 150.0, "y": 100.0, "vy": 0.0, "parachute_state": "closed"}],
        }
        env.reset(options={"state": state})
        obs, reward, term, trunc, info = env.step(3)
        self.assertEqual(len(env._paratroopers), 0)
        self.assertEqual(len(env._bullets), 0)
        self.assertEqual(env._score, 5)
        self.assertEqual(reward, 5.0)

        # 4. Bullet vs Parachute collision (destroys parachute, paratrooper falls fast)
        state = {
            "bullets": [{"x": 180.0, "y": 88.0, "vx": 0.0, "vy": 0.0}],
            "paratroopers": [{"x": 180.0, "y": 100.0, "vy": 1.0, "parachute_state": "open"}],
        }
        env.reset(options={"state": state})
        obs, reward, term, trunc, info = env.step(3)
        self.assertEqual(len(env._bullets), 0)
        self.assertEqual(len(env._paratroopers), 1)
        self.assertEqual(env._paratroopers[0]["parachute_state"], "destroyed")
        self.assertEqual(env._paratroopers[0]["vy"], 4.0)  # PARA_FAST_FALL_SPEED

    def test_landing_rules_and_game_over(self):
        """Test landing rules and terminal state for 4 paratroopers landed on one side."""
        env = GymParatrooperEnv()

        # Let's inject 3 landed paratroopers on the left, and 1 falling paratrooper about to land on the left.
        state = {
            "landed_left": 3,
            "landed_positions_left": [50.0, 70.0, 90.0],
            "paratroopers": [{"x": 60.0, "y": 269.0, "vy": 1.0, "parachute_state": "open"}],
            "helicopters": [],
        }
        env.reset(options={"state": state})
        self.assertEqual(env._landed_left, 3)

        obs, reward, term, trunc, info = env.step(3)
        self.assertEqual(env._landed_left, 4)
        self.assertTrue(term)
        self.assertEqual(reward, -50.0)

    def test_bomb_hit_turret_game_over(self):
        """Test that a bomb hitting the turret results in game over."""
        env = GymParatrooperEnv()

        # Place a bomb right above the turret (turret base is [185, 215] at y=270)
        state = {
            "bombs": [{"x": 200.0, "y": 268.0, "vy": 3.0}],
            "helicopters": [],
        }
        env.reset(options={"state": state})

        obs, reward, term, trunc, info = env.step(3)
        self.assertTrue(term)
        self.assertEqual(reward, -50.0)

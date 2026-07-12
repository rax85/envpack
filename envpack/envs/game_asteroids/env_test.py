"""Tests for GymAsteroidsEnv."""

import unittest
import numpy as np
import math

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from envpack.envs.game_asteroids.env import GymAsteroidsEnv, IDLE, ROTATE_LEFT, ROTATE_RIGHT, THRUST, SHOOT


class TestGymAsteroidsEnv(unittest.TestCase):
    """Tests for the GymAsteroidsEnv Gymnasium environment."""

    def test_gym_compliance(self):
        """Test Gymnasium compliance using check_env."""
        env = GymAsteroidsEnv()
        check_env(env, skip_render_check=True)

    def test_initial_state(self):
        """Test that the initial state is correct after reset."""
        env = GymAsteroidsEnv()
        obs, info = env.reset()

        # Check observation structure
        self.assertIn("observation", obs)
        self.assertIn("valid_mask", obs)
        self.assertIn("score", obs)
        self.assertIn("lives", obs)

        # Check observation shapes and dtypes
        self.assertEqual(obs["observation"].shape, (300, 400, 3))
        self.assertEqual(obs["observation"].dtype, np.uint8)
        self.assertEqual(obs["valid_mask"].shape, (5,))
        self.assertEqual(obs["valid_mask"].dtype, np.int8)
        self.assertEqual(obs["score"].shape, (1,))
        self.assertEqual(obs["score"].dtype, np.int32)
        self.assertEqual(obs["lives"].shape, (1,))
        self.assertEqual(obs["lives"].dtype, np.int32)

        # Defaults
        self.assertEqual(env.lives, 3)
        self.assertEqual(env.score, 0)
        self.assertEqual(env.ship_px, 200.0)
        self.assertEqual(env.ship_py, 150.0)
        self.assertEqual(env.ship_vx, 0.0)
        self.assertEqual(env.ship_vy, 0.0)
        self.assertEqual(len(env.asteroids), 4)

    def test_ship_movement_and_drag(self):
        """Test ship rotation, thrust, and drag friction."""
        env = GymAsteroidsEnv()
        env.reset()

        # Initial angle is -pi/2
        initial_angle = env.ship_angle

        # Rotate Left (increases/decreases counter-clockwise angle)
        env.step(ROTATE_LEFT)
        self.assertLess(env.ship_angle, initial_angle)

        # Rotate Right
        env.step(ROTATE_RIGHT)
        self.assertAlmostEqual(env.ship_angle, initial_angle)

        # Thrust
        env.step(THRUST)
        # Velocity should increase in the direction of the angle
        # ship_angle = -pi/2 (facing UP), so vy should decrease (become negative)
        self.assertLess(env.ship_vy, 0.0)
        
        vy_before = env.ship_vy
        # Step IDLE: drag should reduce velocity slightly (by 0.99)
        env.step(IDLE)
        self.assertGreater(env.ship_vy, vy_before)  # moving closer to 0

    def test_toroidal_wrap_around(self):
        """Test toroidal wrapping on screen boundaries."""
        env = GymAsteroidsEnv()
        
        # Ship near right edge (x=399), moving right
        state = {
            "lives": 3,
            "score": 0,
            "ship_pos": (399.0, 150.0),
            "ship_vel": (3.0, 0.0),
            "ship_angle": 0.0,
            "asteroids": [],
            "gems": [],
            "lasers": [],
        }
        env.reset(options={"state": state})

        env.step(IDLE)
        # New px = (399.0 + 3.0) % 400 = 2.0 (under drag, px = (399 + 3*0.99)%400 ~ 1.97)
        self.assertLess(env.ship_px, 5.0)
        self.assertGreater(env.ship_px, 0.0)

    def test_shooting_mechanics_and_cooldown(self):
        """Test shooting lasers and shooting cooldown."""
        env = GymAsteroidsEnv()
        state = {
            "lives": 3,
            "score": 0,
            "ship_pos": (200.0, 150.0),
            "ship_vel": (0.0, 0.0),
            "ship_angle": 0.0,
            "asteroids": [],
            "gems": [],
            "lasers": [],
        }
        env.reset(options={"state": state})

        # Firing 1
        env.step(SHOOT)
        self.assertEqual(len(env.lasers), 1)
        self.assertEqual(env.shoot_cooldown, 5)

        # Firing immediately again should be blocked by cooldown
        env.step(SHOOT)
        self.assertEqual(len(env.lasers), 1)

        # Cool down
        for _ in range(5):
            env.step(IDLE)

        # Fire again
        env.step(SHOOT)
        self.assertEqual(len(env.lasers), 2)

    def test_asteroid_splitting_and_gem_dropping(self):
        """Test lasers splitting large and medium asteroids, and dropping gems."""
        env = GymAsteroidsEnv()
        
        # Injected state:
        # Large asteroid at (100, 100), Laser at (100, 120) moving up (vx=0, vy=-10)
        state = {
            "lives": 3,
            "score": 0,
            "ship_pos": (200.0, 200.0),
            "ship_vel": (0.0, 0.0),
            "ship_angle": 0.0,
            "asteroids": [
                {"x": 100.0, "y": 100.0, "vx": 0.0, "vy": 0.0, "radius": 24.0},
            ],
            "gems": [],
            "lasers": [
                {"x": 100.0, "y": 120.0, "vx": 0.0, "vy": -8.0, "lifetime": 30},
            ],
        }
        env.reset(options={"state": state})

        # Advance step: laser moves to (100, 112).
        # Distance between laser and asteroid is 12 (<= LASER_RADIUS(2) + AST_RADIUS(24) = 26).
        # Should collide, split asteroid to 2 medium (radius 12), spawn 1 gem, increase score by 20.
        obs, reward, term, trunc, info = env.step(IDLE)
        self.assertEqual(env.score, 20)
        self.assertEqual(reward, 20.0 + 0.01)
        self.assertEqual(len(env.asteroids), 2)
        for ast in env.asteroids:
            self.assertEqual(ast["radius"], 12.0)
        self.assertEqual(len(env.gems), 1)

    def test_gem_collection(self):
        """Test ship collecting mineral gems."""
        env = GymAsteroidsEnv()
        
        # Injected state:
        # Gem at (205, 150), Ship at (200, 150). Distance is 5 (<= SHIP_RADIUS(8) + GEM_RADIUS(4) + 3 = 15).
        state = {
            "lives": 3,
            "score": 0,
            "ship_pos": (200.0, 150.0),
            "ship_vel": (0.0, 0.0),
            "ship_angle": 0.0,
            "asteroids": [],
            "gems": [
                {"x": 205.0, "y": 150.0, "vx": 0.0, "vy": 0.0, "lifetime": 100},
            ],
            "lasers": [],
        }
        env.reset(options={"state": state})

        # Step: Ship collects gem, gets +50 score.
        obs, reward, term, trunc, info = env.step(IDLE)
        self.assertEqual(env.score, 50)
        self.assertEqual(reward, 50.0 + 0.01)
        self.assertEqual(len(env.gems), 0)

    def test_ship_collision_and_death(self):
        """Test ship colliding with asteroid, losing life, and game over."""
        env = GymAsteroidsEnv()
        
        # 1. Collision with remaining lives: resets ship, pushes asteroid
        state = {
            "lives": 3,
            "score": 0,
            "ship_pos": (200.0, 150.0),
            "ship_vel": (0.0, 0.0),
            "ship_angle": 0.0,
            "asteroids": [
                {"x": 205.0, "y": 150.0, "vx": 0.0, "vy": 0.0, "radius": 12.0},
            ],
            "gems": [],
            "lasers": [],
        }
        env.reset(options={"state": state})

        # Step: Collide!
        obs, reward, term, trunc, info = env.step(IDLE)
        self.assertEqual(env.lives, 2)
        self.assertAlmostEqual(reward, -50.0 + 0.01)
        # Ship resets to center
        self.assertEqual(env.ship_px, 200.0)
        self.assertEqual(env.ship_py, 150.0)
        # Asteroid pushed > 80px away
        ast_dist = math.sqrt((env.asteroids[0]["x"] - 200.0) ** 2 + (env.asteroids[0]["y"] - 150.0) ** 2)
        self.assertGreaterEqual(ast_dist, 80.0)

        # 2. Collision with 1 life remaining: Game Over
        state = {
            "lives": 1,
            "score": 0,
            "ship_pos": (200.0, 150.0),
            "ship_vel": (0.0, 0.0),
            "ship_angle": 0.0,
            "asteroids": [
                {"x": 205.0, "y": 150.0, "vx": 0.0, "vy": 0.0, "radius": 12.0},
            ],
            "gems": [],
            "lasers": [],
        }
        env.reset(options={"state": state})

        obs, reward, term, trunc, info = env.step(IDLE)
        self.assertTrue(term)
        self.assertEqual(env.lives, 0)

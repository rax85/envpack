"""Tests for GymGravityDuelEnv."""

import copy
import math
import unittest
import numpy as np
import gymnasium as gym

from envpack.envs.game_gravity_duel.env import (
    GymGravityDuelEnv,
    PLAY_WIDTH,
    PLAY_HEIGHT,
    STAR_RADIUS,
    SHIP_RADIUS,
    ROTATION_SPEED,
    MAX_HP,
    CANVAS_SIZE
)

class TestGymGravityDuelEnv(unittest.TestCase):
    """Tests for the GymGravityDuelEnv environment."""

    def test_initial_state(self):
        """Test that the initial state is set up correctly."""
        env = GymGravityDuelEnv()
        obs, _ = env.reset()

        np.testing.assert_allclose(env._p1_pos, [80.0, 200.0])
        np.testing.assert_allclose(env._p2_pos, [320.0, 200.0])
        self.assertEqual(env._p1_hp, MAX_HP)
        self.assertEqual(env._p2_hp, MAX_HP)
        self.assertEqual(len(env._missiles), 0)

        # Check observation structure
        self.assertIn("observation", obs)
        self.assertIn("total_score", obs)
        self.assertEqual(obs["observation"].shape, (24,))
        np.testing.assert_array_equal(obs["total_score"], [0, 0])

    def test_thrust_and_rotation(self):
        """Test rotating and thrusting."""
        env = GymGravityDuelEnv()
        env.reset()

        # Action: [1, 2] -> P1 Rotate Left, P2 Rotate Right
        action = np.array([1, 2], dtype=np.int32)
        env.step(action)
        self.assertAlmostEqual(env._p1_angle, 2 * math.pi - ROTATION_SPEED)
        self.assertAlmostEqual(env._p2_angle, math.pi + ROTATION_SPEED)

        # Action: [3, 0] -> P1 thrust, P2 IDLE
        env.reset()
        # Angle is 0.0, thrusting increases vx
        action = np.array([3, 0], dtype=np.int32)
        env.step(action)
        self.assertGreater(env._p1_vel[0], 0.0)

    def test_gravity_star_attraction(self):
        """Test gravity pulling ship towards center star."""
        env = GymGravityDuelEnv()
        env.reset()

        # Place P1 at x=150, y=200 (left of star at 200, 200)
        # Velocity starts at 0, should accelerate towards right (positive x direction)
        state = env._get_state()
        state["p1_pos"] = [150.0, 200.0]
        state["p1_vel"] = [0.0, 0.0]
        env.reset(options={"state": state})

        env.step(np.array([0, 0], dtype=np.int32))
        self.assertGreater(env._p1_vel[0], 0.0)
        self.assertEqual(env._p1_vel[1], 0.0)

    def test_wrap_around(self):
        """Test wrapping around screen boundaries."""
        env = GymGravityDuelEnv()
        env.reset()

        # Place P1 at x=399, moving right
        state = env._get_state()
        state["p1_pos"] = [399.0, 200.0]
        state["p1_vel"] = [5.0, 0.0]
        env.reset(options={"state": state})

        # Apply gravity pull from (200,200) which would pull left (negative x)
        # But velocity 5.0 is large enough that next x exceeds 400
        env.step(np.array([0, 0], dtype=np.int32))
        self.assertLess(env._p1_pos[0], 200.0) # wrapped around to 399 + vel - gravity_accel

    def test_star_collision(self):
        """Test collision with gravity star causing death/respawn."""
        env = GymGravityDuelEnv()
        env.reset()

        # Place P1 inside the gravity star (star center at 200, 200, radius 20)
        state = env._get_state()
        state["p1_pos"] = [205.0, 200.0]
        env.reset(options={"state": state})

        obs, reward, terminated, truncated, info = env.step(np.array([0, 0], dtype=np.int32))
        # P1 should die, opponent P2 gets +1 score, P1 respawns at (80, 200) with MAX_HP
        self.assertEqual(env._scores[1], 1)
        self.assertEqual(env._p1_hp, MAX_HP)
        np.testing.assert_allclose(env._p1_pos, [80.0, 200.0])
        self.assertEqual(reward, -3.0) # penalty for hitting star

    def test_firing_missile_and_hit(self):
        """Test missile firing, trajectory, and hitting opponent."""
        env = GymGravityDuelEnv()
        env.reset()

        # Action: [4, 0] -> P1 fire missile, P2 IDLE
        action = np.array([4, 0], dtype=np.int32)
        env.step(action)
        self.assertEqual(len(env._missiles), 1)
        m = env._missiles[0]
        self.assertEqual(m["owner"], 0)

        # Let's test hit: place a missile right next to P2 (320, 200)
        state = env._get_state()
        state["missiles"] = [{
            "pos": [318.0, 200.0],
            "vel": [5.0, 0.0],
            "owner": 0,
            "lifetime": 100,
            "trail": []
        }]
        env.reset(options={"state": state})

        obs, reward, terminated, truncated, info = env.step(np.array([0, 0], dtype=np.int32))
        self.assertEqual(env._p2_hp, MAX_HP - 1)
        self.assertEqual(len(env._missiles), 0)
        self.assertEqual(reward, 1.0) # reward +1.0 for P1 damaging P2

    def test_state_saving_and_restoring(self):
        """Test state save/restore options."""
        env = GymGravityDuelEnv()
        env.reset(seed=42)

        env.step(np.array([3, 4], dtype=np.int32))
        _, _, _, _, info = env.step(np.array([1, 0], dtype=np.int32))
        saved_state = info["state"]

        new_env = GymGravityDuelEnv()
        new_env.reset(options={"state": saved_state})

        np.testing.assert_allclose(new_env._p1_pos, env._p1_pos)
        np.testing.assert_allclose(new_env._p1_vel, env._p1_vel)
        self.assertEqual(new_env._p1_angle, env._p1_angle)
        self.assertEqual(new_env._p1_hp, env._p1_hp)
        np.testing.assert_allclose(new_env._p2_pos, env._p2_pos)
        np.testing.assert_allclose(new_env._p2_vel, env._p2_vel)
        self.assertEqual(new_env._p2_angle, env._p2_angle)
        self.assertEqual(new_env._p2_hp, env._p2_hp)
        np.testing.assert_array_equal(new_env._scores, env._scores)
        self.assertEqual(new_env._steps, env._steps)

    def test_rendering(self):
        """Test render output dimensions."""
        env = GymGravityDuelEnv()
        env.reset()
        img = env.render()
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (CANVAS_SIZE[1], CANVAS_SIZE[0], 3))

    def test_gymnasium_compliance(self):
        """Test Gymnasium standard checks."""
        from gymnasium.utils.env_checker import check_env
        env = GymGravityDuelEnv()
        check_env(env, skip_render_check=True)

if __name__ == "__main__":
    unittest.main()

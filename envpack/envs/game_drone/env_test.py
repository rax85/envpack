"""Tests for the GymDroneEnv environment."""

import unittest
import numpy as np

from envpack.envs.game_drone.env import GymDroneEnv


class TestGymDroneEnv(unittest.TestCase):
    """Tests for the GymDroneEnv environment."""

    def test_initial_state(self):
        """Test that the initial state of the drone environment is correct."""
        env = GymDroneEnv()
        obs, _ = env.reset()

        # Altitude should be 0, fuel full
        self.assertEqual(env._pos[1], 0.0)
        self.assertEqual(env._fuel, 200.0)
        self.assertEqual(env._step_count, 0)
        self.assertFalse(env._crashed)
        self.assertFalse(env._landed)

        # Observation checks
        self.assertIn("observation", obs)
        self.assertIn("valid_mask", obs)
        self.assertEqual(obs["observation"].shape, (12,))
        np.testing.assert_array_equal(obs["valid_mask"], [1, 1, 1, 1])

    def test_flight_mechanics(self):
        """Test that throttle application produces velocity changes."""
        env = GymDroneEnv()
        env.reset()

        # Give maximum throttle (action = [1.0, 0.0, 0.0, 0.0])
        action = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Step a few times to build velocity
        for _ in range(5):
            obs, reward, terminated, truncated, _ = env.step(action)

        # Check velocity increase along x/z due to runway heading angle
        airspeed = np.linalg.norm(env._vel)
        self.assertGreater(airspeed, 0.0)
        self.assertFalse(terminated)

    def test_terrain_height(self):
        """Test that procedural terrain height returns reasonable positive values."""
        env = GymDroneEnv()
        h = env._get_terrain_height(1000.0, 1000.0)
        self.assertGreaterEqual(h, 0.0)

        # Runways should be flat (0.0 height)
        h1 = env._get_terrain_height(200.0, 200.0)
        h2 = env._get_terrain_height(1600.0, 1600.0)
        self.assertAlmostEqual(h1, 0.0, places=2)
        self.assertAlmostEqual(h2, 0.0, places=2)

    def test_gymnasium_compliance(self):
        """Test that the environment complies with Gymnasium specs."""
        from gymnasium.utils.env_checker import check_env
        env = GymDroneEnv()
        check_env(env, skip_render_check=True)

    def test_state_saving_and_restoring(self):
        """Test saving and restoring state works correctly."""
        env = GymDroneEnv()
        env.reset()

        env.step(np.array([1.0, 0.2, 0.0, 0.0], dtype=np.float32))
        _, _, _, _, info = env.step(np.array([1.0, 0.0, 0.1, 0.0], dtype=np.float32))

        saved_state = info["state"]

        new_env = GymDroneEnv()
        new_env.reset(options={"state": saved_state})

        np.testing.assert_array_equal(new_env._pos, env._pos)
        np.testing.assert_array_equal(new_env._vel, env._vel)
        self.assertEqual(new_env._pitch, env._pitch)
        self.assertEqual(new_env._roll, env._roll)
        self.assertEqual(new_env._yaw, env._yaw)
        self.assertEqual(new_env._fuel, env._fuel)


if __name__ == "__main__":
    unittest.main()

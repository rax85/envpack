"""Tests for the GymSpaceInvadersEnv environment."""

import unittest
import numpy as np

from envpack.envs.game_space_invaders.env import GymSpaceInvadersEnv


class TestGymSpaceInvadersEnv(unittest.TestCase):
    """Tests for the GymSpaceInvadersEnv environment."""

    def test_initial_state(self):
        """Test that the initial state of the environment is set up correctly."""
        env = GymSpaceInvadersEnv()
        obs, _ = env.reset()

        self.assertEqual(env._lives, 3)
        self.assertEqual(env._score, 0)
        self.assertEqual(len(env._invaders), 40)
        self.assertEqual(len(env._lasers), 0)
        self.assertEqual(len(env._enemy_bullets), 0)

        # Check observation structure
        self.assertIn("observation", obs)
        self.assertIn("valid_mask", obs)
        self.assertIn("score", obs)
        self.assertIn("lives", obs)

        # Grid checks
        self.assertEqual(obs["observation"].shape, (300, 400, 3))
        self.assertEqual(obs["score"][0], 0)
        self.assertEqual(obs["lives"][0], 3)

        # All moves initially valid
        np.testing.assert_array_equal(obs["valid_mask"], [1, 1, 1, 1])

    def test_movement(self):
        """Test moving left and right changes coordinates."""
        env = GymSpaceInvadersEnv()
        env.reset()
        initial_x = env._player_x

        # Move LEFT
        obs, reward, terminated, truncated, _ = env.step(1)
        self.assertLess(env._player_x, initial_x)
        self.assertFalse(terminated)

        # Move RIGHT
        initial_x = env._player_x
        obs, reward, terminated, truncated, _ = env.step(2)
        self.assertGreater(env._player_x, initial_x)

    def test_shooting_cooldown(self):
        """Test that shooting sets cooldown and disables shoot in valid_mask."""
        env = GymSpaceInvadersEnv()
        env.reset()

        # Fire laser
        obs, reward, terminated, truncated, _ = env.step(3)
        self.assertEqual(len(env._lasers), 1)
        self.assertEqual(env._cooldown, 12)
        # Laser fire action (3) should be disabled in the mask during cooldown
        self.assertEqual(obs["valid_mask"][3], 0)

    def test_invaders_cleared(self):
        """Test that clearing all invaders resets them and awards points."""
        env = GymSpaceInvadersEnv()
        env.reset()

        # Mark all invaders as dead
        for inv in env._invaders:
            inv["alive"] = False

        # Stepping should trigger reset wave and points
        obs, reward, terminated, truncated, _ = env.step(0)
        self.assertEqual(env._score, 1000)
        self.assertEqual(len([i for i in env._invaders if i["alive"]]), 40)

    def test_gymnasium_compliance(self):
        """Test that the environment complies with Gymnasium specs."""
        from gymnasium.utils.env_checker import check_env
        env = GymSpaceInvadersEnv()
        check_env(env, skip_render_check=True)

    def test_state_saving_and_restoring(self):
        """Test saving and restoring state works correctly."""
        env = GymSpaceInvadersEnv()
        env.reset()

        env.step(1)
        env.step(3)
        _, _, _, _, info = env.step(0)

        saved_state = info["state"]

        new_env = GymSpaceInvadersEnv()
        new_env.reset(options={"state": saved_state})

        self.assertEqual(new_env._player_x, env._player_x)
        self.assertEqual(new_env._score, env._score)
        self.assertEqual(new_env._lives, env._lives)
        self.assertEqual(len(new_env._lasers), len(env._lasers))
        self.assertEqual(new_env._cooldown, env._cooldown)


if __name__ == "__main__":
    unittest.main()

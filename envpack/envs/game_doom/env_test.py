"""Tests for GymDoomEnv environment."""

import copy
import unittest
import numpy as np

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from envpack.envs.game_doom.env import GymDoomEnv
import envpack.envs.game_doom.env as doom_env


class TestGymDoomEnv(unittest.TestCase):
    """Tests for the GymDoomEnv environment."""

    def test_initial_state(self):
        """Test that the initial state of the environment is set up correctly."""
        env = GymDoomEnv()
        obs, _ = env.reset()

        # Check default values
        self.assertEqual(env.player_health, 100)
        self.assertEqual(env.player_ammo, 20)
        self.assertEqual(env.player_score, 0)
        self.assertEqual(env.player_x, 1.5)
        self.assertEqual(env.player_y, 1.5)
        self.assertEqual(env.player_angle, 0.0)

        # Check observation structure
        self.assertIn("observation", obs)
        self.assertIn("valid_mask", obs)
        self.assertIn("health", obs)
        self.assertIn("ammo", obs)
        self.assertIn("score", obs)

        self.assertEqual(obs["observation"].shape, (240, 320, 3))
        self.assertEqual(obs["observation"].dtype, np.uint8)
        np.testing.assert_array_equal(obs["valid_mask"], [1, 1, 1, 1, 1])
        self.assertEqual(obs["health"][0], 100)
        self.assertEqual(obs["ammo"][0], 20)
        self.assertEqual(obs["score"][0], 0)

    def test_movement(self):
        """Test moving and turning."""
        env = GymDoomEnv()
        env.reset()

        # Turn left
        _, _, _, _, _ = env.step(doom_env.TURN_LEFT)
        self.assertLess(env.player_angle, 0.0)

        # Reset and turn right
        env.reset()
        _, _, _, _, _ = env.step(doom_env.TURN_RIGHT)
        self.assertGreater(env.player_angle, 0.0)

        # Move forward (from 1.5, 1.5, angle 0.0, moving right, should increase player_x)
        env.reset()
        px_before = env.player_x
        _, _, _, _, _ = env.step(doom_env.MOVE_FORWARD)
        self.assertGreater(env.player_x, px_before)

        # Move backward (should decrease player_x)
        px_before = env.player_x
        _, _, _, _, _ = env.step(doom_env.MOVE_BACKWARD)
        self.assertLess(env.player_x, px_before)

    def test_shooting_and_killing(self):
        """Test shooting and killing enemies with custom state injection."""
        env = GymDoomEnv()
        
        # Inject state with one enemy directly in front of the player
        custom_state = {
            "player_x": 1.5,
            "player_y": 1.5,
            "player_angle": 0.0,
            "health": 100,
            "ammo": 5,
            "score": 0,
            "enemies": [
                {"x": 2.5, "y": 1.5, "health": 100, "status": "idle"}
            ],
            "items": [],
            "gun_frame": 0,
            "total_steps": 0,
        }
        env.reset(options={"state": custom_state})

        # Shoot once (reduces enemy health by 50)
        obs, reward, terminated, truncated, info = env.step(doom_env.SHOOT)
        self.assertEqual(env.enemies[0]["health"], 50)
        self.assertEqual(env.enemies[0]["status"], "alert")
        self.assertEqual(env.player_ammo, 4)
        self.assertEqual(env.gun_frame, 1)  # Firing flash
        self.assertFalse(terminated)

        # Shoot again (kills enemy, enemy health -> 0)
        obs, reward, terminated, truncated, info = env.step(doom_env.SHOOT)
        self.assertEqual(env.enemies[0]["health"], 0)
        self.assertEqual(env.enemies[0]["status"], "dead")
        self.assertEqual(env.player_score, 100)
        self.assertTrue(terminated)  # Terminated because all active enemies are dead
        self.assertGreater(reward, 5.0)

    def test_no_ammo_shooting(self):
        """Test that shooting without ammo fails gracefully."""
        env = GymDoomEnv()
        custom_state = {
            "player_x": 1.5,
            "player_y": 1.5,
            "player_angle": 0.0,
            "health": 100,
            "ammo": 0,
            "score": 0,
            "enemies": [
                {"x": 2.5, "y": 1.5, "health": 100, "status": "idle"}
            ],
            "items": [],
            "gun_frame": 0,
            "total_steps": 0,
        }
        env.reset(options={"state": custom_state})

        # Shoot action (should not deal damage since ammo is 0)
        obs, reward, terminated, truncated, info = env.step(doom_env.SHOOT)
        self.assertEqual(env.enemies[0]["health"], 100)
        self.assertEqual(env.player_ammo, 0)
        self.assertEqual(env.gun_frame, 0)
        self.assertLess(reward, 0.0)

    def test_item_pickup(self):
        """Test automatic pickup of health and ammo packs."""
        env = GymDoomEnv()
        custom_state = {
            "player_x": 1.5,
            "player_y": 1.5,
            "player_angle": 0.0,
            "health": 50,
            "ammo": 10,
            "score": 0,
            "enemies": [
                {"x": 10.0, "y": 10.0, "health": 100, "status": "idle"}  # far away
            ],
            "items": [
                {"x": 1.7, "y": 1.5, "type": "health", "active": True},
                {"x": 1.5, "y": 1.7, "type": "ammo", "active": True},
            ],
            "gun_frame": 0,
            "total_steps": 0,
        }
        env.reset(options={"state": custom_state})

        # Step in place (e.g. Turn Left) to trigger proximity pickup checks
        obs, reward, terminated, truncated, info = env.step(doom_env.TURN_LEFT)
        
        # Verify items are picked up
        self.assertEqual(env.player_health, 75)  # 50 + 25
        self.assertEqual(env.player_ammo, 20)    # 10 + 10
        self.assertFalse(env.items[0]["active"])
        self.assertFalse(env.items[1]["active"])
        self.assertGreater(reward, 0.0)

    def test_enemy_ai_and_damage(self):
        """Test that alert enemies move towards player and attack."""
        env = GymDoomEnv()
        custom_state = {
            "player_x": 1.5,
            "player_y": 1.5,
            "player_angle": 0.0,
            "health": 100,
            "ammo": 20,
            "score": 0,
            "enemies": [
                {"x": 2.2, "y": 1.5, "health": 100, "status": "alert"}  # close to player, alert
            ],
            "items": [],
            "gun_frame": 0,
            "total_steps": 0,
        }
        env.reset(options={"state": custom_state})

        # Distance is hypot(2.2 - 1.5, 0.0) = 0.7.
        # Since distance < 0.8, the enemy should attack the player.
        obs, reward, terminated, truncated, info = env.step(doom_env.TURN_LEFT)
        
        self.assertEqual(env.player_health, 95)  # took 5 damage
        self.assertLess(reward, 0.0)             # received negative reward for damage

    def test_rendering(self):
        """Test that the render function returns the correct shape and format."""
        env = GymDoomEnv()
        env.reset()
        
        img = env.render()
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (240, 320, 3))
        self.assertEqual(img.dtype, np.uint8)

    def test_gymnasium_compliance(self):
        """Test Gymnasium compliance using check_env."""
        env = GymDoomEnv()
        check_env(env, skip_render_check=True)

    def test_seeding_and_determinism(self):
        """Test that environment seeding produces deterministic resets and steps."""
        env1 = GymDoomEnv()
        env2 = GymDoomEnv()

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        np.testing.assert_array_equal(obs1["observation"], obs2["observation"])

        # Take same steps and verify identical states
        for action in [doom_env.TURN_LEFT, doom_env.MOVE_FORWARD, doom_env.SHOOT]:
            o1, r1, term1, trunc1, _ = env1.step(action)
            o2, r2, term2, trunc2, _ = env2.step(action)
            np.testing.assert_array_equal(o1["observation"], o2["observation"])
            self.assertEqual(r1, r2)
            self.assertEqual(term1, term2)
            self.assertEqual(trunc1, trunc2)

    def test_invalid_action(self):
        """Test that invalid action values raise a ValueError."""
        env = GymDoomEnv()
        env.reset()
        with self.assertRaises(ValueError):
            env.step(9)


if __name__ == "__main__":
    unittest.main()

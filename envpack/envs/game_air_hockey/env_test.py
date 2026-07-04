"""Tests for GymAirHockeyEnv."""

import copy
import unittest
import numpy as np

from envpack.envs.game_air_hockey import env as air_hockey_env
from envpack.envs.game_air_hockey.env import GymAirHockeyEnv, CANVAS_SIZE, TABLE_WIDTH, TABLE_HEIGHT, GOAL_WIDTH, MALLET_RADIUS, PUCK_RADIUS


class TestGymAirHockeyEnv(unittest.TestCase):
    """Tests for the GymAirHockeyEnv environment."""

    def test_initial_state(self):
        """Test that the initial state is set up correctly."""
        env = GymAirHockeyEnv()
        obs, _ = env.reset()

        # Check default positions
        np.testing.assert_allclose(env._p1_pos, [TABLE_WIDTH / 2, TABLE_HEIGHT * 0.75])
        np.testing.assert_allclose(env._p2_pos, [TABLE_WIDTH / 2, TABLE_HEIGHT * 0.25])
        np.testing.assert_allclose(env._puck_pos, [TABLE_WIDTH / 2, TABLE_HEIGHT / 2])

        # Check observation structure
        self.assertIn("observation", obs)
        self.assertIn("total_score", obs)
        self.assertEqual(obs["observation"].shape, (12,))
        np.testing.assert_array_equal(obs["total_score"], [0, 0])

    def test_step_physics_updates(self):
        """Test mallet movements and boundary clamping."""
        env = GymAirHockeyEnv()
        env.reset()

        # Move P1 and P2 down/up
        action = np.array([[0.0, 1.0], [0.0, -1.0]], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        # Check that mallets moved
        self.assertGreater(env._p1_pos[1], TABLE_HEIGHT * 0.75)
        self.assertLess(env._p2_pos[1], TABLE_HEIGHT * 0.25)

        # Move off board to verify clamping
        action_large = np.array([[10.0, 10.0], [-10.0, -10.0]], dtype=np.float32)
        env.step(action_large)
        
        # Check boundary clamps
        self.assertLessEqual(env._p1_pos[0], TABLE_WIDTH - MALLET_RADIUS)
        self.assertGreaterEqual(env._p2_pos[0], MALLET_RADIUS)

    def test_puck_wall_bounces(self):
        """Test puck bouncing off left/right side walls and solid horizontal boundaries."""
        env = GymAirHockeyEnv()
        env.reset()

        # 1. Left side wall bounce
        env._puck_pos = np.array([PUCK_RADIUS + 2.0, 100.0], dtype=np.float32)
        env._puck_vel = np.array([-5.0, 0.0], dtype=np.float32)
        env.step(np.zeros((2, 2), dtype=np.float32))
        self.assertEqual(env._puck_pos[0], PUCK_RADIUS)
        self.assertGreater(env._puck_vel[0], 0.0)

        # 2. Right side wall bounce
        env._puck_pos = np.array([TABLE_WIDTH - PUCK_RADIUS - 2.0, 100.0], dtype=np.float32)
        env._puck_vel = np.array([5.0, 0.0], dtype=np.float32)
        env.step(np.zeros((2, 2), dtype=np.float32))
        self.assertEqual(env._puck_pos[0], TABLE_WIDTH - PUCK_RADIUS)
        self.assertLess(env._puck_vel[0], 0.0)

        # 3. Solid top wall bounce (outside goal coordinates)
        env._puck_pos = np.array([10.0, PUCK_RADIUS + 2.0], dtype=np.float32)
        env._puck_vel = np.array([0.0, -5.0], dtype=np.float32)
        env.step(np.zeros((2, 2), dtype=np.float32))
        self.assertEqual(env._puck_pos[1], PUCK_RADIUS)
        self.assertGreater(env._puck_vel[1], 0.0)

        # 4. Solid bottom wall bounce (outside goal coordinates)
        env._puck_pos = np.array([10.0, TABLE_HEIGHT - PUCK_RADIUS - 2.0], dtype=np.float32)
        env._puck_vel = np.array([0.0, 5.0], dtype=np.float32)
        env.step(np.zeros((2, 2), dtype=np.float32))
        self.assertEqual(env._puck_pos[1], TABLE_HEIGHT - PUCK_RADIUS)
        self.assertLess(env._puck_vel[1], 0.0)

    def test_goal_scoring(self):
        """Test scoring goals and resetting puck."""
        # 1. P1 scores (top goal)
        env = GymAirHockeyEnv()
        env.reset()
        # Place puck right before top goal
        env._puck_pos = np.array([TABLE_WIDTH / 2, PUCK_RADIUS + 1.0], dtype=np.float32)
        env._puck_vel = np.array([0.0, -3.0], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(np.zeros((2, 2), dtype=np.float32))
        self.assertEqual(env._scores[0], 1)
        self.assertEqual(reward, 1.0)
        # Puck reset to center
        np.testing.assert_allclose(env._puck_pos, [TABLE_WIDTH / 2, TABLE_HEIGHT / 2])

        # 2. P2 scores (bottom goal)
        env = GymAirHockeyEnv()
        env.reset()
        # Place puck right before bottom goal
        env._puck_pos = np.array([TABLE_WIDTH / 2, TABLE_HEIGHT - PUCK_RADIUS - 1.0], dtype=np.float32)
        env._puck_vel = np.array([0.0, 3.0], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(np.zeros((2, 2), dtype=np.float32))
        self.assertEqual(env._scores[1], 1)
        self.assertEqual(reward, -1.0)
        np.testing.assert_allclose(env._puck_pos, [TABLE_WIDTH / 2, TABLE_HEIGHT / 2])

    def test_collisions(self):
        """Test physics calculations for mallet-puck collisions."""
        env = GymAirHockeyEnv()
        env.reset()

        # Place puck adjacent to P1 mallet, moving towards each other
        env._p1_pos = np.array([100.0, 350.0], dtype=np.float32)
        env._puck_pos = np.array([100.0, 350.0 - MALLET_RADIUS - PUCK_RADIUS + 2.0], dtype=np.float32)
        env._puck_vel = np.array([0.0, 4.0], dtype=np.float32)
        # Move mallet up
        action = np.array([[0.0, -1.0], [0.0, 0.0]], dtype=np.float32)
        env.step(action)

        # Puck should be pushed out and bounce away
        self.assertLess(env._puck_pos[1], 350.0 - MALLET_RADIUS - PUCK_RADIUS)
        self.assertLess(env._puck_vel[1], 0.0)

        # Zero distance collision resolution test
        env = GymAirHockeyEnv()
        env.reset()
        env._p1_pos = np.array([100.0, 350.0], dtype=np.float32)
        env._puck_pos = np.array([100.0, 350.0], dtype=np.float32)
        env._resolve_collision(env._p1_pos, env._p1_vel)
        self.assertNotEqual(env._puck_pos[0], 100.0)

    def test_game_over_7_goals(self):
        """Test game termination when a player reaches 7 goals."""
        env = GymAirHockeyEnv()
        env.reset()

        # Force P1 score to 6, then score
        env._scores = np.array([6, 0], dtype=np.int32)
        env._puck_pos = np.array([TABLE_WIDTH / 2, PUCK_RADIUS + 1.0], dtype=np.float32)
        env._puck_vel = np.array([0.0, -3.0], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(np.zeros((2, 2), dtype=np.float32))
        self.assertTrue(terminated)
        self.assertEqual(reward, 11.0) # Goal reward (1.0) + Win bonus (10.0)

        # Force P2 score to 6, then score
        env = GymAirHockeyEnv()
        env.reset()
        env._scores = np.array([0, 6], dtype=np.int32)
        env._puck_pos = np.array([TABLE_WIDTH / 2, TABLE_HEIGHT - PUCK_RADIUS - 1.0], dtype=np.float32)
        env._puck_vel = np.array([0.0, 3.0], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(np.zeros((2, 2), dtype=np.float32))
        self.assertTrue(terminated)
        self.assertEqual(reward, -11.0) # Goal reward (-1.0) + Loss penalty (-10.0)

    def test_rendering_and_close(self):
        """Test visuals rendering."""
        env = GymAirHockeyEnv()
        env.reset(seed=42)

        img = env.render()
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (CANVAS_SIZE[1], CANVAS_SIZE[0], 3))

        env.close()

    def test_font_loading_fallback(self):
        """Test font loading fallback."""
        from unittest.mock import patch
        with patch("matplotlib.font_manager.findfont", side_effect=Exception("Font error")):
            env = GymAirHockeyEnv()
            self.assertIsNotNone(env._title_font)

    def test_gymnasium_compliance(self):
        """Test compliance with Gymnasium standard checks."""
        from gymnasium.utils.env_checker import check_env
        env = GymAirHockeyEnv()
        check_env(env, skip_render_check=True)

    def test_seeding_and_determinism(self):
        """Test seeding determinism."""
        env1 = GymAirHockeyEnv()
        env2 = GymAirHockeyEnv()
        
        obs1, _ = env1.reset(seed=456)
        obs2, _ = env2.reset(seed=456)
        
        np.testing.assert_allclose(obs1["observation"], obs2["observation"])

    def test_state_saving_and_restoring(self):
        """Test state save/restore options."""
        env = GymAirHockeyEnv()
        env.reset(seed=42)
        
        env.step(np.array([[0.5, 0.5], [-0.5, -0.5]], dtype=np.float32))
        _, _, _, _, info = env.step(np.array([[0.0, -0.5], [0.5, 0.0]], dtype=np.float32))
        saved_state = info["state"]
        
        new_env = GymAirHockeyEnv()
        new_env.reset(options={"state": saved_state})
        
        np.testing.assert_allclose(new_env._p1_pos, env._p1_pos)
        np.testing.assert_allclose(new_env._p1_vel, env._p1_vel)
        np.testing.assert_allclose(new_env._p2_pos, env._p2_pos)
        np.testing.assert_allclose(new_env._p2_vel, env._p2_vel)
        np.testing.assert_allclose(new_env._puck_pos, env._puck_pos)
        np.testing.assert_allclose(new_env._puck_vel, env._puck_vel)
        np.testing.assert_array_equal(new_env._scores, env._scores)
        self.assertEqual(new_env._steps, env._steps)


if __name__ == "__main__":
    unittest.main()

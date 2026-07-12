"""Tests for GymArtilleryFortsEnv."""

import copy
import math
import unittest
import numpy as np
import gymnasium as gym

from envpack.envs.game_artillery_forts.env import (
    GymArtilleryFortsEnv,
    PLAY_WIDTH,
    PLAY_HEIGHT,
    MAX_HP,
    CANVAS_SIZE
)

class TestGymArtilleryFortsEnv(unittest.TestCase):
    """Tests for the GymArtilleryFortsEnv environment."""

    def test_initial_state(self):
        """Test that the initial state is set up correctly."""
        env = GymArtilleryFortsEnv()
        obs, _ = env.reset()

        self.assertEqual(env._p1_hp, MAX_HP)
        self.assertEqual(env._p2_hp, MAX_HP)
        self.assertEqual(len(env._shells), 0)

        # Check observation structure
        self.assertIn("observation", obs)
        self.assertIn("total_score", obs)
        self.assertEqual(obs["observation"].shape, (49,))
        np.testing.assert_array_equal(obs["total_score"], [0, 0])

    def test_aim_and_power_controls(self):
        """Test aiming and power adjust controls."""
        env = GymArtilleryFortsEnv()
        env.reset()

        # Action: [1, 2] -> P1 Aim Up, P2 Aim Down
        # P1 angle range is [-pi/2, 0.0], starting at -pi/4. Aim Up decreases angle.
        # P2 angle range is [-pi, -pi/2], starting at -3pi/4. Aim Down decreases angle.
        action = np.array([1, 2], dtype=np.int32)
        env.step(action)
        self.assertLess(env._p1_angle, -math.pi / 4)
        self.assertLess(env._p2_angle, -3 * math.pi / 4)

        # Action: [3, 4] -> P1 Power Up, P2 Power Down
        env.reset()
        action = np.array([3, 4], dtype=np.int32)
        env.step(action)
        self.assertGreater(env._p1_power, 6.0)
        self.assertLess(env._p2_power, 6.0)

    def test_shell_firing_and_trajectory(self):
        """Test shell firing and trajectory update."""
        env = GymArtilleryFortsEnv()
        env.reset()

        # Action: [5, 0] -> P1 Fire shell, P2 IDLE
        action = np.array([5, 0], dtype=np.int32)
        env.step(action)

        self.assertEqual(len(env._shells), 1)
        shell = env._shells[0]
        self.assertEqual(shell["owner"], 0)

        # Step again, check that velocity and position updated under gravity
        old_x, old_y = shell["pos"]
        old_vx, old_vy = shell["vel"]
        env.step(np.array([0, 0], dtype=np.int32))

        new_shell = env._shells[0]
        self.assertNotEqual(new_shell["pos"][0], old_x)
        # vy should increase under gravity
        self.assertGreater(new_shell["vel"][1], old_vy)

    def test_terrain_cratering(self):
        """Test cratering physics on terrain."""
        env = GymArtilleryFortsEnv()
        env.reset()

        # Place a shell heading straight down at x=100
        # Check initial terrain height at x=100
        initial_y = env._terrain[100]

        state = env._get_state()
        state["shells"] = [{
            "pos": [100.0, initial_y - 2.0],
            "vel": [0.0, 5.0],
            "owner": 0
        }]
        env.reset(options={"state": state})

        # Step to cause impact
        env.step(np.array([0, 0], dtype=np.int32))

        # Terrain height (y-coordinate) should be pushed down (greater y value) at x=100
        self.assertGreater(env._terrain[100], initial_y)

    def test_fort_falling(self):
        """Test fort falling when terrain underneath is lowered."""
        env = GymArtilleryFortsEnv()
        env.reset()

        # Let's lower the terrain at x=80 (where P1 fort sits)
        # Check initial fort y
        old_p1_y = env._p1_y

        state = env._get_state()
        state["terrain"][80] = old_p1_y + 10.0
        env.reset(options={"state": state})

        # Step to resolve updates
        env.step(np.array([0, 0], dtype=np.int32))

        # P1 Y should fall to match the new terrain height
        self.assertEqual(env._p1_y, old_p1_y + 10.0)

    def test_state_saving_and_restoring(self):
        """Test state save/restore options."""
        env = GymArtilleryFortsEnv()
        env.reset(seed=42)

        env.step(np.array([5, 5], dtype=np.int32))
        _, _, _, _, info = env.step(np.array([1, 2], dtype=np.int32))
        saved_state = info["state"]

        new_env = GymArtilleryFortsEnv()
        new_env.reset(options={"state": saved_state})

        np.testing.assert_allclose(new_env._terrain, env._terrain)
        self.assertEqual(new_env._p1_y, env._p1_y)
        self.assertEqual(new_env._p1_angle, env._p1_angle)
        self.assertEqual(new_env._p1_power, env._p1_power)
        self.assertEqual(new_env._p1_hp, env._p1_hp)
        self.assertEqual(new_env._p2_y, env._p2_y)
        self.assertEqual(new_env._p2_angle, env._p2_angle)
        self.assertEqual(new_env._p2_power, env._p2_power)
        self.assertEqual(new_env._p2_hp, env._p2_hp)
        self.assertEqual(new_env._wind_speed, env._wind_speed)
        np.testing.assert_array_equal(new_env._scores, env._scores)
        self.assertEqual(new_env._steps, env._steps)

    def test_rendering(self):
        """Test render canvas dimensions."""
        env = GymArtilleryFortsEnv()
        env.reset()
        img = env.render()
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (CANVAS_SIZE[1], CANVAS_SIZE[0], 3))

    def test_gymnasium_compliance(self):
        """Test Gymnasium standard checks."""
        from gymnasium.utils.env_checker import check_env
        env = GymArtilleryFortsEnv()
        check_env(env, skip_render_check=True)

if __name__ == "__main__":
    unittest.main()

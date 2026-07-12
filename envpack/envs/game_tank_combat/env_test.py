"""Tests for GymTankCombatEnv."""

import copy
import math
import unittest
import numpy as np
import gymnasium as gym

from envpack.envs.game_tank_combat.env import (
    GymTankCombatEnv,
    PLAY_WIDTH,
    PLAY_HEIGHT,
    TANK_RADIUS,
    BULLET_SPEED,
    ROTATION_SPEED,
    MAX_HP,
    CANVAS_SIZE
)

class TestGymTankCombatEnv(unittest.TestCase):
    """Tests for the GymTankCombatEnv environment."""

    def test_initial_state(self):
        """Test that the initial state is set up correctly."""
        env = GymTankCombatEnv()
        obs, _ = env.reset()

        np.testing.assert_allclose(env._p1_pos, [60.0, 60.0])
        np.testing.assert_allclose(env._p2_pos, [340.0, 340.0])
        self.assertEqual(env._p1_hp, MAX_HP)
        self.assertEqual(env._p2_hp, MAX_HP)
        self.assertEqual(len(env._bullets), 0)

        # Check observation structure
        self.assertIn("observation", obs)
        self.assertIn("total_score", obs)
        self.assertEqual(obs["observation"].shape, (18,))
        np.testing.assert_array_equal(obs["total_score"], [0, 0])

    def test_movement_and_rotation(self):
        """Test rotating and moving forward."""
        env = GymTankCombatEnv()
        env.reset()

        # Action: [1, 2] -> P1 Rotate Left, P2 Rotate Right
        action = np.array([1, 2], dtype=np.int32)
        env.step(action)
        self.assertAlmostEqual(env._p1_angle, 2 * math.pi - ROTATION_SPEED)
        self.assertAlmostEqual(env._p2_angle, math.pi + ROTATION_SPEED)

        # Action: [3, 0] -> P1 move forward, P2 IDLE
        env.reset()
        # Angle is 0.0, moving forward increases x
        action = np.array([3, 0], dtype=np.int32)
        env.step(action)
        self.assertGreater(env._p1_pos[0], 60.0)
        self.assertEqual(env._p1_pos[1], 60.0)

    def test_wall_collision_and_sliding(self):
        """Test wall collisions: cannot move through walls."""
        env = GymTankCombatEnv()
        env.reset()
        state = env._get_state()
        state["p1_pos"] = [53.0, 60.0]
        state["p1_angle"] = math.pi
        env.reset(options={"state": state})

        # Try to move forward (towards left wall)
        action = np.array([3, 0], dtype=np.int32)
        env.step(action)
        self.assertGreaterEqual(env._p1_pos[0], 52.0)

    def test_shooting_and_bullet_movement(self):
        """Test firing a bullet and its forward progress."""
        env = GymTankCombatEnv()
        env.reset()

        # Action: [4, 0] -> P1 shoot, P2 IDLE
        action = np.array([4, 0], dtype=np.int32)
        env.step(action)

        self.assertEqual(len(env._bullets), 1)
        bullet = env._bullets[0]
        self.assertEqual(bullet["owner"], 0)
        self.assertEqual(bullet["bounces"], 0)

        # Step again, check bullet position increased along x (since P1 angle is 0.0)
        old_bx = bullet["pos"][0]
        env.step(np.array([0, 0], dtype=np.int32))
        self.assertGreater(env._bullets[0]["pos"][0], old_bx)

    def test_bullet_bouncing(self):
        """Test bullet bouncing up to 2 times, and destroyed on 3rd bounce."""
        env = GymTankCombatEnv()
        env.reset()

        # Inject bullet heading straight for a wall
        # Let's place it at x=358, heading right towards wall starting at x=360
        state = env._get_state()
        state["bullets"] = [{
            "pos": [358.0, 60.0],
            "vel": [BULLET_SPEED, 0.0],
            "owner": 0,
            "bounces": 0
        }]
        env.reset(options={"state": state})

        # Step 1: Bullet bounces off right wall, velocity is reflected to negative
        env.step(np.array([0, 0], dtype=np.int32))
        self.assertEqual(len(env._bullets), 1)
        self.assertEqual(env._bullets[0]["bounces"], 1)
        self.assertLess(env._bullets[0]["vel"][0], 0.0)

        # Bounce 2: Let's move it near the left boundary at x=0
        state = env._get_state()
        state["bullets"] = [{
            "pos": [2.0, 60.0],
            "vel": [-BULLET_SPEED, 0.0],
            "owner": 0,
            "bounces": 1
        }]
        env.reset(options={"state": state})

        env.step(np.array([0, 0], dtype=np.int32))
        self.assertEqual(len(env._bullets), 1)
        self.assertEqual(env._bullets[0]["bounces"], 2)
        self.assertGreater(env._bullets[0]["vel"][0], 0.0)

        # Bounce 3: Hit outer left boundary again with 2 bounces already -> should be destroyed
        state = env._get_state()
        state["bullets"] = [{
            "pos": [2.0, 60.0],
            "vel": [-BULLET_SPEED, 0.0],
            "owner": 0,
            "bounces": 2
        }]
        env.reset(options={"state": state})

        env.step(np.array([0, 0], dtype=np.int32))
        self.assertEqual(len(env._bullets), 0)

    def test_damage_and_respawn(self):
        """Test bullet hitting a tank, causing damage and respawning when HP reaches 0."""
        env = GymTankCombatEnv()
        env.reset()

        # Inject bullet right next to P2 (at 340.0, 340.0)
        state = env._get_state()
        state["bullets"] = [{
            "pos": [330.0, 340.0],
            "vel": [BULLET_SPEED, 0.0],
            "owner": 0,
            "bounces": 0
        }]
        env.reset(options={"state": state})

        # Step to hit P2
        obs, reward, terminated, truncated, info = env.step(np.array([0, 0], dtype=np.int32))
        self.assertEqual(env._p2_hp, MAX_HP - 1)
        self.assertEqual(reward, 1.0) # P1 gets +1.0 reward for damaging P2
        self.assertEqual(len(env._bullets), 0)

        # Let's set P2's HP to 1 and trigger elimination
        state = env._get_state()
        state["p2_hp"] = 1
        state["bullets"] = [{
            "pos": [330.0, 340.0],
            "vel": [BULLET_SPEED, 0.0],
            "owner": 0,
            "bounces": 0
        }]
        env.reset(options={"state": state})

        obs, reward, terminated, truncated, info = env.step(np.array([0, 0], dtype=np.int32))
        # P2 should be eliminated, score for P1 increments to 1, P2 respawns back to MAX_HP
        self.assertEqual(env._scores[0], 1)
        self.assertEqual(env._p2_hp, MAX_HP)
        self.assertEqual(reward, 6.0) # damage (1.0) + kill (5.0) = 6.0

    def test_state_saving_and_restoring(self):
        """Test saving and restoring state."""
        env = GymTankCombatEnv()
        env.reset(seed=42)

        env.step(np.array([3, 4], dtype=np.int32))
        _, _, _, _, info = env.step(np.array([1, 0], dtype=np.int32))
        saved_state = info["state"]

        new_env = GymTankCombatEnv()
        new_env.reset(options={"state": saved_state})

        np.testing.assert_allclose(new_env._p1_pos, env._p1_pos)
        self.assertEqual(new_env._p1_angle, env._p1_angle)
        self.assertEqual(new_env._p1_hp, env._p1_hp)
        np.testing.assert_allclose(new_env._p2_pos, env._p2_pos)
        self.assertEqual(new_env._p2_angle, env._p2_angle)
        self.assertEqual(new_env._p2_hp, env._p2_hp)
        np.testing.assert_array_equal(new_env._scores, env._scores)
        self.assertEqual(new_env._steps, env._steps)

    def test_rendering(self):
        """Test render mode returning visual RGB frame."""
        env = GymTankCombatEnv()
        env.reset()
        img = env.render()
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (CANVAS_SIZE[1], CANVAS_SIZE[0], 3))

    def test_gymnasium_compliance(self):
        """Test Gymnasium environment standard verification."""
        from gymnasium.utils.env_checker import check_env
        env = GymTankCombatEnv()
        check_env(env, skip_render_check=True)

if __name__ == "__main__":
    unittest.main()

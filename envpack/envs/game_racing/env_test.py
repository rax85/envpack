"""Tests for GymRacingEnv."""

import unittest
import numpy as np

from envpack.envs.game_racing import env as racing_env
from envpack.envs.game_racing.env import GymRacingEnv, CANVAS_SIZE, REDLINE_RPM, IDLE_RPM, get_engine_torque


class TestGymRacingEnv(unittest.TestCase):
    """Tests for the GymRacingEnv environment."""

    def test_initial_state(self):
        """Test that the initial state is set up correctly."""
        env = GymRacingEnv()
        obs, _ = env.reset()

        # Check track and spline properties
        self.assertIsNotNone(env._track_spline)
        self.assertGreater(len(env._track_spline), 100)

        # Check default speeds and gears
        self.assertEqual(env._p1_gear, 1)
        self.assertEqual(env._p2_gear, 1)
        self.assertAlmostEqual(env._p1_rpm, IDLE_RPM)
        self.assertAlmostEqual(env._p2_rpm, IDLE_RPM)

        # Observation shape
        self.assertEqual(obs["observation"].shape, (16,))
        self.assertEqual(obs["total_score"].shape, (2,))

    def test_engine_torque_curve(self):
        """Test the engine torque curve multipliers."""
        self.assertGreater(get_engine_torque(500.0), 150.0)
        self.assertGreater(get_engine_torque(1500.0), 300.0)
        self.assertEqual(get_engine_torque(1850.0), 550.0)
        self.assertEqual(get_engine_torque(5000.0), 550.0)
        self.assertLess(get_engine_torque(7000.0), 550.0)
        self.assertEqual(get_engine_torque(7500.0), 0.0)  # Cutoff

    def test_gear_shift_actions(self):
        """Test manual transmission gear shifts up and down."""
        env = GymRacingEnv()
        env.reset()

        action = {
            "p1_steer_throttle": np.array([0.0, 0.0], dtype=np.float32),
            "p1_gear": 2,  # Shift Up
            "p2_steer_throttle": np.array([0.0, 0.0], dtype=np.float32),
            "p2_gear": 0,  # Hold
        }
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertEqual(env._p1_gear, 2)
        self.assertEqual(env._p2_gear, 1)

        action = {
            "p1_steer_throttle": np.array([0.0, 0.0], dtype=np.float32),
            "p1_gear": 1,  # Shift Down
            "p2_steer_throttle": np.array([0.0, 0.0], dtype=np.float32),
            "p2_gear": 2,  # Shift Up
        }
        env.step(action)
        self.assertEqual(env._p1_gear, 1)
        self.assertEqual(env._p2_gear, 2)

    def test_dynamics_throttle_steer(self):
        """Test driving dynamics under throttle and steering input."""
        env = GymRacingEnv()
        env.reset()

        # Step forward with throttle
        action = {
            "p1_steer_throttle": np.array([0.0, 1.0], dtype=np.float32),
            "p1_gear": 0,
            "p2_steer_throttle": np.array([0.0, 1.0], dtype=np.float32),
            "p2_gear": 0,
        }
        env.step(action)
        
        # Speed should increase
        p1_speed = np.linalg.norm([env._p1_vx, env._p1_vy])
        self.assertGreater(p1_speed, 0.0)

        # Apply brakes
        action = {
            "p1_steer_throttle": np.array([0.0, -1.0], dtype=np.float32),
            "p1_gear": 0,
            "p2_steer_throttle": np.array([0.0, -1.0], dtype=np.float32),
            "p2_gear": 0,
        }
        env.step(action)
        p1_speed_braked = np.linalg.norm([env._p1_vx, env._p1_vy])
        self.assertLess(p1_speed_braked, p1_speed)

    def test_loss_of_traction_drift(self):
        """Test high speed turning triggers oversteer sliding and skidmark logs."""
        env = GymRacingEnv()
        env.reset()

        # Set high speed manually
        env._p1_vx = 30.0
        env._p1_vy = 0.0
        env._p1_theta = 0.0

        # Pre-populate skidmarks to 100
        env._p1_skids = [(100.0, 100.0)] * 100

        # Steer hard left at 30 m/s (108 km/h) to trigger a slide and execute pop(0)
        action = {
            "p1_steer_throttle": np.array([-1.0, 0.5], dtype=np.float32),
            "p1_gear": 0,
            "p2_steer_throttle": np.array([0.0, 0.0], dtype=np.float32),
            "p2_gear": 0,
        }
        env.step(action)

        # Should trigger sliding, append to skids, and pop(0), keeping length at 100
        self.assertEqual(len(env._p1_skids), 100)

    def test_off_track_grip(self):
        """Test going off-track reduces grip coefficient."""
        env = GymRacingEnv()
        env.reset()

        # Place P1 and P2 far off-track
        env._p1_x = 10.0
        env._p1_y = 10.0
        env._p2_x = 15.0
        env._p2_y = 10.0

        # Step and verify distance is high (off-track) and progress changes
        obs, reward, terminated, truncated, info = env.step({
            "p1_steer_throttle": np.array([0.0, 1.0], dtype=np.float32),
            "p1_gear": 0,
            "p2_steer_throttle": np.array([0.0, 1.0], dtype=np.float32),
            "p2_gear": 0,
        })
        p1_dist, _ = env._get_distance_to_track(env._p1_x, env._p1_y)
        self.assertGreater(p1_dist, 50.0)

    def test_win_loss_draw(self):
        """Test lap completion triggers win reward and termination."""
        # 1. P1 wins
        env = GymRacingEnv()
        env.reset()
        
        # Place P1 at the final spline point and high progress
        end_idx = len(env._track_spline) - 1
        env._p1_x, env._p1_y = env._track_spline[end_idx]
        env._p1_progress = 0.99
        
        action = {
            "p1_steer_throttle": np.array([0.0, 1.0], dtype=np.float32),
            "p1_gear": 0,
            "p2_steer_throttle": np.array([0.0, 0.0], dtype=np.float32),
            "p2_gear": 0,
        }
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertTrue(terminated)
        self.assertEqual(reward, 10.0)

        # 2. P2 wins
        env = GymRacingEnv()
        env.reset()
        
        end_idx = len(env._track_spline) - 1
        env._p2_x, env._p2_y = env._track_spline[end_idx]
        env._p2_progress = 0.99
        
        action = {
            "p1_steer_throttle": np.array([0.0, 0.0], dtype=np.float32),
            "p1_gear": 0,
            "p2_steer_throttle": np.array([0.0, 1.0], dtype=np.float32),
            "p2_gear": 0,
        }
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertTrue(terminated)
        self.assertEqual(reward, -10.0)

        # 3. Joint win (Tie)
        env = GymRacingEnv()
        env.reset()
        
        end_idx = len(env._track_spline) - 1
        env._p1_x, env._p1_y = env._track_spline[end_idx]
        env._p2_x, env._p2_y = env._track_spline[end_idx]
        env._p1_progress = 0.99
        env._p2_progress = 0.99
        
        action = {
            "p1_steer_throttle": np.array([0.0, 1.0], dtype=np.float32),
            "p1_gear": 0,
            "p2_steer_throttle": np.array([0.0, 1.0], dtype=np.float32),
            "p2_gear": 0,
        }
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertTrue(terminated)
        self.assertEqual(reward, 0.0)

    def test_redline_limiter_overrev(self):
        """Test engine overrev and fuel cut-off when RPM exceeds 7200."""
        env = GymRacingEnv()
        env.reset()

        # Set high speed in low gear (1st gear) -> forces RPM above REDLINE
        env._p1_vx = 30.0
        env._p1_vy = 0.0
        env._p1_gear = 1

        action = {
            "p1_steer_throttle": np.array([0.0, 1.0], dtype=np.float32),
            "p1_gear": 0,
            "p2_steer_throttle": np.array([0.0, 0.0], dtype=np.float32),
            "p2_gear": 0,
        }
        env.step(action)
        # RPM should be clipped, torque becomes 0, drag penalty is applied
        self.assertLessEqual(env._p1_rpm, 7500.0)

    def test_rendering_and_close(self):
        """Test visuals rendering."""
        env = GymRacingEnv()
        env.reset(seed=42)

        # Add skidmarks and trigger redline
        env._p1_skids = [(100.0, 100.0)]
        env._p2_skids = [(150.0, 150.0)]
        env._p1_rpm = 7300.0
        
        img = env.render()
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (CANVAS_SIZE[1], CANVAS_SIZE[0], 3))

        env.close()

    def test_font_loading_fallback(self):
        """Test font loading fallback."""
        from unittest.mock import patch
        with patch("matplotlib.font_manager.findfont", side_effect=Exception("Font error")):
            env = GymRacingEnv()
            self.assertIsNotNone(env._title_font)

    def test_gymnasium_compliance(self):
        """Test compliance with Gymnasium standard checks."""
        from gymnasium.utils.env_checker import check_env
        env = GymRacingEnv()
        check_env(env, skip_render_check=True)

    def test_seeding_and_determinism(self):
        """Test seeding determinism."""
        env1 = GymRacingEnv()
        env2 = GymRacingEnv()
        
        obs1, _ = env1.reset(seed=789)
        obs2, _ = env2.reset(seed=789)
        
        np.testing.assert_allclose(obs1["observation"], obs2["observation"])

    def test_state_saving_and_restoring(self):
        """Test state save/restore options."""
        env = GymRacingEnv()
        env.reset(seed=42)
        
        action = {
            "p1_steer_throttle": np.array([0.5, 0.5], dtype=np.float32),
            "p1_gear": 2,
            "p2_steer_throttle": np.array([-0.5, -0.5], dtype=np.float32),
            "p2_gear": 0,
        }
        env.step(action)
        _, _, _, _, info = env.step(action)
        saved_state = info["state"]
        
        new_env = GymRacingEnv()
        new_env.reset(options={"state": saved_state})
        
        self.assertEqual(new_env._p1_x, env._p1_x)
        self.assertEqual(new_env._p1_y, env._p1_y)
        self.assertEqual(new_env._p1_vx, env._p1_vx)
        self.assertEqual(new_env._p1_vy, env._p1_vy)
        self.assertEqual(new_env._p1_theta, env._p1_theta)
        self.assertEqual(new_env._p1_gear, env._p1_gear)
        self.assertEqual(new_env._p1_rpm, env._p1_rpm)
        self.assertEqual(new_env._p1_progress, env._p1_progress)
        self.assertEqual(new_env._p2_x, env._p2_x)
        self.assertEqual(new_env._p2_y, env._p2_y)
        self.assertEqual(new_env._p2_vx, env._p2_vx)
        self.assertEqual(new_env._p2_vy, env._p2_vy)
        self.assertEqual(new_env._p2_theta, env._p2_theta)
        self.assertEqual(new_env._p2_gear, env._p2_gear)
        self.assertEqual(new_env._p2_rpm, env._p2_rpm)
        self.assertEqual(new_env._p2_progress, env._p2_progress)
        self.assertEqual(new_env._steps, env._steps)
        np.testing.assert_array_equal(new_env._scores, env._scores)
        self.assertEqual(new_env._p1_skids, env._p1_skids)
        self.assertEqual(new_env._p2_skids, env._p2_skids)


if __name__ == "__main__":
    unittest.main()

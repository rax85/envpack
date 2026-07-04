"""Tests for GymRaptorEnv."""

import copy
import unittest
import numpy as np

from envpack.envs.game_raptor import env as raptor_env
from envpack.envs.game_raptor.env import GymRaptorEnv, CANVAS_SIZE, GRID_ROWS, GRID_COLS, STAY, LEFT, RIGHT, UP, DOWN


class TestGymRaptorEnv(unittest.TestCase):
    """Tests for the GymRaptorEnv environment."""

    def test_initial_state(self):
        """Test that the initial state is set up correctly."""
        env = GymRaptorEnv()
        obs, _ = env.reset()

        self.assertEqual(env._player_pos, [GRID_ROWS - 2, GRID_COLS // 2])
        self.assertEqual(env._shield, 100)
        self.assertEqual(env._score, 0)
        self.assertEqual(env._money, 0)

        # Check observation structure
        self.assertIn("observation", obs)
        self.assertIn("valid_mask", obs)
        self.assertIn("total_score", obs)
        self.assertIn("shield", obs)

        self.assertEqual(obs["total_score"][0], 0)
        self.assertEqual(obs["shield"][0], 100)

    def test_valid_move_boundaries(self):
        """Test that the player stays within bounds and valid_mask aligns."""
        env = GymRaptorEnv()
        obs, _ = env.reset()

        # Place player at top-left of allowed boundaries (row 10, col 0)
        env._player_pos = [10, 0]
        obs = env._create_observation()
        # Cannot move LEFT or UP
        np.testing.assert_array_equal(obs["valid_mask"], [1, 0, 1, 0, 1])

        # Step LEFT -> should not move
        obs, reward, term, trunc, _ = env.step(LEFT)
        self.assertEqual(env._player_pos, [10, 0])

        # Place player at bottom-right of boundaries (row 19, col 14)
        env._player_pos = [19, GRID_COLS - 1]
        obs = env._create_observation()
        # Cannot move RIGHT or DOWN
        np.testing.assert_array_equal(obs["valid_mask"], [1, 1, 0, 1, 0])

        # Step RIGHT -> should not move
        env.step(RIGHT)
        self.assertEqual(env._player_pos, [19, GRID_COLS - 1])

    def test_shooting_and_laser_movement(self):
        """Test that lasers auto-fire and move upwards."""
        env = GymRaptorEnv()
        env.reset(seed=42)

        # Step three times to trigger laser firing
        env.step(STAY)
        env.step(STAY)
        env.step(STAY)
        
        self.assertEqual(len(env._lasers), 1)
        # Laser should have moved up 1 row by the step it was fired (fired at player_row - 1, then moved to player_row - 2)
        expected_r = env._player_pos[0] - 2
        self.assertEqual(env._lasers[0], [expected_r, env._player_pos[1]])

        # Step again -> laser moves up
        env.step(STAY)
        self.assertEqual(env._lasers[0][0], expected_r - 1)

    def test_enemy_spawning_and_movement(self):
        """Test that enemies spawn and move downwards."""
        env = GymRaptorEnv()
        env.reset(seed=42)

        # Force an enemy spawn
        env._enemies.append(([0, 5], raptor_env.ENEMY_BASIC))
        env._enemies.append(([1, 6], raptor_env.ENEMY_SHOOTER))

        # Steps to verify they move down (every 2 steps)
        env.step(STAY)
        env.step(STAY)

        self.assertEqual(env._enemies[0][0], [1, 5])
        self.assertEqual(env._enemies[1][0], [2, 6])

    def test_collisions(self):
        """Test laser-enemy, player-enemy, player-bullet, and player-coin collisions."""
        # 1. Laser vs Enemy
        env = GymRaptorEnv()
        env.reset()
        from unittest.mock import MagicMock
        env.np_random = MagicMock()
        env.np_random.random = MagicMock(return_value=0.1)
        env._enemies.append(([5, 5], raptor_env.ENEMY_BASIC))
        env._lasers.append(([6, 5]))
        
        # Step triggers collision processing
        env.step(STAY)
        self.assertEqual(len(env._enemies), 0)
        self.assertGreater(env._score, 0)
        self.assertGreater(env._money, 0)

        # 2. Player vs Enemy Ship
        env = GymRaptorEnv()
        env.reset()
        env._player_pos = [15, 5]
        env._enemies.append(([15, 5], raptor_env.ENEMY_BASIC))
        
        env.step(STAY)
        self.assertEqual(len(env._enemies), 0)
        self.assertEqual(env._shield, 70) # -30 shield damage

        # 3. Player vs Enemy Bullet
        env = GymRaptorEnv()
        env.reset()
        env._player_pos = [15, 5]
        env._bullets.append([15, 5])
        
        env.step(STAY)
        self.assertEqual(len(env._bullets), 0)
        self.assertEqual(env._shield, 90) # -10 shield damage

        # 4. Player vs Coin
        env = GymRaptorEnv()
        env.reset()
        env._player_pos = [15, 5]
        env._coins.append([15, 5])
        
        env.step(STAY)
        self.assertEqual(len(env._coins), 0)
        self.assertEqual(env._money, 50)
        self.assertEqual(env._score, 500)

    def test_termination(self):
        """Test environment termination when shield reaches 0."""
        env = GymRaptorEnv()
        env.reset()
        env._shield = 10
        env._enemies.append(([env._player_pos[0], env._player_pos[1]], raptor_env.ENEMY_BASIC))
        
        obs, reward, terminated, truncated, info = env.step(STAY)
        self.assertTrue(terminated)
        self.assertEqual(env._shield, 0)

    def test_rendering_and_close(self):
        """Test rendering visual canvas output with all entity types present."""
        env = GymRaptorEnv()
        env.reset(seed=42)

        # Make some steps with movement to fill history
        env.step(LEFT)
        env.step(RIGHT)
        env.step(UP)
        env.step(DOWN)

        # Clear existing entities to prevent interference
        env._enemies = []
        env._bullets = []
        env._coins = []
        env._lasers = []

        # Manually add all entity types to test all rendering branches
        env._enemies.append(([2, 2], raptor_env.ENEMY_SHOOTER))
        env._bullets.append([3, 3])
        env._coins.append([4, 4])
        env._lasers.append([5, 5])

        # Step on an even count to move the coin (covering coin movement lines 281-283)
        env._step_count = 1
        env.step(LEFT)
        self.assertEqual(env._coins[0], [5, 4])

        # Render
        img = env.render()
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (CANVAS_SIZE[1], CANVAS_SIZE[0], 3))

        env.close()

    def test_shooter_spawning(self):
        """Test spawning logic branches by mocking np_random."""
        env = GymRaptorEnv()
        env.reset()
        
        # Step count multiple of 4, spawn_roll in [0.4, 0.6] for ENEMY_SHOOTER
        env._step_count = 3
        from unittest.mock import MagicMock
        env.np_random = MagicMock()
        env.np_random.random = MagicMock(return_value=0.5)
        env.np_random.integers = MagicMock(return_value=5)
        
        env.step(STAY) # step_count becomes 4
        self.assertEqual(env._enemies[-1], ([0, 5], raptor_env.ENEMY_SHOOTER))
        
        # Test coin spawn (spawn_roll = 0.7)
        env._step_count = 7
        env.np_random.random = MagicMock(return_value=0.7)
        env.step(STAY) # step_count becomes 8
        self.assertEqual(env._coins[-1], [0, 5])

    def test_font_loading_fallback(self):
        """Test font loading fallback."""
        from unittest.mock import patch
        with patch("matplotlib.font_manager.findfont", side_effect=Exception("Font error")):
            env = GymRaptorEnv()
            self.assertIsNotNone(env._title_font)

    def test_invalid_action_value_error(self):
        """Test invalid actions raise ValueError."""
        env = GymRaptorEnv()
        env.reset()
        with self.assertRaises(ValueError):
            env.step(5)
        with self.assertRaises(ValueError):
            env.step(-1)

    def test_gymnasium_compliance(self):
        """Test compliance with Gymnasium standard checker."""
        from gymnasium.utils.env_checker import check_env
        env = GymRaptorEnv()
        check_env(env, skip_render_check=True)

    def test_seeding_and_determinism(self):
        """Test environment seeding determinism."""
        env1 = GymRaptorEnv()
        env2 = GymRaptorEnv()
        
        obs1, _ = env1.reset(seed=456)
        obs2, _ = env2.reset(seed=456)
        
        np.testing.assert_array_equal(obs1["observation"], obs2["observation"])
        
        for _ in range(10):
            action = env1.action_space.sample()
            o1, r1, term1, trunc1, _ = env1.step(action)
            o2, r2, term2, trunc2, _ = env2.step(action)
            
            np.testing.assert_array_equal(o1["observation"], o2["observation"])
            np.testing.assert_array_equal(o1["valid_mask"], o2["valid_mask"])
            self.assertEqual(r1, r2)
            self.assertEqual(term1, term2)

    def test_state_saving_and_restoring(self):
        """Test state save/restore options."""
        env = GymRaptorEnv()
        env.reset(seed=42)
        
        env.step(UP)
        env.step(LEFT)
        
        _, _, _, _, info = env.step(STAY)
        saved_state = info["state"]
        
        new_env = GymRaptorEnv()
        new_env.reset(options={"state": saved_state})
        
        self.assertEqual(new_env._player_pos, env._player_pos)
        self.assertEqual(new_env._shield, env._shield)
        self.assertEqual(new_env._score, env._score)
        self.assertEqual(new_env._money, env._money)
        self.assertEqual(new_env._step_count, env._step_count)
        self.assertEqual(new_env._enemies, env._enemies)
        self.assertEqual(new_env._lasers, env._lasers)
        self.assertEqual(new_env._bullets, env._bullets)
        self.assertEqual(new_env._coins, env._coins)
        self.assertEqual(new_env._stars, env._stars)
        self.assertEqual(new_env._move_history, env._move_history)


if __name__ == "__main__":
    unittest.main()

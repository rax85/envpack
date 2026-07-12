"""Tests for GymPacmanEnv."""

import unittest
import numpy as np

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from envpack.envs.game_pacman.env import GymPacmanEnv, UP, DOWN, LEFT, RIGHT, MAZE_LAYOUT, POWER_PELLETS_START


class TestGymPacmanEnv(unittest.TestCase):
    """Tests for the GymPacmanEnv Gymnasium environment."""

    def test_gym_compliance(self):
        """Test Gymnasium compliance using check_env."""
        env = GymPacmanEnv()
        check_env(env, skip_render_check=True)

    def test_initial_state(self):
        """Test that the initial state is correct after reset."""
        env = GymPacmanEnv()
        obs, info = env.reset()

        # Check observation structure
        self.assertIn("observation", obs)
        self.assertIn("valid_mask", obs)
        self.assertIn("score", obs)
        self.assertIn("lives", obs)

        # Check observation shapes and dtypes
        self.assertEqual(obs["observation"].shape, (300, 400, 3))
        self.assertEqual(obs["observation"].dtype, np.uint8)
        self.assertEqual(obs["valid_mask"].shape, (4,))
        self.assertEqual(obs["valid_mask"].dtype, np.int8)
        self.assertEqual(obs["score"].shape, (1,))
        self.assertEqual(obs["score"].dtype, np.int32)
        self.assertEqual(obs["lives"].shape, (1,))
        self.assertEqual(obs["lives"].dtype, np.int32)

        # Defaults
        self.assertEqual(env.score, 0)
        self.assertEqual(env.lives, 3)
        self.assertEqual(env.frightened_timer, 0)
        self.assertEqual(env.pacman_pos, (13, 7))

    def test_movement_and_wall_collision(self):
        """Test Pacman movement and wall blocking."""
        env = GymPacmanEnv()
        env.reset()
        
        # Pacman starts at (13, 7) moving RIGHT.
        # In MAZE_LAYOUT:
        # Row 13 is: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        # (13, 7) is open. Moving LEFT (2) should go to (13, 6).
        obs, reward, term, trunc, info = env.step(LEFT)
        self.assertEqual(env.pacman_pos, (13, 6))
        self.assertEqual(env.pacman_dir, LEFT)

        # Let's verify valid_mask at (13, 6):
        # Left neighbor (13, 5): 0 (valid)
        # Right neighbor (13, 7): 0 (valid)
        # Up neighbor (12, 6): MAZE_LAYOUT[12][6] is 0
        # Down neighbor (14, 6): MAZE_LAYOUT[14][6] is 1 (wall)
        # So valid mask should be UP=1, DOWN=0, LEFT=1, RIGHT=1
        mask = env._get_valid_mask()
        self.assertEqual(mask[UP], 1)
        self.assertEqual(mask[DOWN], 0)
        self.assertEqual(mask[LEFT], 1)
        self.assertEqual(mask[RIGHT], 1)

        # Move UP into (12, 6). MAZE_LAYOUT[12][6] is 0.
        env.step(UP)
        self.assertEqual(env.pacman_pos, (12, 6))

        # Trying to move LEFT into (12, 5) which is MAZE_LAYOUT[12][5] = 1 (wall).
        # This action is invalid, so Pacman continues in current direction (UP) if possible.
        # Up neighbor of (12, 6) is (11, 6). MAZE_LAYOUT[11][6] is 0. So it should move UP.
        env.step(LEFT)
        self.assertEqual(env.pacman_pos, (11, 6))

    def test_custom_state_inject(self):
        """Test resetting the environment with a custom state injected."""
        env = GymPacmanEnv()
        state = {
            "pacman_pos": (4, 4),
            "pacman_dir": LEFT,
            "ghosts": [
                {"name": "Blinky", "pos": (1, 1), "dir": RIGHT, "start_pos": (7, 7)},
            ],
            "dots": [(4, 5), (4, 6)],
            "power_pellets": [(1, 1)],
            "score": 120,
            "lives": 2,
            "frightened_timer": 15,
        }
        obs, info = env.reset(options={"state": state})
        self.assertEqual(env.pacman_pos, (4, 4))
        self.assertEqual(env.pacman_dir, LEFT)
        self.assertEqual(len(env.ghosts), 1)
        self.assertEqual(env.ghosts[0]["name"], "Blinky")
        self.assertEqual(env.ghosts[0]["pos"], (1, 1))
        self.assertEqual(env.dots, {(4, 5), (4, 6)})
        self.assertEqual(env.power_pellets, {(1, 1)})
        self.assertEqual(env.score, 120)
        self.assertEqual(env.lives, 2)
        self.assertEqual(env.frightened_timer, 15)

    def test_dot_and_pellet_consumption(self):
        """Test score increases and frightened mode transition on eating pellets."""
        env = GymPacmanEnv()
        state = {
            "pacman_pos": (4, 4),
            "pacman_dir": RIGHT,
            "ghosts": [],
            "dots": [(4, 5), (1, 2)],  # (1, 2) is a dummy dot to prevent level clear
            "power_pellets": [(4, 6)],
            "score": 0,
            "lives": 3,
            "frightened_timer": 0,
        }
        env.reset(options={"state": state})

        # Step 1: Move right to eat dot
        obs, reward, term, trunc, info = env.step(RIGHT)
        self.assertEqual(env.pacman_pos, (4, 5))
        self.assertEqual(env.score, 10)
        self.assertNotIn((4, 5), env.dots)
        self.assertAlmostEqual(reward, 10.0 - 0.01)

        # Step 2: Move right to eat power pellet
        obs, reward, term, trunc, info = env.step(RIGHT)
        self.assertEqual(env.pacman_pos, (4, 6))
        self.assertEqual(env.score, 60) # 10 + 50
        self.assertNotIn((4, 6), env.power_pellets)
        self.assertEqual(env.frightened_timer, 40)
        self.assertAlmostEqual(reward, 50.0 - 0.01)

    def test_ghost_combat(self):
        """Test Pacman eating frightened ghost, and dying to normal ghost."""
        env = GymPacmanEnv()
        
        # 1. Normal ghost collision: Pacman dies
        state = {
            "pacman_pos": (4, 4),
            "pacman_dir": RIGHT,
            "ghosts": [
                {"name": "Blinky", "pos": (4, 5), "dir": LEFT, "start_pos": (7, 7)},
            ],
            "dots": [],
            "power_pellets": [],
            "score": 0,
            "lives": 3,
            "frightened_timer": 0,
        }
        env.reset(options={"state": state})
        
        # Step: Pacman moves RIGHT, Blinky moves LEFT (or towards Pacman). They collide.
        obs, reward, term, trunc, info = env.step(RIGHT)
        self.assertEqual(env.lives, 2)
        # Pacman should be reset to PACMAN_START
        self.assertEqual(env.pacman_pos, (13, 7))
        # Blinky should be reset to its start_pos
        self.assertEqual(env.ghosts[0]["pos"], (7, 7))

        # 2. Frightened ghost collision: Ghost eaten
        state = {
            "pacman_pos": (4, 4),
            "pacman_dir": RIGHT,
            "ghosts": [
                {"name": "Blinky", "pos": (4, 5), "dir": LEFT, "start_pos": (7, 7)},
            ],
            "dots": [(1, 2)], # keep some dot so level doesn't reset
            "power_pellets": [],
            "score": 0,
            "lives": 3,
            "frightened_timer": 20,
        }
        env.reset(options={"state": state})
        
        # Step: Pacman moves RIGHT to (4, 5). Blinky is frightened so moves half speed (doesn't move on step 1 of frightened mode if odd step, or we manually check).
        # Pacman lands on Blinky.
        obs, reward, term, trunc, info = env.step(RIGHT)
        self.assertEqual(env.lives, 3) # no life lost
        self.assertEqual(env.score, 200) # ate ghost
        # Blinky should be reset to start_pos
        self.assertEqual(env.ghosts[0]["pos"], (7, 7))

    def test_level_clear(self):
        """Test level reset when all dots and pellets are cleared."""
        env = GymPacmanEnv()
        state = {
            "pacman_pos": (4, 4),
            "pacman_dir": RIGHT,
            "ghosts": [],
            "dots": [(4, 5)],
            "power_pellets": [],
            "score": 0,
            "lives": 3,
            "frightened_timer": 0,
        }
        env.reset(options={"state": state})

        # Eat the last dot
        obs, reward, term, trunc, info = env.step(RIGHT)
        # Should get dot score + level clear score (10 + 500 = 510)
        self.assertEqual(env.score, 510)
        # Entities reset
        self.assertEqual(env.pacman_pos, (13, 7))
        # Dots and pellets regenerated
        self.assertGreater(len(env.dots), 0)
        self.assertEqual(len(env.power_pellets), len(POWER_PELLETS_START))

"""Tests for the GymTetrisEnv environment."""

import copy
import unittest
import numpy as np

from envpack.envs.game_tetris import env as tetris_env
from envpack.envs.game_tetris.env import GymTetrisEnv


class TestGymTetrisEnv(unittest.TestCase):
    """Tests for the GymTetrisEnv environment."""

    def test_initial_state(self):
        """Test that the initial state of the environment is set up correctly."""
        env = GymTetrisEnv()
        obs, _ = env.reset()

        # Board should be empty
        self.assertTrue(np.all(env._board == 0))
        self.assertEqual(env._score, 0)
        self.assertEqual(env._lines_cleared, 0)
        self.assertEqual(env._total_moves, 0)

        # Check observation structure
        self.assertIn("observation", obs)
        self.assertIn("valid_mask", obs)
        self.assertIn("total_score", obs)
        
        # Grid checks
        grid = obs["observation"]
        self.assertEqual(obs["total_score"][0], 0)
        self.assertEqual(grid.shape, (20, 10))

        # Falling piece should be present (represented by value 8)
        self.assertEqual(np.count_nonzero(grid == 8), np.count_nonzero(env._current_shape != 0))

    def test_movement_left_right(self):
        """Test moving the piece left and right."""
        env = GymTetrisEnv()
        env.reset()
        
        # Set a fixed piece type (O-piece, 2x2 shape) for testing
        env._current_type = 2
        env._current_shape = tetris_env.SHAPES[2].copy()
        env._current_pos = (0, 4)  # Start at col 4
        
        # Move Left
        obs, reward, terminated, truncated, info = env.step(tetris_env.LEFT)
        # Note: gravity also applies in step, so row position will increase by 1
        self.assertEqual(env._current_pos, (1, 3))
        
        # Move Right
        env.step(tetris_env.RIGHT)
        self.assertEqual(env._current_pos, (2, 4))

    def test_rotate(self):
        """Test rotating the piece."""
        env = GymTetrisEnv()
        env.reset()
        
        # Use T-piece (3x3 shape)
        env._current_type = 3
        env._current_shape = tetris_env.SHAPES[3].copy()
        env._current_pos = (0, 4)
        
        initial_shape = env._current_shape.copy()
        
        # Rotate
        env.step(tetris_env.ROTATE)
        
        # Shape should be rotated clockwise (which is rot90 with k=-1)
        expected_shape = np.rot90(initial_shape, k=-1)
        np.testing.assert_array_equal(env._current_shape, expected_shape)

    def test_hard_drop(self):
        """Test hard dropping the piece."""
        env = GymTetrisEnv()
        env.reset()
        
        # Set board empty, O-piece at row 0, col 4
        env._current_type = 2
        env._current_shape = tetris_env.SHAPES[2].copy()
        env._current_pos = (0, 4)
        
        # Hard drop should drop to row 18 and lock it on row 18 & 19
        obs, reward, terminated, _, _ = env.step(tetris_env.HARD_DROP)
        
        # Check that the piece is locked on the board
        self.assertEqual(env._board[18, 4], 2)
        self.assertEqual(env._board[18, 5], 2)
        self.assertEqual(env._board[19, 4], 2)
        self.assertEqual(env._board[19, 5], 2)
        
        # Next piece should be spawned at the top
        self.assertEqual(env._current_pos[0], 0)

    def test_line_clear(self):
        """Test that clearing lines updates the board and score."""
        env = GymTetrisEnv()
        env.reset()
        
        # Fill row 19 completely except for cols 4 and 5
        env._board[19, :] = 1
        env._board[19, 4] = 0
        env._board[19, 5] = 0
        
        # Place O-piece (2x2) at row 18, col 4 and hard drop it
        env._current_type = 2
        env._current_shape = tetris_env.SHAPES[2].copy()
        env._current_pos = (18, 4)
        
        # Hard drop to lock it and clear row 19
        obs, reward, terminated, _, _ = env.step(tetris_env.HARD_DROP)
        
        # Row 19 should have been cleared, row 18 shifts down to row 19
        # Cols 4 and 5 on row 19 should now be filled with value 2 (from the O-piece lock)
        # The rest of row 19 was cleared, but the other row 18 of the O-piece remains on row 19.
        # Wait, let's verify if row 19 is cleared.
        # Before clear:
        # Row 19: [1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
        # Row 18: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # After O-piece (2x2) lock:
        # Row 19 becomes: [1, 1, 1, 1, 2, 2, 1, 1, 1, 1] (fully occupied, cleared!)
        # Row 18 becomes: [0, 0, 0, 0, 2, 2, 0, 0, 0, 0] (O-piece top half, shifted down to Row 19 after clear)
        # So row 19 should have the O-piece top half: [0, 0, 0, 0, 2, 2, 0, 0, 0, 0]
        np.testing.assert_array_equal(env._board[19], [0, 0, 0, 0, 2, 2, 0, 0, 0, 0])
        self.assertEqual(env._lines_cleared, 1)
        self.assertEqual(env._score, 100)

    def test_collision_detection(self):
        """Test the collision detection logic."""
        env = GymTetrisEnv()
        env.reset()
        
        # T-piece at boundary
        shape = tetris_env.SHAPES[3]
        
        # Outside left boundary
        self.assertTrue(env._check_collision(env._board, shape, (0, -2)))
        # Outside right boundary
        self.assertTrue(env._check_collision(env._board, shape, (0, 9)))
        # Outside bottom boundary
        self.assertTrue(env._check_collision(env._board, shape, (19, 4)))
        # Safe position
        self.assertFalse(env._check_collision(env._board, shape, (0, 4)))

    def test_state_saving_and_restoring(self):
        """Test saving and restoring state works correctly."""
        env = GymTetrisEnv()
        env.reset()
        
        # Perform some steps
        env.step(tetris_env.LEFT)
        env.step(tetris_env.ROTATE)
        _, _, _, _, info = env.step(tetris_env.DOWN)
        
        saved_state = info["state"]
        
        new_env = GymTetrisEnv()
        new_env.reset(options={"state": saved_state})
        
        self.assertTrue(np.array_equal(new_env._board, env._board))
        self.assertEqual(new_env._score, env._score)
        self.assertEqual(new_env._lines_cleared, env._lines_cleared)
        self.assertEqual(new_env._total_moves, env._total_moves)
        self.assertEqual(new_env._current_type, env._current_type)
        self.assertTrue(np.array_equal(new_env._current_shape, env._current_shape))
        self.assertEqual(new_env._current_pos, env._current_pos)
        self.assertEqual(new_env._next_type, env._next_type)
        self.assertEqual(new_env._move_history, env._move_history)


if __name__ == "__main__":
    unittest.main()

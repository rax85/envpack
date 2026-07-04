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
    def test_gymnasium_compliance(self):
        """Test that the environment complies with Gymnasium interface requirements."""
        from gymnasium.utils.env_checker import check_env
        env = GymTetrisEnv()
        check_env(env)

    def test_seeding_and_determinism(self):
        """Test that the environment is deterministic when seeded."""
        env1 = GymTetrisEnv()
        env2 = GymTetrisEnv()
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        # Initial shapes and types should match
        self.assertEqual(env1._current_type, env2._current_type)
        self.assertEqual(env1._next_type, env2._next_type)
        self.assertEqual(env1._current_pos, env2._current_pos)
        np.testing.assert_array_equal(env1._board, env2._board)
        
        # Take a few identical actions
        for action in [0, 1, 2, 3]:
            obs1, r1, term1, trunc1, _ = env1.step(action)
            obs2, r2, term2, trunc2, _ = env2.step(action)
            self.assertEqual(r1, r2)
            self.assertEqual(term1, term2)
            self.assertEqual(trunc1, trunc2)
            np.testing.assert_array_equal(obs1["observation"], obs2["observation"])
            np.testing.assert_array_equal(obs1["valid_mask"], obs2["valid_mask"])
            self.assertEqual(env1._next_type, env2._next_type)

    def test_down_action(self):
        """Test the DOWN action (1 step soft drop + 1 step gravity)."""
        env = GymTetrisEnv()
        env.reset()
        
        # Use O-piece
        env._current_type = 2
        env._current_shape = tetris_env.SHAPES[2].copy()
        env._current_pos = (0, 4)
        
        # Take DOWN step
        env.step(tetris_env.DOWN)
        # Position should be 2 rows down (row 2)
        self.assertEqual(env._current_pos, (2, 4))

    def test_invalid_action(self):
        """Test that an invalid action is ignored for horizontal move but gravity still applies."""
        env = GymTetrisEnv()
        env.reset()
        
        # Use O-piece at left border (col 0)
        env._current_type = 2
        env._current_shape = tetris_env.SHAPES[2].copy()
        env._current_pos = (0, 0)
        
        # Try to move left (invalid since col is 0 and shape occupies col 0)
        obs, reward, terminated, truncated, info = env.step(tetris_env.LEFT)
        
        # Position should be moved down by gravity (row 1) but col remains 0
        self.assertEqual(env._current_pos, (1, 0))
        # Move history should record LEFT (0) as invalid (False)
        self.assertEqual(env._move_history[-1], (tetris_env.LEFT, False))

    def test_font_loading_fallback(self):
        """Test that environment falls back to default font if truetype loading fails."""
        from unittest.mock import patch
        with patch("matplotlib.font_manager.findfont", side_effect=Exception("Font error")):
            env = GymTetrisEnv()
            self.assertIsNotNone(env._font)

    def test_rendering_and_close(self):
        """Test rendering, arrow drawing in all directions, and close."""
        env = GymTetrisEnv()
        env.reset()
        
        # Move RIGHT, DOWN, LEFT, ROTATE, HARD_DROP to build history
        env.step(tetris_env.RIGHT)
        env.step(tetris_env.DOWN)
        env.step(tetris_env.LEFT)
        env.step(tetris_env.ROTATE)
        env.step(tetris_env.HARD_DROP)
        
        # Render (this will draw LEFT, RIGHT, ROTATE, DOWN, HARD_DROP arrows)
        img = env.render()
        self.assertIsNotNone(img)

        # Fill history to > 8 to trigger move_history.pop(0)
        for _ in range(5):
            env.step(tetris_env.LEFT)
            
        self.assertEqual(len(env._move_history), 8)
        
        # Invalid arrow action
        from PIL import ImageDraw, Image
        canvas = Image.new("RGB", (100, 100))
        draw = ImageDraw.Draw(canvas)
        env._draw_arrow(draw, 50, 50, 99, (255, 255, 255))
        
        env.close()

    def test_line_clearing_and_scoring(self):
        """Test clearing 1, 2, 3, and 4 lines and their rewards / scoring."""
        env = GymTetrisEnv()
        env.reset()
        
        # 1 Line clear
        env._board = np.zeros(tetris_env.GRID_SIZE, dtype=np.int32)
        env._board[19, :] = 1
        env._current_pos = (18, 4)
        env._current_shape = np.array([[1]], dtype=np.int32)
        cleared, game_over = env._lock_and_update()
        self.assertEqual(cleared, 1)
        self.assertEqual(env._score, 100)
        self.assertEqual(env._get_line_reward(1), 0.1)
        
        # 2 Lines clear
        env._score = 0
        env._board = np.zeros(tetris_env.GRID_SIZE, dtype=np.int32)
        env._board[18, :] = 1
        env._board[19, :] = 1
        env._current_pos = (17, 4)
        env._current_shape = np.array([[1]], dtype=np.int32)
        cleared, game_over = env._lock_and_update()
        self.assertEqual(cleared, 2)
        self.assertEqual(env._score, 300)
        self.assertEqual(env._get_line_reward(2), 0.3)

        # 3 Lines clear
        env._score = 0
        env._board = np.zeros(tetris_env.GRID_SIZE, dtype=np.int32)
        env._board[17, :] = 1
        env._board[18, :] = 1
        env._board[19, :] = 1
        env._current_pos = (16, 4)
        env._current_shape = np.array([[1]], dtype=np.int32)
        cleared, game_over = env._lock_and_update()
        self.assertEqual(cleared, 3)
        self.assertEqual(env._score, 500)
        self.assertEqual(env._get_line_reward(3), 0.5)

        # 4 Lines clear
        env._score = 0
        env._board = np.zeros(tetris_env.GRID_SIZE, dtype=np.int32)
        env._board[16, :] = 1
        env._board[17, :] = 1
        env._board[18, :] = 1
        env._board[19, :] = 1
        env._current_pos = (15, 4)
        env._current_shape = np.array([[1]], dtype=np.int32)
        cleared, game_over = env._lock_and_update()
        self.assertEqual(cleared, 4)
        self.assertEqual(env._score, 800)
        self.assertEqual(env._get_line_reward(4), 1.0)
        
        # 0 Lines clear reward
        self.assertEqual(env._get_line_reward(0), 0.0)

    def test_game_over_conditions(self):
        """Test game over condition when board is full and new piece cannot spawn."""
        env = GymTetrisEnv()
        env.reset()
        
        # Spawn piece failure
        env._board = np.zeros(tetris_env.GRID_SIZE, dtype=np.int32)
        env._board[0:3, :] = 1
        self.assertFalse(env._spawn_piece())
        
        # Lock and update triggers spawn_piece failure
        env._board = np.zeros(tetris_env.GRID_SIZE, dtype=np.int32)
        env._board[0:3, 1:] = 1  # Leave column 0 empty so the rows are not cleared
        env._current_pos = (19, 4)
        env._current_shape = np.array([[1]], dtype=np.int32)
        cleared, game_over = env._lock_and_update()
        self.assertTrue(game_over)
        
        # Test step collision check on gravity step triggering game over
        env = GymTetrisEnv()
        env.reset()
        env._board[1:4, 1:] = 1  # Leave col 0 empty so it won't be cleared
        env._current_pos = (0, 4)
        obs, reward, terminated, truncated, info = env.step(tetris_env.DOWN)
        self.assertTrue(terminated)
        self.assertEqual(reward, -1.0)
        
        # Test step collision check on hard drop triggering game over
        env = GymTetrisEnv()
        env.reset()
        env._board[1:4, 1:] = 1  # Leave col 0 empty so it won't be cleared
        env._current_pos = (0, 4)
        obs, reward, terminated, truncated, info = env.step(tetris_env.HARD_DROP)
        self.assertTrue(terminated)
        self.assertEqual(reward, -1.0)


if __name__ == "__main__":
    unittest.main()

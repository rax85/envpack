"""Tests for GymSudokuEnv."""

import copy
import unittest
import numpy as np

from envpack.envs.game_sudoku import env as sudoku_env
from envpack.envs.game_sudoku.env import GymSudokuEnv, CANVAS_SIZE


class TestGymSudokuEnv(unittest.TestCase):
    """Tests for the GymSudokuEnv environment."""

    def test_initial_state(self):
        """Test that the initial state is set up correctly."""
        env = GymSudokuEnv(clues=30)
        obs, _ = env.reset()

        self.assertEqual(env.clues, 30)
        self.assertEqual(np.sum(env._given_mask), 30)
        self.assertEqual(np.sum(env._grid != 0), 30)

        # Check observation structure
        self.assertIn("observation", obs)
        self.assertIn("given_mask", obs)
        self.assertIn("valid_mask", obs)
        self.assertIn("total_score", obs)

        # Total score should represent the number of correct cells initially
        self.assertEqual(obs["total_score"][0], env._score)

        # Valid mask shape should be (9, 9, 10)
        self.assertEqual(obs["valid_mask"].shape, (9, 9, 10))

    def test_editable_vs_given_cells(self):
        """Test that editable cells can be modified while given cells are blocked."""
        env = GymSudokuEnv(clues=30)
        env.reset(seed=42)

        # Find a given cell and an editable cell
        given_r, given_c = np.argwhere(env._given_mask == 1)[0]
        edit_r, edit_c = np.argwhere(env._given_mask == 0)[0]

        given_val_before = env._grid[given_r, given_c]
        
        # 1. Attempt to edit given cell -> should be blocked, score stays same, invalid moves incremented
        invalid_moves_before = env._invalid_moves
        action = np.array([given_r, given_c, 5 if given_val_before != 5 else 6], dtype=np.int32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        self.assertEqual(env._grid[given_r, given_c], given_val_before)
        self.assertEqual(env._invalid_moves, invalid_moves_before + 1)
        self.assertEqual(reward, -0.01)  # Just the step penalty

        # 2. Attempt to edit editable cell -> should update, score/reward changes
        # Let's find what the correct value is
        correct_val = env._solved_grid[edit_r, edit_c]
        
        # Place correct value
        action = np.array([edit_r, edit_c, correct_val], dtype=np.int32)
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertEqual(env._grid[edit_r, edit_c], correct_val)
        self.assertGreater(reward, 0.0) # +1.0 for correct - 0.01 step penalty

        # Place incorrect value (conflict or incorrect)
        # Find a value that is not correct_val
        wrong_val = 5 if correct_val != 5 else 6
        action = np.array([edit_r, edit_c, wrong_val], dtype=np.int32)
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertEqual(env._grid[edit_r, edit_c], wrong_val)
        self.assertLess(reward, 0.0) # -1.0 score diff + step penalty

        # Clear cell
        action = np.array([edit_r, edit_c, 0], dtype=np.int32)
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertEqual(env._grid[edit_r, edit_c], 0)

    def test_conflict_detection(self):
        """Test that placing conflicting digits triggers a conflict penalty."""
        env = GymSudokuEnv(clues=30)
        env.reset(seed=42)

        # 1. Row/Col conflict
        edit_r, edit_c = np.argwhere(env._given_mask == 0)[0]
        row_vals = env._grid[edit_r, :]
        given_row_vals = row_vals[env._given_mask[edit_r, :] == 1]
        
        if len(given_row_vals) > 0:
            conflict_val = given_row_vals[0]
            action = np.array([edit_r, edit_c, conflict_val], dtype=np.int32)
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertTrue(env._has_conflict(edit_r, edit_c, conflict_val))
            self.assertLess(reward, -0.1)

        # 2. Block conflict (same 3x3 block, different row and column)
        env._grid = np.zeros((9, 9), dtype=np.int32)
        env._grid[0, 0] = 5
        self.assertTrue(env._has_conflict(1, 1, 5))

    def test_rendering_and_close(self):
        """Test rendering visual arrays and checking arrow/move history rendering in footer."""
        env = GymSudokuEnv(clues=30)
        env.reset(seed=42)

        # Step a few times to build move history (valid/invalid)
        env.step(np.array([0, 0, 1], dtype=np.int32)) # invalid (given cell)
        env.step(np.array([0, 1, 1], dtype=np.int32)) # valid
        
        # Fill history to pop oldest
        for _ in range(8):
            env.step(np.array([0, 1, 0], dtype=np.int32))

        self.assertEqual(len(env._move_history), 8)

        # Setup board state manually to test all cell text coloring branches
        env._grid = np.zeros((9, 9), dtype=np.int32)
        env._given_mask = np.zeros((9, 9), dtype=np.int8)
        env._solved_grid = np.ones((9, 9), dtype=np.int32) # solved is all 1s

        # - Given cell
        env._grid[0, 0] = 1
        env._given_mask[0, 0] = 1
        # - Correct cell
        env._grid[0, 4] = 1
        # - Conflicting cells (both are incorrect value 2, creating row conflict)
        env._grid[0, 1] = 2
        env._grid[0, 2] = 2
        # - Incorrect but non-conflicting cell (value 3 in row 1 where solved is 1)
        env._grid[1, 3] = 3

        # Render
        img = env.render()
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (CANVAS_SIZE[1], CANVAS_SIZE[0], 3))

        env.close()

    def test_font_loading_fallback(self):
        """Test that environment falls back to default font if truetype loading fails."""
        from unittest.mock import patch
        with patch("matplotlib.font_manager.findfont", side_effect=Exception("Font error")):
            env = GymSudokuEnv()
            self.assertIsNotNone(env._cell_font)

    def test_font_loading_fallback_old_pillow(self):
        """Test that environment falls back to default font if Pillow does not support size argument."""
        from unittest.mock import patch
        
        orig_load_default = sudoku_env.ImageFont.load_default
        def mock_load_default(*args, **kwargs):
            if kwargs or args:
                raise TypeError("load_default() takes no arguments")
            return orig_load_default()
            
        with patch("matplotlib.font_manager.findfont", side_effect=Exception("Font error")), \
             patch("envpack.envs.game_sudoku.env.ImageFont.load_default", side_effect=mock_load_default):
            env = GymSudokuEnv()
            self.assertIsNotNone(env._cell_font)

    def test_invalid_action_value_error(self):
        """Test that invalid action values raise ValueError."""
        env = GymSudokuEnv()
        env.reset()
        
        with self.assertRaises(ValueError):
            env.step(np.array([-1, 0, 5], dtype=np.int32))
        with self.assertRaises(ValueError):
            env.step(np.array([0, 9, 5], dtype=np.int32))
        with self.assertRaises(ValueError):
            env.step(np.array([0, 0, 11], dtype=np.int32))

    def test_gymnasium_compliance(self):
        """Test environment compliance with Gymnasium api standards."""
        from gymnasium.utils.env_checker import check_env
        env = GymSudokuEnv()
        check_env(env, skip_render_check=True)

    def test_seeding_and_determinism(self):
        """Test that seeding is fully deterministic."""
        env1 = GymSudokuEnv()
        env2 = GymSudokuEnv()
        
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        
        np.testing.assert_array_equal(obs1["observation"], obs2["observation"])
        np.testing.assert_array_equal(env1._solved_grid, env2._solved_grid)

        # Run steps and verify determinism
        for _ in range(5):
            action = env1.action_space.sample()
            o1, r1, term1, trunc1, _ = env1.step(action)
            o2, r2, term2, trunc2, _ = env2.step(action)
            
            np.testing.assert_array_equal(o1["observation"], o2["observation"])
            np.testing.assert_array_equal(o1["valid_mask"], o2["valid_mask"])
            self.assertEqual(r1, r2)
            self.assertEqual(term1, term2)
            self.assertEqual(trunc1, trunc2)

    def test_solve_puzzle_completion(self):
        """Test that filling the final cell correctly terminates the game with completion bonus."""
        env = GymSudokuEnv(clues=80) # Only 1 empty cell
        env.reset(seed=42)
        
        # Cover _has_conflict with val=0
        self.assertFalse(env._has_conflict(0, 0, 0))

        edit_r, edit_c = np.argwhere(env._given_mask == 0)[0]
        correct_val = env._solved_grid[edit_r, edit_c]
        
        obs, reward, terminated, truncated, info = env.step(np.array([edit_r, edit_c, correct_val], dtype=np.int32))
        
        self.assertTrue(terminated)
        self.assertGreaterEqual(reward, 10.0) # Completion bonus

    def test_full_board_with_conflicts(self):
        """Test that a full board with conflicts does not terminate and doesn't get completion bonus."""
        env = GymSudokuEnv()
        env.reset()
        
        # Manually fill grid with conflicts (all 1s)
        env._grid = np.ones((9, 9), dtype=np.int32)
        env._given_mask = np.zeros((9, 9), dtype=np.int8)
        
        # Step to trigger check
        obs, reward, terminated, truncated, info = env.step(np.array([0, 0, 1], dtype=np.int32))
        
        self.assertFalse(terminated)
        self.assertLess(reward, 0.0)

    def test_state_saving_and_restoring(self):
        """Test that state saving and restoring works correctly."""
        env = GymSudokuEnv()
        env.reset(seed=42)
        
        # Make a move
        edit_r, edit_c = np.argwhere(env._given_mask == 0)[0]
        correct_val = env._solved_grid[edit_r, edit_c]
        _, _, _, _, info = env.step(np.array([edit_r, edit_c, correct_val], dtype=np.int32))
        
        saved_state = info["state"]
        
        new_env = GymSudokuEnv()
        new_env.reset(options={"state": saved_state})
        
        np.testing.assert_array_equal(new_env._grid, env._grid)
        np.testing.assert_array_equal(new_env._solved_grid, env._solved_grid)
        np.testing.assert_array_equal(new_env._given_mask, env._given_mask)
        self.assertEqual(new_env._score, env._score)
        self.assertEqual(new_env._total_moves, env._total_moves)
        self.assertEqual(new_env._invalid_moves, env._invalid_moves)
        self.assertEqual(new_env._move_history, env._move_history)


if __name__ == "__main__":
    unittest.main()

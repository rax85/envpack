"""Tests for GymCheckersEnv."""

import copy
import unittest
import numpy as np

from envpack.envs.game_checkers import env as checkers_env
from envpack.envs.game_checkers.env import GymCheckersEnv, CANVAS_SIZE, EMPTY, P1_NORMAL, P1_KING, P2_NORMAL, P2_KING


class TestGymCheckersEnv(unittest.TestCase):
    """Tests for the GymCheckersEnv environment."""

    def test_initial_state(self):
        """Test that the initial state is set up correctly."""
        env = GymCheckersEnv()
        obs, _ = env.reset()

        p1_cnt, p2_cnt = env._get_pieces_count()
        self.assertEqual(p1_cnt, 12)
        self.assertEqual(p2_cnt, 12)
        self.assertEqual(env._current_player, 1)

        # Check observation structure
        self.assertIn("observation", obs)
        self.assertIn("valid_mask", obs)
        self.assertIn("current_player", obs)

        self.assertEqual(obs["current_player"], 1)
        self.assertEqual(obs["valid_mask"].shape, (8, 8, 8, 8))

    def test_regular_move_execution(self):
        """Test a normal single-step movement."""
        env = GymCheckersEnv()
        env.reset()

        # Find a valid move for Player 1
        # P1 normal is on (5, 0)
        # Check if P1 can move to (4, 1)
        self.assertEqual(env._grid[5, 0], P1_NORMAL)
        self.assertEqual(env._grid[4, 1], EMPTY)

        action = np.array([5, 0, 4, 1], dtype=np.int32)
        obs, reward, terminated, truncated, info = env.step(action)

        self.assertEqual(env._grid[5, 0], EMPTY)
        self.assertEqual(env._grid[4, 1], P1_NORMAL)
        # Turn should switch to Player 2
        self.assertEqual(env._current_player, 2)
        self.assertEqual(reward, -0.01)

    def test_mandatory_capture_rule(self):
        """Test that normal moves are invalid if a jump capture is available."""
        env = GymCheckersEnv()
        env.reset()

        # Clear board and set up a mandatory jump case
        # P1 at (4, 2), P2 at (3, 3)
        env._grid = np.zeros((8, 8), dtype=np.int32)
        env._grid[4, 2] = P1_NORMAL
        env._grid[3, 3] = P2_NORMAL

        obs = env._create_observation()
        
        # Valid moves should only consist of the jump (4,2) -> (2,4)
        valid_actions = np.argwhere(obs["valid_mask"] == 1)
        self.assertEqual(len(valid_actions), 1)
        np.testing.assert_array_equal(valid_actions[0], [4, 2, 2, 4])

        # Try to execute a normal diagonal step (4,2) -> (3,1) -> should be invalid
        obs, reward, terminated, truncated, info = env.step(np.array([4, 2, 3, 1], dtype=np.int32))
        self.assertEqual(env._grid[4, 2], P1_NORMAL)
        self.assertLess(reward, -0.1) # Invalid move penalty

    def test_multi_jump_sequence(self):
        """Test consecutive multi-jump sequence tracking turns correctly."""
        env = GymCheckersEnv()
        env.reset()

        # P1 at (6, 2)
        # P2 targets at (5, 3) and (3, 3)
        env._grid = np.zeros((8, 8), dtype=np.int32)
        env._grid[6, 2] = P1_NORMAL
        env._grid[5, 3] = P2_NORMAL
        env._grid[3, 3] = P2_NORMAL

        # Step 1: P1 jumps over (5,3) to (4,4)
        obs, reward, terminated, truncated, info = env.step(np.array([6, 2, 4, 4], dtype=np.int32))
        
        # Captured piece removed
        self.assertEqual(env._grid[5, 3], EMPTY)
        # Jumper at landing spot
        self.assertEqual(env._grid[4, 4], P1_NORMAL)
        # Jumper has another jump (4,4) -> (2,2) over (3,3)
        # Turn should REMAIN with Player 1, active jumper is tracked
        self.assertEqual(env._current_player, 1)
        self.assertEqual(env._active_jumper, (4, 4))
        
        # Verify valid moves only contain the follow-up jump
        obs = env._create_observation()
        valid_actions = np.argwhere(obs["valid_mask"] == 1)
        self.assertEqual(len(valid_actions), 1)
        np.testing.assert_array_equal(valid_actions[0], [4, 4, 2, 2])

        # Step 2: execute second jump
        obs, reward, terminated, truncated, info = env.step(np.array([4, 4, 2, 2], dtype=np.int32))
        self.assertEqual(env._grid[3, 3], EMPTY)
        self.assertEqual(env._grid[2, 2], P1_NORMAL)
        # Turn should switch to Player 2 now
        self.assertEqual(env._current_player, 2)
        self.assertIsNone(env._active_jumper)

    def test_king_promotion(self):
        """Test piece promotion to King at back rows and backward movements."""
        env = GymCheckersEnv()
        env.reset()

        # P1 normal at (1, 1), Solved solution target at row 0
        env._grid = np.zeros((8, 8), dtype=np.int32)
        env._grid[1, 1] = P1_NORMAL

        # Move to row 0 -> should promote to P1_KING
        obs, reward, terminated, truncated, info = env.step(np.array([1, 1, 0, 2], dtype=np.int32))
        self.assertEqual(env._grid[0, 2], P1_KING)
        self.assertGreater(reward, 0.4) # Normal move penalty (-0.01) + King bonus (+0.5)

        # King should be able to move backwards (row increases)
        # Turn is currently P2. Let's switch it back to P1.
        env._current_player = 1
        obs = env._create_observation()
        # King at (0,2) can move backwards to (1,1) or (1,3)
        valid_actions = np.argwhere(obs["valid_mask"] == 1)
        self.assertEqual(len(valid_actions), 2)
        
        # Move backwards
        env.step(np.array([0, 2, 1, 3], dtype=np.int32))
        self.assertEqual(env._grid[1, 3], P1_KING)

        # P2 promotion test
        env = GymCheckersEnv()
        env.reset()
        env._grid = np.zeros((8, 8), dtype=np.int32)
        env._grid[6, 6] = P2_NORMAL
        env._current_player = 2
        
        obs, reward, terminated, truncated, info = env.step(np.array([6, 6, 7, 5], dtype=np.int32))
        self.assertEqual(env._grid[7, 5], P2_KING)
        self.assertLess(reward, -0.4) # step penalty (-0.01) + P2 King penalty (-0.5)

    def test_win_loss_draw_conditions(self):
        """Test termination conditions on pieces count and block stalemates."""
        # 1. P2 has no pieces left -> P1 wins
        env = GymCheckersEnv()
        env.reset()
        env._grid = np.zeros((8, 8), dtype=np.int32)
        env._grid[5, 5] = P1_NORMAL
        env._grid[4, 4] = P2_NORMAL
        
        # P1 jumps P2, leaving P2 with 0 pieces
        obs, reward, terminated, truncated, info = env.step(np.array([5, 5, 3, 3], dtype=np.int32))
        self.assertTrue(terminated)
        self.assertEqual(reward, 10.0) # Win reward

        # 2. P1 has no pieces left -> P2 wins
        env = GymCheckersEnv()
        env.reset()
        env._grid = np.zeros((8, 8), dtype=np.int32)
        env._grid[4, 4] = P1_NORMAL
        env._grid[3, 3] = P2_NORMAL
        env._current_player = 2

        # P2 jumps P1, leaving P1 with 0 pieces
        obs, reward, terminated, truncated, info = env.step(np.array([3, 3, 5, 5], dtype=np.int32))
        self.assertTrue(terminated)
        self.assertEqual(reward, -10.0) # Win reward (Player 2 perspective from P1 reward format)

        # 3. Blocked Stalemate -> Blocked player loses
        env = GymCheckersEnv()
        env.reset()
        # Set a board where P1 has only 1 piece blocked by P2
        env._grid = np.zeros((8, 8), dtype=np.int32)
        env._grid[0, 0] = P1_NORMAL
        env._grid[1, 1] = P2_NORMAL # Blocked diagonal
        env._grid[3, 3] = P2_NORMAL # Free piece for P2
        
        env._current_player = 2 # P2's turn
        
        # P2 moves their free piece, switching turn to blocked P1
        obs, reward, terminated, truncated, info = env.step(np.array([3, 3, 4, 2], dtype=np.int32))
        self.assertTrue(terminated)
        self.assertEqual(reward, -10.0) # P1 is blocked, so P2 wins (P1 gets -10.0)

        # 4. Draw counter limits
        env = GymCheckersEnv()
        env.reset()
        env._draw_counter = 99
        # Execute invalid move
        obs, reward, terminated, truncated, info = env.step(np.array([0, 0, 0, 0], dtype=np.int32))
        self.assertTrue(truncated)

    def test_rendering_and_close(self):
        """Test rendering visual canvas and empty piece checks."""
        env = GymCheckersEnv()
        env.reset(seed=42)

        # Call empty piece check methods to cover lines 144 & 180
        self.assertEqual(env._get_jumps_for_piece(4, 4), [])
        self.assertEqual(env._get_normal_moves_for_piece(4, 4), [])

        # Run 8 valid steps using a King piece to exceed history size 6 and test pop(0)
        env._grid = np.zeros((8, 8), dtype=np.int32)
        env._grid[4, 4] = P1_KING
        for _ in range(4):
            env.step(np.array([4, 4, 5, 5], dtype=np.int32))
            env._current_player = 1
            env.step(np.array([5, 5, 4, 4], dtype=np.int32))
            env._current_player = 1

        # Run 7 invalid steps to exceed invalid history size 6 and test pop(0)
        for _ in range(7):
            env.step(np.array([0, 0, 0, 0], dtype=np.int32))

        # Manually add P1_KING back to cover King visual rendering branch
        env._grid[0, 1] = P1_KING

        img = env.render()
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (CANVAS_SIZE[1], CANVAS_SIZE[0], 3))

        env.close()

    def test_font_loading_fallback(self):
        """Test font loading fallback."""
        from unittest.mock import patch
        with patch("matplotlib.font_manager.findfont", side_effect=Exception("Font error")):
            env = GymCheckersEnv()
            self.assertIsNotNone(env._title_font)

    def test_invalid_action_value_error(self):
        """Test invalid action bounds raise ValueError."""
        env = GymCheckersEnv()
        env.reset()
        with self.assertRaises(ValueError):
            env.step(np.array([-1, 0, 0, 0], dtype=np.int32))
        with self.assertRaises(ValueError):
            env.step(np.array([0, 8, 0, 0], dtype=np.int32))

    def test_gymnasium_compliance(self):
        """Test compliance with Gymnasium standard checks."""
        from gymnasium.utils.env_checker import check_env
        env = GymCheckersEnv()
        check_env(env, skip_render_check=True)

    def test_seeding_and_determinism(self):
        """Test seeding determinism."""
        env1 = GymCheckersEnv()
        env2 = GymCheckersEnv()
        
        obs1, _ = env1.reset(seed=789)
        obs2, _ = env2.reset(seed=789)
        
        np.testing.assert_array_equal(obs1["observation"], obs2["observation"])

        # Run moves
        for _ in range(5):
            action = env1.action_space.sample()
            o1, r1, term1, trunc1, _ = env1.step(action)
            o2, r2, term2, trunc2, _ = env2.step(action)
            
            np.testing.assert_array_equal(o1["observation"], o2["observation"])
            np.testing.assert_array_equal(o1["valid_mask"], o2["valid_mask"])
            self.assertEqual(r1, r2)
            self.assertEqual(term1, term2)

    def test_state_saving_and_restoring(self):
        """Test state save/restore options."""
        env = GymCheckersEnv()
        env.reset(seed=42)
        
        env.step(np.array([5, 0, 4, 1], dtype=np.int32))
        _, _, _, _, info = env.step(np.array([2, 1, 3, 0], dtype=np.int32))
        saved_state = info["state"]
        
        new_env = GymCheckersEnv()
        new_env.reset(options={"state": saved_state})
        
        np.testing.assert_array_equal(new_env._grid, env._grid)
        self.assertEqual(new_env._current_player, env._current_player)
        self.assertEqual(new_env._active_jumper, env._active_jumper)
        self.assertEqual(new_env._total_moves, env._total_moves)
        self.assertEqual(new_env._draw_counter, env._draw_counter)
        self.assertEqual(new_env._game_draw_counter, env._game_draw_counter)
        self.assertEqual(new_env._move_history, env._move_history)

    def test_game_draw_40_move_rule(self):
        """Test the 40-move draw rule (80 plies without captures or normal piece moves)."""
        env = GymCheckersEnv()
        env.reset()
        
        # Setup board with only kings, so all moves are king moves
        env._grid = np.zeros((8, 8), dtype=np.int32)
        env._grid[0, 0] = P1_KING
        env._grid[7, 7] = P2_KING
        
        # Make 79 non-capturing king moves
        for i in range(79):
            env._current_player = 1 if (i % 2 == 0) else 2
            # Set game draw counter directly to simulate progress
            env._game_draw_counter = i
            # Move the king between (0,0) and (1,1) for P1, or (7,7) and (6,6) for P2
            if env._current_player == 1:
                action = np.array([0, 0, 1, 1], dtype=np.int32) if i % 4 in (0, 1) else np.array([1, 1, 0, 0], dtype=np.int32)
            else:
                action = np.array([7, 7, 6, 6], dtype=np.int32) if i % 4 in (0, 1) else np.array([6, 6, 7, 7], dtype=np.int32)
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertFalse(terminated)

        # 80th step should trigger the 40-move rule draw
        env._game_draw_counter = 79
        env._current_player = 1
        action = np.array([0, 0, 1, 1], dtype=np.int32)
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertTrue(terminated)
        self.assertEqual(reward, 0.0)

    def test_initial_blocked_state_termination(self):
        """Test that if a player is blocked at the start of a step, the game ends immediately."""
        env = GymCheckersEnv()
        env.reset()
        
        # Setup P1 blocked, P2 has pieces but it's P1's turn
        env._grid = np.zeros((8, 8), dtype=np.int32)
        env._grid[0, 0] = P1_NORMAL
        env._grid[1, 1] = P2_NORMAL
        env._current_player = 1
        
        # Taking any action should immediately terminate the game with a loss for P1 (reward -10.0)
        obs, reward, terminated, truncated, info = env.step(np.array([0, 0, 0, 0], dtype=np.int32))
        self.assertTrue(terminated)
        self.assertEqual(reward, -10.0)

    def test_symmetric_rewards(self):
        """Test that rewards/penalties are correctly sign-adjusted for zero-sum perspective."""
        env = GymCheckersEnv()
        env.reset()
        
        # Player 1's turn - invalid move: P1 is penalized (reward becomes more negative)
        env._current_player = 1
        # Set a clear valid moves list to ensure 0,0,0,0 is invalid
        _, reward_p1, _, _, _ = env.step(np.array([0, 0, 0, 0], dtype=np.int32))
        self.assertAlmostEqual(reward_p1, -0.11)  # step penalty (-0.01) + invalid penalty (-0.1)

        # Player 2's turn - invalid move: P2 is penalized, meaning reward increases (more positive for P1)
        env.reset()
        env._current_player = 2
        _, reward_p2, _, _, _ = env.step(np.array([0, 0, 0, 0], dtype=np.int32))
        self.assertAlmostEqual(reward_p2, 0.11)   # step penalty (+0.01) + invalid penalty (+0.1)


if __name__ == "__main__":
    unittest.main()

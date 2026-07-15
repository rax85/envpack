"""Tests for GymBattleshipEnv."""

import unittest
import numpy as np

from envpack.envs.game_battleship.env import GymBattleshipEnv


class TestGymBattleshipEnv(unittest.TestCase):
    """Tests for the GymBattleshipEnv environment."""

    def test_initial_state(self):
        """Test that the initial state is set up correctly."""
        env = GymBattleshipEnv()
        obs, _ = env.reset()

        self.assertEqual(env._current_player, 1)
        self.assertEqual(env._total_moves, 0)
        self.assertEqual(env._draw_counter, 0)

        # Check ship cells count
        p1_rem, p2_rem = env._get_ships_remaining()
        self.assertEqual(p1_rem, 17)
        self.assertEqual(p2_rem, 17)

        # Check observation structure
        self.assertIn("observation", obs)
        self.assertIn("valid_mask", obs)
        self.assertIn("current_player", obs)
        self.assertIn("ships_left", obs)

        # Check shapes
        self.assertEqual(obs["observation"].shape, (8, 8))
        self.assertEqual(obs["valid_mask"].shape, (8, 8))
        self.assertEqual(obs["current_player"], 1)
        np.testing.assert_array_equal(obs["ships_left"], [17, 17])

        # All cells initially valid to shoot
        np.testing.assert_array_equal(obs["valid_mask"], np.ones((8, 8), dtype=np.int8))

    def test_shooting_mechanics(self):
        """Test targeting empty cells and ships."""
        env = GymBattleshipEnv()
        env.reset()

        # Find an empty coordinate on Player 2's board to test a Miss
        empty_indices = np.argwhere(env._p2_board == 0)
        miss_coord = empty_indices[0]

        # Player 1 shoots and misses
        obs, reward, terminated, truncated, _ = env.step(miss_coord)
        self.assertEqual(env._p1_shots[miss_coord[0], miss_coord[1]], 2)  # 2 = Miss
        self.assertFalse(terminated)
        self.assertEqual(env._current_player, 2)  # Turn switched to P2

        # Find a ship coordinate on Player 1's board to test a Hit
        ship_indices = np.argwhere(env._p1_board == 1)
        hit_coord = ship_indices[0]

        # Player 2 shoots and hits
        obs, reward, terminated, truncated, _ = env.step(hit_coord)
        self.assertEqual(env._p2_shots[hit_coord[0], hit_coord[1]], 1)  # 1 = Hit
        self.assertEqual(env._current_player, 1)  # Turn switched back to P1

        p1_rem, p2_rem = env._get_ships_remaining()
        self.assertEqual(p1_rem, 16)
        self.assertEqual(p2_rem, 17)

    def test_invalid_repeated_shot(self):
        """Test shooting at the same coordinate twice returns invalid penalty."""
        env = GymBattleshipEnv()
        env.reset()

        coord = np.array([0, 0], dtype=np.int32)
        env.step(coord)

        # Active player (now P2) shoots at same cell -> valid since P2 hasn't shot there
        env.step(coord)

        # P1's turn again. Shooting same cell should trigger invalid penalty
        obs, reward, terminated, truncated, _ = env.step(coord)
        self.assertEqual(env._draw_counter, 1)
        # P1 is the shooter, so reward should have P1 invalid penalty (negative)
        self.assertLess(reward, 0)

    def test_gymnasium_compliance(self):
        """Test that the environment complies with Gymnasium specs."""
        from gymnasium.utils.env_checker import check_env
        env = GymBattleshipEnv()
        check_env(env, skip_render_check=True)

    def test_state_saving_and_restoring(self):
        """Test saving and restoring state works correctly."""
        env = GymBattleshipEnv()
        env.reset()

        env.step(np.array([0, 0], dtype=np.int32))
        _, _, _, _, info = env.step(np.array([0, 0], dtype=np.int32))

        saved_state = info["state"]

        new_env = GymBattleshipEnv()
        new_env.reset(options={"state": saved_state})

        np.testing.assert_array_equal(new_env._p1_board, env._p1_board)
        np.testing.assert_array_equal(new_env._p2_board, env._p2_board)
        np.testing.assert_array_equal(new_env._p1_shots, env._p1_shots)
        np.testing.assert_array_equal(new_env._p2_shots, env._p2_shots)
        self.assertEqual(new_env._current_player, env._current_player)


if __name__ == "__main__":
    unittest.main()

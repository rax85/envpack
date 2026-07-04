"""Tests for the Gym2048Env environment."""

# pylint: disable=protected-access
import copy
import unittest

import numpy as np

from PIL import Image

from envpack.envs.game_2048 import env as gym_2048_env
from envpack.envs.game_2048.env import Gym2048Env


class TestGym2048Env(unittest.TestCase):
    # pylint: disable=too-many-public-methods
    """Tests for the Gym2048Env environment."""

    def test_initial_state(self):
        """Test that the initial state has two non-zero tiles."""
        env = Gym2048Env()
        env.reset()
        self.assertEqual(np.count_nonzero(env._grid), 2)

    def test_pack_nomerge(self):
        """Test packing with no merges."""
        vals = [4, 0, 2, 0]
        env = Gym2048Env()
        packed, reward = env._pack(vals)
        self.assertTrue(np.array_equal(packed, [4, 2, 0, 0]))
        self.assertEqual(reward, 0)

    def test_pack_merge_single(self):
        """Test packing with a single merge."""
        vals = [2, 0, 2, 0]
        env = Gym2048Env()
        packed, reward = env._pack(vals)
        self.assertTrue(np.array_equal(packed, [4, 0, 0, 0]))
        self.assertEqual(reward, 4)

    def test_pack_merge_double(self):
        """Test packing with a double merge."""
        vals = [2, 2, 2, 2]
        env = Gym2048Env()
        packed, reward = env._pack(vals)
        self.assertTrue(np.array_equal(packed, [4, 4, 0, 0]))
        self.assertEqual(reward, 8)

    def test_pack_merge_slide(self):
        """Test packing with a merge and a slide."""
        vals = [2, 2, 0, 2]
        env = Gym2048Env()
        packed, reward = env._pack(vals)
        self.assertTrue(np.array_equal(packed, [4, 2, 0, 0]))
        self.assertEqual(reward, 4)

    def _save(self, env, name):
        """Helper to save a screenshot of the environment."""
        env._render()
        rgb_data = env.render()
        image = Image.fromarray(rgb_data)
        image.save(f"/tmp/test_{name}.png")

    def test_render(self):
        """Test rendering the environment."""
        env = Gym2048Env()
        env._grid = np.zeros(gym_2048_env.GRID_SIZE, dtype=np.int32)
        env._grid[0, 2] = 0
        env._grid[1, 2] = 64
        env._grid[2, 2] = 128
        env._grid[3, 2] = 2048
        self._save(env, "render")

    def test_move_nomerge_l(self):
        """Test moving left with no merges."""
        env = Gym2048Env()
        env._grid = np.zeros(gym_2048_env.GRID_SIZE, dtype=np.int32)
        env._grid[2, 1] = 128
        env._grid[2, 3] = 256
        self._save(env, "move_nomerge_l0")
        _observation, reward, done, _truncated, _info = env.step(gym_2048_env.LEFT)
        self._save(env, "move_nomerge_l1")

        self.assertEqual(env._grid[2, 0], 128)
        self.assertEqual(env._grid[2, 1], 256)
        self.assertFalse(done)
        self.assertEqual(reward, 0)

    def test_move_nomerge_r(self):
        """Test moving right with no merges."""
        env = Gym2048Env()
        env._grid = np.zeros(gym_2048_env.GRID_SIZE, dtype=np.int32)
        env._grid[2, 0] = 128
        env._grid[2, 2] = 256
        self._save(env, "test_move_nomerge_r0")
        _observation, reward, done, _truncated, _info = env.step(gym_2048_env.RIGHT)
        self._save(env, "test_move_nomerge_r1")

        self.assertEqual(env._grid[2, 2], 128)
        self.assertEqual(env._grid[2, 3], 256)
        self.assertFalse(done)
        self.assertEqual(reward, 0)

    def test_move_nomerge_u(self):
        """Test moving up with no merges."""
        env = Gym2048Env()
        env._grid = np.zeros(gym_2048_env.GRID_SIZE, dtype=np.int32)
        env._grid[1, 2] = 128
        env._grid[3, 2] = 256
        self._save(env, "test_move_nomerge_u0")
        _observation, reward, done, _truncated, _info = env.step(gym_2048_env.UP)
        self._save(env, "test_move_nomerge_u1")

        self.assertEqual(env._grid[0, 2], 128)
        self.assertEqual(env._grid[1, 2], 256)
        self.assertFalse(done)
        self.assertEqual(reward, 0)

    def test_move_nomerge_d(self):
        """Test moving down with no merges."""
        env = Gym2048Env()
        env._grid = np.zeros(gym_2048_env.GRID_SIZE, dtype=np.int32)
        env._grid[0, 2] = 128
        env._grid[2, 2] = 256
        self._save(env, "test_move_nomerge_d0")
        _observation, reward, done, _truncated, _info = env.step(gym_2048_env.DOWN)
        self._save(env, "test_move_nomerge_d1")

        self.assertEqual(env._grid[2, 2], 128)
        self.assertEqual(env._grid[3, 2], 256)
        self.assertFalse(done)
        self.assertEqual(reward, 0)

    def test_move_merge_l(self):
        """Test moving left with merges."""
        env = Gym2048Env()
        env._grid = np.zeros(gym_2048_env.GRID_SIZE, dtype=np.int32)
        env._grid[2, 1] = 128
        env._grid[2, 3] = 128
        self._save(env, "move_merge_l0")
        _observation, reward, done, _truncated, _info = env.step(gym_2048_env.LEFT)
        self._save(env, "move_merge_l1")

        self.assertEqual(env._grid[2, 0], 256)
        self.assertFalse(done)
        self.assertEqual(reward, 256)

    def test_move_merge_r(self):
        """Test moving right with merges."""
        env = Gym2048Env()
        env._grid = np.zeros(gym_2048_env.GRID_SIZE, dtype=np.int32)
        env._grid[2, 0] = 128
        env._grid[2, 2] = 128
        self._save(env, "move_merge_r0")
        _observation, reward, done, _truncated, _info = env.step(gym_2048_env.RIGHT)
        self._save(env, "move_merge_r1")

        self.assertEqual(env._grid[2, 3], 256)
        self.assertFalse(done)
        self.assertEqual(reward, 256)

    def test_move_merge_u(self):
        """Test moving up with merges."""
        env = Gym2048Env()
        env.reset(options={"nospawn": True})
        env._grid[1, 2] = 128
        env._grid[3, 2] = 128
        self._save(env, "move_merge_u0")
        _observation, reward, done, _truncated, _info = env.step(gym_2048_env.UP)
        self._save(env, "move_merge_u1")

        self.assertEqual(env._grid[0, 2], 256)
        self.assertFalse(done)
        self.assertEqual(reward, 256)

    def test_move_merge_d(self):
        """Test moving down with merges."""
        env = Gym2048Env()
        env.reset(options={"nospawn": True})
        env._grid[0, 2] = 128
        env._grid[2, 2] = 128
        self._save(env, "move_merge_d0")
        _observation, reward, done, _truncated, _info = env.step(gym_2048_env.DOWN)
        self._save(env, "move_merge_d1")

        self.assertEqual(env._grid[3, 2], 256)
        self.assertFalse(done)
        self.assertEqual(reward, 256)

    def test_move_doublemerge(self):
        """Test moving with a double merge."""
        env = Gym2048Env()
        env._grid = np.zeros_like(env._grid)
        env._grid[0, 2] = 128
        env._grid[1, 2] = 128
        env._grid[2, 2] = 128
        env._grid[3, 2] = 128
        self._save(env, "move_merge_double0")
        _observation, reward, done, _truncated, _info = env.step(gym_2048_env.DOWN)
        self._save(env, "move_merge_double1")

        self.assertEqual(env._grid[2, 2], 256)
        self.assertEqual(env._grid[3, 2], 256)
        self.assertFalse(done)
        self.assertEqual(reward, 512)

    def test_merge_first_slide_second_u(self):
        """Test merging first and then sliding second upwards."""
        env = Gym2048Env()
        env._grid = np.zeros(gym_2048_env.GRID_SIZE, dtype=np.int32)
        env._grid[0, 2] = 4
        env._grid[1, 2] = 2
        env._grid[2, 2] = 2
        env._grid[3, 2] = 2
        self._save(env, "test_merge_first_slide_second_u0")
        _observation, reward, done, _truncated, _info = env.step(gym_2048_env.UP)
        self._save(env, "test_merge_first_slide_second_u1")

        self.assertEqual(env._grid[0, 2], 4)
        self.assertEqual(env._grid[1, 2], 4)
        self.assertEqual(env._grid[2, 2], 2)
        self.assertFalse(done)
        self.assertEqual(reward, 4)

    def test_pack(self):
        """Test the _pack method."""
        env = Gym2048Env()
        actual, reward = env._pack([4, 2, 2, 2])
        self.assertTrue(np.array_equal([4, 4, 2, 0], actual))
        self.assertEqual(reward, 4)

    def test_can_pack_or_slide_pack(self):
        """Test _can_pack_or_slide with a pack scenario."""
        vals = [2, 2, 2, 2]
        env = Gym2048Env()
        self.assertTrue(env._can_pack_or_slide(vals))

    def test_can_pack_or_slide_slide(self):
        """Test _can_pack_or_slide with a slide scenario."""
        vals = [2, 0, 2, 2]
        env = Gym2048Env()
        self.assertTrue(env._can_pack_or_slide(vals))

    def test_can_pack_or_slide_neither(self):
        """Test _can_pack_or_slide with no pack or slide possible."""
        vals = [128, 64, 32, 16]
        env = Gym2048Env()
        self.assertFalse(env._can_pack_or_slide(vals))

    def test_can_pack_or_slide_leading_zeros(self):
        """Test _can_pack_or_slide with leading zeros."""
        vals = [0, 0, 2, 4]
        env = Gym2048Env()
        self.assertFalse(env._can_pack_or_slide(vals))

    def test_can_pack_or_slide_leading_zeros_single(self):
        """Test _can_pack_or_slide with a single leading zero."""
        vals = [0, 0, 0, 4]
        env = Gym2048Env()
        self.assertFalse(env._can_pack_or_slide(vals))

    def test_can_pack_or_slide_trailing_zeros(self):
        """Test _can_pack_or_slide with trailing zeros."""
        vals = [2, 4, 0, 0]
        env = Gym2048Env()
        self.assertTrue(env._can_pack_or_slide(vals))

    def test_can_pack_or_slide_leading_and_trailing_zeros(self):
        """Test _can_pack_or_slide with leading and trailing zeros."""
        vals = [0, 4, 0, 0]
        env = Gym2048Env()
        self.assertTrue(env._can_pack_or_slide(vals))

    def test_done(self):
        """Test the done condition."""
        env = Gym2048Env()
        _observation, _info = env.reset()
        self.assertIsNotNone(_observation)
        self.assertEqual(_observation["observation"].dtype, np.int32)
        self.assertEqual(_observation["valid_mask"].dtype, np.int32)
        env._grid = np.asarray(
            [[16, 4, 256, 32], [8, 32, 64, 4], [32, 128, 16, 2], [16, 8, 2, 2]],
            dtype=np.int32,
        )
        self._save(env, "move_merge_done0")
        _observation, reward, done, _truncated, _info = env.step(gym_2048_env.RIGHT)
        self._save(env, "move_merge_done1")

        self.assertEqual(reward, 4)
        self.assertTrue(done)
        self.assertIsNotNone(_observation["observation"])
        self.assertIsNotNone(_observation["valid_mask"])
        self.assertIsNotNone(_observation["total_score"])
        self.assertEqual(_observation["observation"].dtype, np.int32)

    def test_real_world1(self):
        """Test a real-world scenario 1."""
        env = Gym2048Env()
        env.reset()
        env._grid = np.asarray(
            [[0, 4, 8, 4], [0, 2, 8, 2], [0, 4, 8, 2], [2, 8, 4, 2]]
        )
        self._save(env, "test_real_world1_0")
        _observation, reward, done, _truncated, _info = env.step(gym_2048_env.UP)
        self._save(env, "test_real_world1_1")

        self.assertFalse(done)
        self.assertEqual(reward, 20)
        self.assertEqual(env._grid[0, 0], 2)
        self.assertEqual(env._grid[0, 1], 4)
        self.assertEqual(env._grid[0, 2], 16)
        self.assertEqual(env._grid[0, 3], 4)
        self.assertEqual(env._grid[1, 1], 2)
        self.assertEqual(env._grid[1, 2], 8)
        self.assertEqual(env._grid[1, 3], 4)
        self.assertEqual(env._grid[2, 1], 4)
        self.assertEqual(env._grid[2, 2], 4)
        self.assertEqual(env._grid[2, 3], 2)
        self.assertEqual(env._grid[3, 1], 8)

    def test_cant_move_r(self):
        """Test that the agent cannot move right."""
        env = Gym2048Env()
        env.reset()
        env._grid = np.asarray(
            [[0, 0, 0, 2], [0, 0, 0, 2], [0, 0, 0, 2], [0, 0, 0, 2]]
        )
        self.assertTrue(env._can_move(gym_2048_env.UP))
        self.assertTrue(env._can_move(gym_2048_env.DOWN))
        self.assertTrue(env._can_move(gym_2048_env.LEFT))
        print("interesting.......")
        self.assertFalse(env._can_move(gym_2048_env.RIGHT))

    def test_cant_move_u(self):
        """Test that the agent cannot move up."""
        env = Gym2048Env()
        env.reset()
        env._grid = np.asarray(
            [[2, 2, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
        print("interesting.......")
        self.assertFalse(env._can_move(gym_2048_env.UP))

    def test_cant_move_l(self):
        """Test that the agent cannot move left."""
        env = Gym2048Env()
        env.reset()
        env._grid = np.asarray(
            [[2, 0, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0]]
        )
        self.assertFalse(env._can_move(gym_2048_env.LEFT))

    def test_cant_move_d(self):
        """Test that the agent cannot move down."""
        env = Gym2048Env()
        env.reset()
        env._grid = np.asarray(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [2, 2, 2, 2]]
        )
        self.assertFalse(env._can_move(gym_2048_env.DOWN))

    def test_negative_reward(self):
        """Test that a negative reward is given when no move is possible."""
        env = Gym2048Env()
        env.reset()
        env._grid = np.asarray(
            [[0, 0, 0, 2], [0, 0, 0, 8], [0, 0, 2, 4], [0, 0, 0, 4]]
        )
        initial_state = copy.deepcopy(env._grid)
        self._save(env, "test_negative_reward_0")
        _observation, reward, done, _truncated, _info = env.step(gym_2048_env.RIGHT)
        self._save(env, "test_negative_reward1")

        self.assertFalse(done)
        self.assertEqual(reward, -32)
        self.assertTrue(np.array_equal(env._grid, initial_state))

    def test_score_accumulation(self):
        """Test that the score accumulates correctly."""
        env = Gym2048Env()
        env.reset()
        env._grid = np.asarray(
            [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
        self.assertEqual(env._score, 0)
        env.step(gym_2048_env.LEFT)
        self.assertEqual(env._score, 4)
        env._grid = np.asarray(
            [[4, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
        env.step(gym_2048_env.LEFT)
        self.assertEqual(env._score, 12)

    def test_observation_has_score(self):
        """Test that the observation contains the correct total score."""
        env = Gym2048Env()
        env.reset()
        env._grid = np.asarray(
            [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
        observation, _, _, _, _ = env.step(gym_2048_env.LEFT)
        self.assertEqual(observation["total_score"][0], 4)

        env._grid = np.asarray(
            [[4, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
        observation, _, _, _, _ = env.step(gym_2048_env.LEFT)
        self.assertEqual(observation["total_score"][0], 12)

    def test_game_over(self):
        """Test that the game over condition is met."""
        env = Gym2048Env()
        env._grid = np.asarray([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
        _observation, _reward, terminated, _truncated, _info = env.step(gym_2048_env.UP)
        self.assertTrue(terminated)

    def test_2048_termination(self):
        """Test that the game terminates when 2048 is reached."""
        env = Gym2048Env()
        env.reset(options={"nospawn": True})
        env._grid = np.zeros(gym_2048_env.GRID_SIZE, dtype=np.int32)
        env._grid[0, 0] = 1024
        env._grid[0, 1] = 1024
        _observation, reward, terminated, _truncated, _info = env.step(gym_2048_env.LEFT)
        self.assertEqual(reward, 2048)
        self.assertTrue(terminated)
        self.assertEqual(env._grid[0, 0], 2048)

    def test_move_tracking(self):
        """Test that the environment correctly tracks valid and invalid moves."""
        env = Gym2048Env()
        env.reset(options={"nospawn": True})
        # Ensure we have a known state
        env._grid = np.asarray(
            [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
        
        # Valid move (RIGHT)
        _, _, _, _, info = env.step(gym_2048_env.RIGHT)
        self.assertEqual(env._total_moves, 1)
        self.assertEqual(env._valid_moves, 1)
        self.assertEqual(env._invalid_moves, 0)
        self.assertEqual(len(env._move_history), 1)
        self.assertEqual(env._move_history[0], (gym_2048_env.RIGHT, True))
        
        # Check state in info
        state = info["state"]
        self.assertEqual(state["total_moves"], 1)
        self.assertEqual(state["valid_moves"], 1)
        self.assertEqual(state["move_history"][0], (gym_2048_env.RIGHT, True))

        # Force grid state to make RIGHT invalid
        env._grid = np.asarray(
            [[0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )

        # Invalid move (RIGHT again)
        env.step(gym_2048_env.RIGHT)
        self.assertEqual(env._total_moves, 2)
        self.assertEqual(env._valid_moves, 1)
        self.assertEqual(env._invalid_moves, 1)
        self.assertEqual(len(env._move_history), 2)
        self.assertEqual(env._move_history[1], (gym_2048_env.RIGHT, False))

        # Check history limit (8)
        for _ in range(10):
            env.step(gym_2048_env.UP)
        
        self.assertEqual(len(env._move_history), 8)

    def test_step_returns_state(self):
        """Test that step returns the state in info."""
        env = Gym2048Env()
        env.reset()
        _, _, _, _, info = env.step(gym_2048_env.UP)
        self.assertIn("state", info)
        state = info["state"]
        self.assertIn("grid", state)
        self.assertIn("score", state)
        self.assertIn("total_moves", state)
        self.assertIn("valid_moves", state)
        self.assertIn("invalid_moves", state)
        self.assertIn("move_history", state)

    def test_restore_state(self):
        """Test that the environment can be restored from a saved state."""
        env = Gym2048Env()
        env.reset()
        
        # Make some moves
        env.step(gym_2048_env.UP)
        env.step(gym_2048_env.LEFT)
        _, _, _, _, info = env.step(gym_2048_env.RIGHT)
        
        saved_state = info["state"]
        current_grid = env._grid.copy()
        current_score = env._score
        
        # Create a new environment and restore state
        new_env = Gym2048Env()
        new_env.reset(options={"state": saved_state})
        
        self.assertTrue(np.array_equal(new_env._grid, current_grid))
        self.assertEqual(new_env._score, current_score)
        self.assertEqual(new_env._total_moves, env._total_moves)
        self.assertEqual(new_env._valid_moves, env._valid_moves)
        self.assertEqual(new_env._invalid_moves, env._invalid_moves)
        self.assertEqual(new_env._move_history, env._move_history)

    def test_continue_from_restored_state(self):
        """Test that the game can be continued from a restored state."""
        env = Gym2048Env()
        env.reset(options={"nospawn": True})
        
        # Setup a specific state
        # 2 0 0 0
        # 0 0 0 0
        # 0 0 0 0
        # 0 0 0 0
        env._grid = np.zeros(gym_2048_env.GRID_SIZE, dtype=np.int32)
        env._grid[0, 0] = 2
        
        state = env._get_state()
        
        # Restore in new env
        new_env = Gym2048Env()
        new_env.reset(options={"state": state})
        
        # Move RIGHT
        # 0 0 0 2
        # ...
        new_env.step(gym_2048_env.RIGHT)
        
        self.assertEqual(new_env._grid[0, 3], 2)
        # Check that we have exactly 2 tiles now (one moved, one spawned)
        self.assertEqual(np.count_nonzero(new_env._grid), 2)

    def test_state_independence(self):
        """Test that modifying the returned state doesn't affect the environment."""
        env = Gym2048Env()
        env.reset()
        _, _, _, _, info = env.step(gym_2048_env.UP)
        
        state = info["state"]
        state["score"] = 99999
        state["grid"][0, 0] = 1234
        
        self.assertNotEqual(env._score, 99999)
        self.assertNotEqual(env._grid[0, 0], 1234)


if __name__ == "__main__":
    unittest.main()

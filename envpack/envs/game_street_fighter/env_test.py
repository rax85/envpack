"""Tests for GymStreetFighterEnv."""

import unittest
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from envpack.envs.game_street_fighter.env import (
    GymStreetFighterEnv,
    IDLE,
    WALK_LEFT,
    WALK_RIGHT,
    JUMP,
    CROUCH,
    PUNCH,
    KICK,
    SPECIAL_FIREBALL,
)


class TestGymStreetFighterEnv(unittest.TestCase):
    """Unit tests for GymStreetFighterEnv."""

    def test_initial_state(self):
        """Test initial positions, health, facing directions, and observation structure."""
        env = GymStreetFighterEnv()
        obs, info = env.reset()

        self.assertEqual(env.x[0], 100.0)
        self.assertEqual(env.x[1], 300.0)
        self.assertEqual(env.y_offset[0], 0.0)
        self.assertEqual(env.y_offset[1], 0.0)
        self.assertEqual(env.facing[0], 1)
        self.assertEqual(env.facing[1], -1)
        self.assertEqual(env.health[0], 100)
        self.assertEqual(env.health[1], 100)
        self.assertEqual(env.wins[0], 0)
        self.assertEqual(env.wins[1], 0)
        self.assertEqual(env.timer, 99)

        # Check observation keys
        self.assertIn("observation", obs)
        self.assertIn("valid_mask", obs)
        self.assertIn("health", obs)
        self.assertIn("total_score", obs)

        # Check shapes
        self.assertEqual(obs["observation"].shape, (300, 400, 3))
        self.assertEqual(obs["valid_mask"].shape, (2, 8))
        self.assertEqual(obs["health"].shape, (2,))
        self.assertEqual(obs["total_score"].shape, (2,))

        # Verify values
        np.testing.assert_array_equal(obs["valid_mask"], np.ones((2, 8), dtype=np.int8))
        np.testing.assert_array_equal(obs["health"], np.array([100, 100], dtype=np.int32))
        np.testing.assert_array_equal(obs["total_score"], np.array([0, 0], dtype=np.int32))

    def test_walking(self):
        """Test player walking updates positions correctly."""
        env = GymStreetFighterEnv()
        env.reset()

        # Step 1: P1 walks right, P2 walks left
        obs, reward, terminated, truncated, info = env.step(np.array([WALK_RIGHT, WALK_LEFT]))
        self.assertEqual(env.x[0], 100.0 + env.walk_speed)
        self.assertEqual(env.x[1], 300.0 - env.walk_speed)
        self.assertEqual(env.facing[0], 1)
        self.assertEqual(env.facing[1], -1)

        # Step 2: Walk P1 past P2 to test dynamic facing updates
        # Inject custom positions where P1 has crossed P2
        env.reset(options={"state": {"x": [250.0, 200.0]}})
        self.assertEqual(env.facing[0], -1)
        self.assertEqual(env.facing[1], 1)

    def test_jumping(self):
        """Test jumping physics and jump arc."""
        env = GymStreetFighterEnv()
        env.reset()

        # Start jump for P1 (stationary jump since last_horizontal_dir is 0)
        obs, reward, terminated, truncated, info = env.step(np.array([JUMP, IDLE]))
        self.assertEqual(env.state[0], "jump")
        self.assertGreater(env.y_offset[0], 0.0)
        self.assertEqual(env.vy[0], env.jump_vy)
        self.assertEqual(env.vx[0], 0.0)

        # Walk first to set last_horizontal_dir, then jump forward
        env.reset()
        env.step(np.array([WALK_RIGHT, IDLE]))  # set last_horizontal_dir[0] = 1
        env.step(np.array([JUMP, IDLE]))
        self.assertEqual(env.vx[0], env.walk_speed)

        # Step until player lands
        steps = 0
        while env.y_offset[0] > 0 and steps < 100:
            env.step(np.array([IDLE, IDLE]))
            steps += 1

        self.assertEqual(env.y_offset[0], 0.0)
        self.assertEqual(env.state[0], "idle")

    def test_combat_damage_and_hitstun(self):
        """Test that hits deal damage and apply hitstun."""
        env = GymStreetFighterEnv()
        # Place players close together: P1 at 200, P2 at 220
        env.reset(options={"state": {"x": [200.0, 220.0]}})

        # P1 Punches P2 (Reach 25, PUNCH action is 5)
        obs, reward, terminated, truncated, info = env.step(np.array([PUNCH, IDLE]))
        # P2 should be hit!
        self.assertEqual(env.health[1], 95)
        self.assertEqual(env.hitstun[1], 8)
        self.assertEqual(env.state[1], "hitstun")
        self.assertGreater(reward, 0)  # P1 got positive reward for dealing damage

    def test_blocking(self):
        """Test standing and crouching blocks."""
        env = GymStreetFighterEnv()

        # 1. Standing punch vs Standing block (P1 moves right, P2 moves away/right)
        env.reset(options={"state": {"x": [200.0, 220.0]}})
        # P2 walks right (away from P1) which is holding backward/blocking standing
        obs, reward, terminated, truncated, info = env.step(np.array([PUNCH, WALK_RIGHT]))
        # Damage should be blocked (no health loss)
        self.assertEqual(env.health[1], 100)
        self.assertEqual(env.hitstun[1], 0)

        # 2. Crouching punch vs Crouching block
        # Inject crouch state for P1 by stepping CROUCH once, then PUNCH
        env.reset(options={"state": {"x": [200.0, 220.0], "state": ["crouch", "idle"]}})
        # P1 crouching punch vs P2 crouch
        obs, reward, terminated, truncated, info = env.step(np.array([PUNCH, CROUCH]))
        self.assertEqual(env.health[1], 100)
        self.assertEqual(env.hitstun[1], 0)

    def test_fireball_cancellation(self):
        """Test that fireballs from both players cancel each other out."""
        env = GymStreetFighterEnv()
        env.reset(options={"state": {"x": [100.0, 300.0]}})

        # Spawn fireballs for both players
        env.step(np.array([SPECIAL_FIREBALL, SPECIAL_FIREBALL]))
        self.assertEqual(len(env.fireballs), 2)

        # Step until they collide or cross
        steps = 0
        while len(env.fireballs) == 2 and steps < 50:
            env.step(np.array([IDLE, IDLE]))
            steps += 1

        # They should have clashed and canceled out
        self.assertEqual(len(env.fireballs), 0)
        self.assertGreater(len(env.sparks), 0)  # Clash spark spawned

    def test_ko_and_round_reset(self):
        """Test that reducing health to 0 triggers knockdown and eventually resets round/ends match."""
        env = GymStreetFighterEnv()
        # Set Ken to 5 health, Ryu close to Ken
        env.reset(options={"state": {"x": [200.0, 220.0], "health": [100, 5]}})

        # Ryu punches Ken
        obs, reward, terminated, truncated, info = env.step(np.array([PUNCH, IDLE]))
        self.assertEqual(env.health[1], 0)
        self.assertEqual(env.state[1], "knockdown")
        self.assertEqual(env.knockdown[1], 20)  # knockdown timer starts

        # Step 20 times to let knockdown finish
        for _ in range(19):
            obs, reward, terminated, truncated, info = env.step(np.array([IDLE, IDLE]))
            self.assertFalse(terminated)

        # The 20th step of knockdown should reset the round
        obs, reward, terminated, truncated, info = env.step(np.array([IDLE, IDLE]))
        # Round 1 ended. P1 won round 1. Health reset to 100.
        self.assertEqual(env.wins[0], 1)
        self.assertEqual(env.health[0], 100)
        self.assertEqual(env.health[1], 100)

        # Now test match termination: set Ken to 5 health again and P1 wins count to 1
        env.reset(options={"state": {"x": [200.0, 220.0], "health": [100, 5], "wins": [1, 0]}})
        env.step(np.array([PUNCH, IDLE]))

        # Skip knockdown
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(np.array([IDLE, IDLE]))

        # Ryu gets 2nd win -> terminated = True
        self.assertEqual(env.wins[0], 2)
        self.assertTrue(terminated)

    def test_gymnasium_compliance(self):
        """Verify environment compliance with Gymnasium API."""
        env = GymStreetFighterEnv()
        check_env(env, skip_render_check=False)


if __name__ == "__main__":
    unittest.main()

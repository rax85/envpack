"""Tests for GymTowerDefenseEnv."""

import unittest
import numpy as np

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from envpack.envs.game_tower_defense.env import GymTowerDefenseEnv, IDLE


class TestGymTowerDefenseEnv(unittest.TestCase):
    """Tests for the GymTowerDefenseEnv Gymnasium environment."""

    def test_gym_compliance(self):
        """Test Gymnasium compliance using check_env."""
        env = GymTowerDefenseEnv()
        check_env(env, skip_render_check=True)

    def test_initial_state(self):
        """Test that the initial state is correct after reset."""
        env = GymTowerDefenseEnv()
        obs, info = env.reset()

        # Check observation structure
        self.assertIn("observation", obs)
        self.assertIn("valid_mask", obs)
        self.assertIn("health", obs)
        self.assertIn("gold", obs)

        # Check observation shapes and dtypes
        self.assertEqual(obs["observation"].shape, (300, 400, 3))
        self.assertEqual(obs["observation"].dtype, np.uint8)
        self.assertEqual(obs["valid_mask"].shape, (13,))
        self.assertEqual(obs["valid_mask"].dtype, np.int8)
        self.assertEqual(obs["health"].shape, (1,))
        self.assertEqual(obs["health"].dtype, np.int32)
        self.assertEqual(obs["gold"].shape, (1,))
        self.assertEqual(obs["gold"].dtype, np.int32)

        # Defaults
        self.assertEqual(env.health, 20)
        self.assertEqual(env.gold, 100)
        self.assertEqual(env.score, 0)
        self.assertEqual(env.wave, 1)

    def test_build_and_upgrade_towers(self):
        """Test building and upgrading Gun and Laser towers."""
        env = GymTowerDefenseEnv()
        env.reset()

        # Gold is 100. Build Gun Tower at Slot 0 (Action 1) costing 50 gold.
        obs, reward, term, trunc, info = env.step(1)
        self.assertIn(0, env.towers)
        self.assertEqual(env.towers[0]["type"], 1)
        self.assertEqual(env.towers[0]["level"], 1)
        self.assertEqual(env.gold, 50)

        # Upgrade Gun Tower at Slot 0 (Action 1) costing 40 gold.
        obs, reward, term, trunc, info = env.step(1)
        self.assertEqual(env.towers[0]["level"], 2)
        self.assertEqual(env.gold, 10)

        # Build Laser Tower at Slot 1 (Action 8: Build Laser at Slot 1). Cost is 80, we have 10.
        # Action is invalid/ignored.
        obs, reward, term, trunc, info = env.step(8)
        self.assertNotIn(1, env.towers)
        self.assertEqual(env.gold, 10)

    def test_custom_state_inject(self):
        """Test resetting the environment with a custom state injected."""
        env = GymTowerDefenseEnv()
        state = {
            "health": 15,
            "gold": 500,
            "score": 250,
            "wave": 4,
            "towers": [
                {"slot": 2, "type": 2, "level": 2, "cooldown": 0, "target_id": None},
            ],
            "enemies": [
                {"id": 1, "x": 100.0, "y": 50.0, "hp": 40.0, "max_hp": 50.0, "speed": 1.2, "target_idx": 1},
            ],
            "bullets": [
                {"x": 120.0, "y": 90.0, "target_id": 1, "damage": 4.0},
            ],
        }
        obs, info = env.reset(options={"state": state})
        self.assertEqual(env.health, 15)
        self.assertEqual(env.gold, 500)
        self.assertEqual(env.score, 250)
        self.assertEqual(env.wave, 4)
        self.assertIn(2, env.towers)
        self.assertEqual(env.towers[2]["type"], 2)
        self.assertEqual(env.towers[2]["level"], 2)
        self.assertEqual(len(env.enemies), 1)
        self.assertEqual(len(env.bullets), 1)

    def test_combat_and_enemy_escape(self):
        """Test laser attacks, bullets tracking, and enemy escaping exit."""
        env = GymTowerDefenseEnv()
        
        # Injected state:
        # Enemy is close to escape point: WAYPOINTS[5] = (400, 250)
        # Enemy is at (398, 250), moving at speed 2.0 towards waypoint 5.
        state = {
            "health": 20,
            "gold": 100,
            "score": 0,
            "wave": 1,
            "spawned_in_wave": 5,
            "towers": [
                # Laser Tower at Slot 4 (200, 210) has range 70 (Level 1).
                # Enemy is at (398, 250), distance is math.sqrt(198^2 + 40^2) > 70. Out of range.
                {"slot": 4, "type": 2, "level": 1, "cooldown": 0, "target_id": None},
            ],
            "enemies": [
                {"id": 1, "x": 399.0, "y": 250.0, "hp": 10.0, "max_hp": 10.0, "speed": 2.0, "target_idx": 5},
            ],
            "bullets": [],
        }
        env.reset(options={"state": state})

        # Advance 1 step: Enemy should cross the exit
        obs, reward, term, trunc, info = env.step(IDLE)
        self.assertEqual(env.health, 19)
        self.assertEqual(len(env.enemies), 0)
        self.assertEqual(reward, -10.0)

    def test_bullet_hit(self):
        """Test bullet travel and hitting an enemy."""
        env = GymTowerDefenseEnv()
        state = {
            "health": 20,
            "gold": 100,
            "score": 0,
            "wave": 1,
            "spawned_in_wave": 5,
            "towers": [],
            # Enemy at (100, 100) with HP 5.0
            "enemies": [
                {"id": 2, "x": 100.0, "y": 100.0, "hp": 5.0, "max_hp": 5.0, "speed": 0.0, "target_idx": 1},
            ],
            # Bullet at (100, 105) moving towards enemy at (100, 100) (dist 5 <= bullet_speed 10)
            "bullets": [
                {"x": 100.0, "y": 105.0, "target_id": 2, "damage": 5.0},
            ],
        }
        env.reset(options={"state": state})

        # Advance step: bullet should hit and kill the enemy, granting points and gold reward
        obs, reward, term, trunc, info = env.step(IDLE)
        self.assertEqual(len(env.enemies), 0)
        self.assertEqual(len(env.bullets), 0)
        # Gold reward = 15. Gold becomes 100 + 15 = 115.
        self.assertEqual(env.gold, 115)
        self.assertEqual(env.score, 10)
        self.assertEqual(reward, 15.0)

    def test_laser_continuous_damage(self):
        """Test laser tower continuous damage application."""
        env = GymTowerDefenseEnv()
        state = {
            "health": 20,
            "gold": 100,
            "score": 0,
            "wave": 1,
            "spawned_in_wave": 5,
            # Laser Tower at Slot 0: coordinates (100, 90). Level 1 damage is 0.3. Range is 70.
            "towers": [
                {"slot": 0, "type": 2, "level": 1, "cooldown": 0, "target_id": None},
            ],
            # Enemy at (100, 110). Distance is 20 (<= 70 range).
            "enemies": [
                {"id": 3, "x": 100.0, "y": 110.0, "hp": 10.0, "max_hp": 10.0, "speed": 0.0, "target_idx": 1},
            ],
            "bullets": [],
        }
        env.reset(options={"state": state})

        # Step: Laser damages enemy by 0.3 instantly. HP becomes 10.0 - 0.3 = 9.7.
        env.step(IDLE)
        self.assertAlmostEqual(env.enemies[3]["hp"], 9.7)

    def test_termination_loss(self):
        """Test losing the game when health drops to 0."""
        env = GymTowerDefenseEnv()
        state = {
            "health": 1,
            "gold": 100,
            "score": 0,
            "wave": 1,
            "spawned_in_wave": 5,
            "towers": [],
            "enemies": [
                {"id": 4, "x": 399.0, "y": 250.0, "hp": 10.0, "max_hp": 10.0, "speed": 2.0, "target_idx": 5},
            ],
            "bullets": [],
        }
        env.reset(options={"state": state})

        obs, reward, term, trunc, info = env.step(IDLE)
        self.assertEqual(env.health, 0)
        self.assertTrue(term)
        self.assertEqual(reward, -10.0 - 50.0) # escape penalty + loss penalty

import os
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image

# Import environments
from envpack.envs.game_2048.env import Gym2048Env
from envpack.envs.game_snake.env import GymSnakeEnv
from envpack.envs.game_tetris.env import GymTetrisEnv
from envpack.envs.game_sudoku.env import GymSudokuEnv
from envpack.envs.game_raptor.env import GymRaptorEnv
from envpack.envs.game_checkers.env import GymCheckersEnv
from envpack.envs.game_tron.env import GymTronEnv
from envpack.envs.game_air_hockey.env import GymAirHockeyEnv
from envpack.envs.game_racing.env import GymRacingEnv
from envpack.envs.game_doom.env import GymDoomEnv
from envpack.envs.game_paratrooper.env import GymParatrooperEnv
from envpack.envs.game_street_fighter.env import GymStreetFighterEnv
from envpack.envs.game_tank_combat.env import GymTankCombatEnv
from envpack.envs.game_gravity_duel.env import GymGravityDuelEnv
from envpack.envs.game_artillery_forts.env import GymArtilleryFortsEnv
from envpack.envs.game_pacman.env import GymPacmanEnv
from envpack.envs.game_platformer.env import GymPlatformerEnv
from envpack.envs.game_tower_defense.env import GymTowerDefenseEnv
from envpack.envs.game_asteroids.env import GymAsteroidsEnv
from envpack.envs.game_space_invaders.env import GymSpaceInvadersEnv
from envpack.envs.game_battleship.env import GymBattleshipEnv
from envpack.envs.game_drone.env import GymDroneEnv




def sample_action(env, obs):
    if isinstance(obs, dict) and "valid_mask" in obs:
        mask = obs["valid_mask"]
        
        # 1. Discrete action space
        if isinstance(env.action_space, spaces.Discrete):
            valid_indices = np.argwhere(mask > 0)
            if len(valid_indices) > 0:
                return int(np.random.choice(valid_indices.flatten()))
                
        # 2. MultiDiscrete action space
        elif isinstance(env.action_space, spaces.MultiDiscrete):
            # Case A: mask represents the joint action space (e.g. Checkers, Sudoku)
            if mask.shape == tuple(env.action_space.nvec):
                valid_indices = np.argwhere(mask > 0)
                if len(valid_indices) > 0:
                    idx = np.random.choice(len(valid_indices))
                    return valid_indices[idx].astype(env.action_space.dtype)
            
            # Case B: mask represents independent action spaces for each component (e.g. Street Fighter)
            elif len(mask.shape) == 2 and mask.shape[0] == len(env.action_space.nvec):
                action = []
                for i in range(len(env.action_space.nvec)):
                    component_mask = mask[i]
                    valid_indices = np.argwhere(component_mask > 0)
                    if len(valid_indices) > 0:
                        act = np.random.choice(valid_indices.flatten())
                    else:
                        act = np.random.randint(env.action_space.nvec[i])
                    action.append(act)
                return np.array(action, dtype=env.action_space.dtype)
                
    # Fallback to standard action space sampling
    return env.action_space.sample()


def generate_game_gif(env, name_prefix, num_frames=30, duration=500):
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    env.action_space.seed(42)
    
    obs, info = env.reset(seed=42)
    
    frames = []
    # Capture initial frame
    rgb_data = env.render()
    frames.append(Image.fromarray(rgb_data))
    
    for _ in range(num_frames - 1):
        action = sample_action(env, obs)
        obs, reward, done, truncated, info = env.step(action)
        
        rgb_data = env.render()
        frames.append(Image.fromarray(rgb_data))
        
        if done or truncated:
            obs, info = env.reset()
            
    # Save as animated GIF
    os.makedirs("screenshots", exist_ok=True)
    path = os.path.join("screenshots", f"{name_prefix}.gif")
    
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f"Saved {path}")


def main():
    envs = [
        (Gym2048Env(), "screenshot"),
        (GymSnakeEnv(), "snake_screenshot"),
        (GymTetrisEnv(), "tetris_screenshot"),
        (GymSudokuEnv(), "sudoku_screenshot"),
        (GymRaptorEnv(), "raptor_screenshot"),
        (GymCheckersEnv(), "checkers_screenshot"),
        (GymTronEnv(), "tron_screenshot"),
        (GymAirHockeyEnv(), "air_hockey_screenshot"),
        (GymRacingEnv(), "racing_screenshot"),
        (GymDoomEnv(), "doom_screenshot"),
        (GymParatrooperEnv(), "paratrooper_screenshot"),
        (GymStreetFighterEnv(), "street_fighter_screenshot"),
        (GymTankCombatEnv(), "tank_combat_screenshot"),
        (GymGravityDuelEnv(), "gravity_duel_screenshot"),
        (GymArtilleryFortsEnv(), "artillery_forts_screenshot"),
        (GymPacmanEnv(), "pacman_screenshot"),
        (GymPlatformerEnv(), "platformer_screenshot"),
        (GymTowerDefenseEnv(), "tower_defense_screenshot"),
        (GymAsteroidsEnv(), "asteroids_screenshot"),
        (GymSpaceInvadersEnv(), "space_invaders_screenshot"),
        (GymBattleshipEnv(), "battleship_screenshot"),
        (GymDroneEnv(), "drone_screenshot"),
    ]


    
    for env, name in envs:
        print(f"Generating {name}.gif...")
        generate_game_gif(env, name)


if __name__ == "__main__":
    main()

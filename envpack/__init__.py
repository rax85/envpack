"""Envpack: A collection of game environments for Gymnasium."""

from gymnasium.envs.registration import register

register(
    id="envpack/2048-v0",
    entry_point="envpack.envs.game_2048.env:Gym2048Env",
    max_episode_steps=300,
)

register(
    id="envpack/Snake-v0",
    entry_point="envpack.envs.game_snake.env:GymSnakeEnv",
    max_episode_steps=1000,
)

register(
    id="envpack/Tetris-v0",
    entry_point="envpack.envs.game_tetris.env:GymTetrisEnv",
    max_episode_steps=2000,
)

register(
    id="envpack/Sudoku-v0",
    entry_point="envpack.envs.game_sudoku.env:GymSudokuEnv",
    max_episode_steps=200,
)

register(
    id="envpack/Raptor-v0",
    entry_point="envpack.envs.game_raptor.env:GymRaptorEnv",
    max_episode_steps=1000,
)

register(
    id="envpack/Checkers-v0",
    entry_point="envpack.envs.game_checkers.env:GymCheckersEnv",
    max_episode_steps=500,
)

register(
    id="envpack/Tron-v0",
    entry_point="envpack.envs.game_tron.env:GymTronEnv",
    max_episode_steps=1000,
)

register(
    id="envpack/AirHockey-v0",
    entry_point="envpack.envs.game_air_hockey.env:GymAirHockeyEnv",
    max_episode_steps=1000,
)

register(
    id="envpack/Racing-v0",
    entry_point="envpack.envs.game_racing.env:GymRacingEnv",
    max_episode_steps=1000,
)

register(
    id="envpack/Doom-v0",
    entry_point="envpack.envs.game_doom.env:GymDoomEnv",
    max_episode_steps=500,
)

register(
    id="envpack/Paratrooper-v0",
    entry_point="envpack.envs.game_paratrooper.env:GymParatrooperEnv",
    max_episode_steps=1000,
)

register(
    id="envpack/StreetFighter-v0",
    entry_point="envpack.envs.game_street_fighter.env:GymStreetFighterEnv",
    max_episode_steps=1000,
)



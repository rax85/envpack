"""Envpack: A collection of game environments for Gymnasium."""

from gymnasium.envs.registration import register

register(
    id="envpack/2048-v0",
    entry_point="envpack.envs.game_2048.env:Gym2048Env",
    max_episode_steps=300,
)

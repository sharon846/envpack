"""Envpack: a simplified Pacman environment for Gymnasium."""

__version__ = "0.0.1"

from gymnasium.envs.registration import register

register(
    id="envpack/Pacman-v0",
    entry_point="envpack.envs.game_pacman.env:GymPacmanEnv",
)

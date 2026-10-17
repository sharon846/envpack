"""Tests for GymPacmanEnv."""

import unittest
from collections import deque

import numpy as np

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from envpack.envs.game_pacman.env import (
    GymPacmanEnv,
    MAZES,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    CHASER,
    AMBUSHER,
)

SIZES = sorted(MAZES)


class TestGymPacmanEnv(unittest.TestCase):
    """Tests for the GymPacmanEnv Gymnasium environment."""

    def test_gym_compliance(self):
        """Test Gymnasium compliance using check_env for every maze size."""
        for size in SIZES:
            with self.subTest(size=size):
                check_env(GymPacmanEnv(maze_size=size), skip_render_check=True)

    def test_gym_make(self):
        """Test the registered id accepts maze_size."""
        import envpack  # noqa: F401  (registers the envs)

        env = gym.make("envpack/Pacman-v0", maze_size=7)
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs["walls"].shape, (7, 7))

    def test_invalid_maze_size(self):
        with self.assertRaises(ValueError):
            GymPacmanEnv(maze_size=9)

    def test_initial_state(self):
        """Test the observation structure and initial state for every size."""
        for size in SIZES:
            with self.subTest(size=size):
                env = GymPacmanEnv(maze_size=size)
                obs, info = env.reset(seed=0)
                config = MAZES[size]

                self.assertEqual(
                    set(obs),
                    {"walls", "pellets", "pacman_position", "pacman_direction", "ghost_positions"},
                )
                self.assertEqual(obs["walls"].shape, (size, size))
                self.assertEqual(obs["walls"].dtype, np.int8)
                self.assertEqual(obs["pellets"].shape, (size, size))
                self.assertEqual(obs["pellets"].dtype, np.int8)
                self.assertEqual(obs["pacman_position"].shape, (2,))
                self.assertEqual(obs["pacman_position"].dtype, np.int32)
                self.assertEqual(obs["ghost_positions"].shape, (2, 2))
                self.assertEqual(obs["ghost_positions"].dtype, np.int32)
                self.assertTrue(env.observation_space.contains(obs))

                np.testing.assert_array_equal(obs["walls"], np.array(config["layout"]))
                self.assertEqual(tuple(obs["pacman_position"]), config["pacman_start"])
                self.assertEqual(obs["pacman_direction"], LEFT)
                self.assertEqual([tuple(p) for p in obs["ghost_positions"]], config["ghost_starts"])

                # Pellets on every open cell except Pacman's start
                expected = 1 - obs["walls"]
                expected[config["pacman_start"]] = 0
                np.testing.assert_array_equal(obs["pellets"], expected)
                self.assertEqual(info["pellets_eaten"], 0)
                self.assertEqual(info["pellets_total"], int(expected.sum()))
                self.assertIsNone(info["termination_reason"])

    def test_mazes_connected_without_dead_ends(self):
        """Every open cell is reachable from the start and has at least 2 exits."""
        for size in SIZES:
            with self.subTest(size=size):
                layout = np.array(MAZES[size]["layout"])
                open_cells = {tuple(c) for c in np.argwhere(layout == 0)}

                def neighbors(cell):
                    r, c = cell
                    return [n for n in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)] if n in open_cells]

                start = MAZES[size]["pacman_start"]
                seen = {start}
                queue = deque([start])
                while queue:
                    for n in neighbors(queue.popleft()):
                        if n not in seen:
                            seen.add(n)
                            queue.append(n)
                self.assertEqual(seen, open_cells)
                for cell in open_cells:
                    self.assertGreaterEqual(len(neighbors(cell)), 2, cell)
                for g in MAZES[size]["ghost_starts"]:
                    self.assertIn(g, open_cells)

    def _state(self, **overrides):
        state = {
            "pacman_pos": (3, 3),
            "pacman_dir": LEFT,
            "ghost_positions": [(1, 1), (1, 5)],
            "pellets": [(5, 5)],
        }
        state.update(overrides)
        return state

    def test_wall_keeps_pacman_in_place(self):
        env = GymPacmanEnv(maze_size=7, ghost_speed=0.0)
        # (5, 3) has a wall below it at (6, 3).
        env.reset(options={"state": self._state(pacman_pos=(5, 3), pacman_dir=LEFT)})
        obs, reward, term, trunc, info = env.step(DOWN)
        self.assertEqual(env.pacman_pos, (5, 3))
        self.assertEqual(env.pacman_dir, LEFT)
        self.assertEqual(info["action_mask"][DOWN], 0)
        self.assertAlmostEqual(reward, -0.01)

    def test_pellet_reward(self):
        env = GymPacmanEnv(maze_size=7, ghost_speed=0.0)
        env.reset(options={"state": self._state(pellets=[(3, 4), (5, 5)])})
        obs, reward, term, trunc, info = env.step(RIGHT)
        self.assertEqual(env.pacman_pos, (3, 4))
        self.assertEqual(env.pacman_dir, RIGHT)
        self.assertEqual(obs["pellets"][3, 4], 0)
        self.assertAlmostEqual(reward, 10.0 - 0.01)
        self.assertFalse(term)
        self.assertEqual(info["score"], 10)

    def test_clear_maze(self):
        env = GymPacmanEnv(maze_size=7, ghost_speed=0.0)
        env.reset(options={"state": self._state(pellets=[(3, 4)])})
        obs, reward, term, trunc, info = env.step(RIGHT)
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertEqual(info["termination_reason"], "cleared")
        self.assertAlmostEqual(reward, 510.0 - 0.01)

    def test_caught_walking_into_ghost(self):
        env = GymPacmanEnv(maze_size=7, ghost_speed=0.0)
        env.reset(options={"state": self._state(ghost_positions=[(3, 4), (1, 1)])})
        obs, reward, term, trunc, info = env.step(RIGHT)
        self.assertTrue(term)
        self.assertEqual(info["termination_reason"], "caught")
        self.assertAlmostEqual(reward, -100.0 - 0.01)

    def test_caught_by_ghost_move(self):
        env = GymPacmanEnv(maze_size=7, ghost_speed=1.0)
        # Pacman (3, 3) -> (3, 4); Chaser (3, 5) -> (3, 4): caught after ghosts move.
        env.reset(seed=0, options={"state": self._state(ghost_positions=[(3, 5), (1, 1)])})
        _, _, term, _, info = env.step(RIGHT)
        self.assertTrue(term)
        self.assertEqual(info["termination_reason"], "caught")

        # Pacman at (1, 3) pushes UP into a wall and stays; Chaser at (1, 4) moves in.
        env.reset(seed=0, options={"state": self._state(pacman_pos=(1, 3), ghost_positions=[(1, 4), (5, 1)])})
        _, reward, term, _, info = env.step(UP)
        self.assertTrue(term)
        self.assertEqual(info["termination_reason"], "caught")
        self.assertAlmostEqual(reward, -100.0 - 0.01)

    def test_chaser_closes_distance(self):
        env = GymPacmanEnv(maze_size=15, ghost_speed=1.0)
        env.reset(seed=0, options={"state": {
            "pacman_pos": (13, 1),
            "pacman_dir": LEFT,
            "ghost_positions": [(1, 13), (1, 1)],
            "pellets": [(1, 2)],
        }})
        prev = env._dist[env.ghost_positions[CHASER]][env.pacman_pos]
        for _ in range(5):
            env.step(LEFT)  # Pacman is in a corner facing a wall: stays put
            cur = env._dist[env.ghost_positions[CHASER]][env.pacman_pos]
            self.assertEqual(cur, prev - 1)
            prev = cur

    def test_ambusher_targets_ahead_of_pacman(self):
        env = GymPacmanEnv(maze_size=15)
        env.reset(options={"state": {
            "pacman_pos": (10, 4),
            "pacman_dir": RIGHT,
            "ghost_positions": [(1, 1), (1, 13)],
            "pellets": [(1, 2)],
        }})
        self.assertEqual(env._ambusher_target((1, 13)), (10, 8))
        # Lookahead stops at walls: facing LEFT from (10, 2) only reaches (10, 1)
        env.pacman_pos, env.pacman_dir = (10, 2), LEFT
        self.assertEqual(env._ambusher_target((1, 13)), (10, 1))
        # When close, the Ambusher targets Pacman directly
        env.pacman_pos, env.pacman_dir = (10, 4), RIGHT
        self.assertEqual(env._ambusher_target((10, 6)), (10, 4))

    def test_ambusher_moves_toward_ambush_cell(self):
        env = GymPacmanEnv(maze_size=15, ghost_speed=1.0)
        env.reset(seed=0, options={"state": {
            "pacman_pos": (13, 3),
            "pacman_dir": RIGHT,
            "ghost_positions": [(1, 1), (1, 13)],
            "pellets": [(1, 2)],
        }})
        target = env._ambusher_target(env.ghost_positions[AMBUSHER])
        self.assertEqual(target, (13, 7))
        before = env._dist[env.ghost_positions[AMBUSHER]][target]
        env.step(DOWN)  # Wall below: Pacman stays put and keeps facing RIGHT
        after = env._dist[env.ghost_positions[AMBUSHER]][target]
        self.assertEqual(after, before - 1)

    def test_truncation(self):
        env = GymPacmanEnv(maze_size=7, ghost_speed=0.0, max_steps=5)
        env.reset(seed=0)
        for i in range(5):
            _, _, term, trunc, info = env.step(UP if i % 2 else DOWN)
        self.assertFalse(term)
        self.assertTrue(trunc)
        self.assertEqual(info["termination_reason"], "max_steps")

    def test_seed_determinism(self):
        def rollout(seed):
            env = GymPacmanEnv(maze_size=11)
            obs, _ = env.reset(seed=seed)
            rng = np.random.default_rng(123)
            trace = []
            for _ in range(100):
                obs, reward, term, trunc, _ = env.step(int(rng.integers(4)))
                trace.append((tuple(obs["pacman_position"]), obs["ghost_positions"].tobytes(), reward))
                if term or trunc:
                    break
            return trace

        self.assertEqual(rollout(7), rollout(7))

    def test_render_modes(self):
        self.assertIsNone(GymPacmanEnv().render())
        for size in SIZES:
            with self.subTest(size=size):
                frame = GymPacmanEnv(render_mode="rgb_array", maze_size=size).render()
                self.assertEqual(frame.dtype, np.uint8)
                self.assertEqual(frame.ndim, 3)
                self.assertEqual(frame.shape[2], 3)

                text = GymPacmanEnv(render_mode="ansi", maze_size=size).render()
                lines = text.strip("\n").split("\n")
                self.assertEqual(len(lines), size)
                self.assertEqual(text.count("P"), 1)
                self.assertEqual(text.count("C"), 1)
                self.assertEqual(text.count("A"), 1)


if __name__ == "__main__":
    unittest.main()

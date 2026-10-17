"""A Gymnasium environment for a simplified Pacman.

Pacman must collect every pellet in a maze while avoiding two ghosts:
a Chaser that follows the shortest path to Pacman, and an Ambusher that
heads for a cell a few steps ahead of Pacman's direction of travel.
The observation is a structured description of the full game state.
"""

import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from absl import logging
from gymnasium import spaces
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Action Constants
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

DIR_VECTORS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}

# Ghost indices
CHASER = 0
AMBUSHER = 1

# Rewards
STEP_REWARD = -0.01
PELLET_REWARD = 10.0
CAUGHT_REWARD = -100.0
CLEAR_REWARD = 500.0

# 1 represents wall, 0 represents path
MAZE_LAYOUT_7 = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
]

MAZE_LAYOUT_11 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 0
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 1
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],  # 2
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 3
    [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],  # 4
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 5
    [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],  # 6
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 7
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],  # 8
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 9
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 10
]

MAZE_LAYOUT_15 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 0
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],  # 1
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],  # 2
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],  # 3
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 4
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],  # 5
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],  # 6
    [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1],  # 7
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],  # 8
    [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],  # 9
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 10
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],  # 11
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],  # 12
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 13
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 14
]

# Per-size configuration: layout, spawn cells, Ambusher lookahead and step limit.
MAZES = {
    7: {
        "layout": MAZE_LAYOUT_7,
        "pacman_start": (5, 3),
        "ghost_starts": [(1, 1), (1, 5)],
        "lookahead": 2,
        "max_steps": 200,
    },
    11: {
        "layout": MAZE_LAYOUT_11,
        "pacman_start": (9, 5),
        "ghost_starts": [(5, 4), (5, 6)],
        "lookahead": 3,
        "max_steps": 500,
    },
    15: {
        "layout": MAZE_LAYOUT_15,
        "pacman_start": (13, 7),
        "ghost_starts": [(7, 7), (7, 6)],
        "lookahead": 4,
        "max_steps": 1000,
    },
}

CELL_SIZE = 32
HUD_HEIGHT = 32

GHOST_COLORS = {
    CHASER: (255, 0, 0),  # Red
    AMBUSHER: (255, 105, 180),  # Pink
}


class GymPacmanEnv(gym.Env):
    """A Gymnasium environment for a simplified Pacman."""

    metadata = {"render_modes": ["rgb_array", "ansi"]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        maze_size: int = 15,
        ghost_speed: float = 0.8,
        max_steps: Optional[int] = None,
    ) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render_mode: {render_mode}")
        if maze_size not in MAZES:
            raise ValueError(f"maze_size must be one of {sorted(MAZES)}, got {maze_size}")
        if not 0.0 <= ghost_speed <= 1.0:
            raise ValueError(f"ghost_speed must be in [0, 1], got {ghost_speed}")

        self.render_mode = render_mode
        self.maze_size = maze_size
        self.ghost_speed = ghost_speed
        self.SF = 3  # Scale factor for SSAA

        config = MAZES[maze_size]
        self.pacman_start = config["pacman_start"]
        self.ghost_starts = list(config["ghost_starts"])
        self.lookahead = config["lookahead"]
        self.max_steps = max_steps if max_steps is not None else config["max_steps"]

        self._walls = np.array(config["layout"], dtype=np.int8)
        self.height, self.width = self._walls.shape
        self._open_cells = [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if self._walls[r, c] == 0
        ]
        self._dist = self._compute_distances()
        self.pellets_total = len(self._open_cells) - 1  # Pacman's start has no pellet

        # Action Space: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = spaces.Discrete(4)

        max_coord = maze_size - 1
        self.observation_space = spaces.Dict(
            {
                "walls": spaces.Box(low=0, high=1, shape=(self.height, self.width), dtype=np.int8),
                "pellets": spaces.Box(low=0, high=1, shape=(self.height, self.width), dtype=np.int8),
                "pacman_position": spaces.Box(low=0, high=max_coord, shape=(2,), dtype=np.int32),
                "pacman_direction": spaces.Discrete(4),
                "ghost_positions": spaces.Box(low=0, high=max_coord, shape=(2, 2), dtype=np.int32),
            }
        )

        self._hud_font = None
        if render_mode == "rgb_array":
            self._hud_font = self._load_font()

        self.reset()

    def _load_font(self) -> ImageFont.ImageFont:
        """Load the HUD font, scaled by SF for high-quality downsampling."""
        try:
            font_properties = font_manager.FontProperties(family="sans-serif", weight="bold")
            font_file = font_manager.findfont(font_properties)
            return ImageFont.truetype(font_file, 14 * self.SF)
        except Exception:
            logging.warning("Could not load system sans-serif font. Using default font.")
            try:
                return ImageFont.load_default(size=14 * self.SF)
            except Exception:
                return ImageFont.load_default()

    def _is_open(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width and self._walls[r, c] == 0

    def _neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        r, c = pos
        result = []
        for d in range(4):
            dr, dc = DIR_VECTORS[d]
            nxt = (r + dr, c + dc)
            if self._is_open(nxt):
                result.append(nxt)
        return result

    def _compute_distances(self) -> Dict[Tuple[int, int], Dict[Tuple[int, int], int]]:
        """All-pairs shortest path distances between open cells, via BFS."""
        dist = {}
        for src in self._open_cells:
            d = {src: 0}
            queue = deque([src])
            while queue:
                cur = queue.popleft()
                for nxt in self._neighbors(cur):
                    if nxt not in d:
                        d[nxt] = d[cur] + 1
                        queue.append(nxt)
            dist[src] = d
        return dist

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to the initial state."""
        super().reset(seed=seed)

        if options is not None and "state" in options:
            state = options["state"]
            self.pacman_pos = tuple(state["pacman_pos"])
            self.pacman_dir = state["pacman_dir"]
            self.ghost_positions = [tuple(p) for p in state["ghost_positions"]]
            self.pellets = set(map(tuple, state["pellets"]))
            self.steps = state.get("steps", 0)
            self.score = state.get("score", 0)
        else:
            self.pacman_pos = self.pacman_start
            self.pacman_dir = LEFT
            self.ghost_positions = list(self.ghost_starts)
            self.pellets = {cell for cell in self._open_cells if cell != self.pacman_start}
            self.steps = 0
            self.score = 0

        return self._get_obs(), self._get_info(None)

    def _get_valid_mask(self) -> np.ndarray:
        """Calculate valid directions from current Pacman position."""
        mask = np.zeros((4,), dtype=np.int8)
        r, c = self.pacman_pos
        for d in range(4):
            dr, dc = DIR_VECTORS[d]
            if self._is_open((r + dr, c + dc)):
                mask[d] = 1
        return mask

    def _get_obs(self) -> Dict[str, Any]:
        """Generate observation dict."""
        pellets = np.zeros((self.height, self.width), dtype=np.int8)
        for r, c in self.pellets:
            pellets[r, c] = 1
        return {
            "walls": self._walls.copy(),
            "pellets": pellets,
            "pacman_position": np.array(self.pacman_pos, dtype=np.int32),
            "pacman_direction": int(self.pacman_dir),
            "ghost_positions": np.array(self.ghost_positions, dtype=np.int32),
        }

    def _get_info(self, termination_reason: Optional[str]) -> Dict[str, Any]:
        pellets_eaten = self.pellets_total - len(self.pellets)
        return {
            "score": self.score,
            "steps": self.steps,
            "pellets_eaten": pellets_eaten,
            "pellets_total": self.pellets_total,
            "completion": pellets_eaten / self.pellets_total,
            "termination_reason": termination_reason,
            "action_mask": self._get_valid_mask(),
        }

    def _ambusher_target(self, ghost_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Target the farthest open cell up to `lookahead` steps ahead of Pacman.

        Falls back to Pacman itself once the Ambusher is close or already
        standing on its ambush cell, so it does not hover next to Pacman.
        """
        if self._dist[ghost_pos][self.pacman_pos] <= self.lookahead:
            return self.pacman_pos
        dr, dc = DIR_VECTORS[self.pacman_dir]
        target = self.pacman_pos
        for _ in range(self.lookahead):
            nxt = (target[0] + dr, target[1] + dc)
            if not self._is_open(nxt):
                break
            target = nxt
        if target == ghost_pos:
            return self.pacman_pos
        return target

    def _ghost_next_pos(self, ghost_pos: Tuple[int, int], target: Tuple[int, int]) -> Tuple[int, int]:
        """Step one cell along a shortest path to target, breaking ties randomly."""
        options = self._neighbors(ghost_pos)
        if not options:
            return ghost_pos
        best = min(self._dist[n][target] for n in options)
        candidates = [n for n in options if self._dist[n][target] == best]
        return candidates[self.np_random.integers(len(candidates))]

    def _is_caught(self) -> bool:
        return any(g == self.pacman_pos for g in self.ghost_positions)

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one step."""
        if not (0 <= action <= 3):
            raise ValueError(f"Invalid action: {action}")

        self.steps += 1
        reward = STEP_REWARD
        terminated = False
        termination_reason = None

        # 1. Move Pacman; walking into a wall leaves him in place.
        dr, dc = DIR_VECTORS[action]
        nxt = (self.pacman_pos[0] + dr, self.pacman_pos[1] + dc)
        if self._is_open(nxt):
            self.pacman_pos = nxt
            self.pacman_dir = action

        # 2. Eat pellet
        if self.pacman_pos in self.pellets:
            self.pellets.remove(self.pacman_pos)
            self.score += int(PELLET_REWARD)
            reward += PELLET_REWARD

        # 3. Pacman walked into a ghost, or cleared the maze
        if self._is_caught():
            terminated, termination_reason = True, "caught"
        elif not self.pellets:
            self.score += int(CLEAR_REWARD)
            reward += CLEAR_REWARD
            terminated, termination_reason = True, "cleared"

        # 4. Move ghosts, then check whether a ghost moved onto Pacman. Pacman
        # moves first, so a ghost and Pacman can never pass through each other.
        if not terminated:
            for i, ghost_pos in enumerate(self.ghost_positions):
                if self.np_random.random() >= self.ghost_speed:
                    continue
                if i == CHASER:
                    target = self.pacman_pos
                else:
                    target = self._ambusher_target(ghost_pos)
                self.ghost_positions[i] = self._ghost_next_pos(ghost_pos, target)

            if self._is_caught():
                terminated, termination_reason = True, "caught"

        if termination_reason == "caught":
            reward += CAUGHT_REWARD

        # 5. Step limit
        truncated = False
        if not terminated and self.steps >= self.max_steps:
            truncated, termination_reason = True, "max_steps"

        return self._get_obs(), float(reward), terminated, truncated, self._get_info(termination_reason)

    def render(self) -> Optional[Any]:
        """Render the environment for Gymnasium interface."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        if self.render_mode == "ansi":
            return self._render_ansi()
        return None

    def _render_ansi(self) -> str:
        """Render the maze as text: # wall, . pellet, P Pacman, C Chaser, A Ambusher."""
        grid = [
            ["#" if self._walls[r, c] else " " for c in range(self.width)]
            for r in range(self.height)
        ]
        for r, c in self.pellets:
            grid[r][c] = "."
        grid[self.pacman_pos[0]][self.pacman_pos[1]] = "P"
        for (r, c), symbol in zip(self.ghost_positions, ["C", "A"]):
            grid[r][c] = symbol
        return "\n".join("".join(row) for row in grid) + "\n"

    def _render_frame(self) -> np.ndarray:
        """Render on a supersampled canvas and downsample using LANCZOS."""
        if self._hud_font is None:
            self._hud_font = self._load_font()

        width = self.width * CELL_SIZE
        height = self.height * CELL_SIZE + HUD_HEIGHT
        canvas = Image.new("RGB", (width * self.SF, height * self.SF), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        cell_size = CELL_SIZE * self.SF
        x_offset = 0
        y_offset = HUD_HEIGHT * self.SF

        def cell_center(pos: Tuple[int, int]) -> Tuple[int, int]:
            r, c = pos
            return (
                x_offset + c * cell_size + cell_size // 2,
                y_offset + r * cell_size + cell_size // 2,
            )

        # 1. Draw Maze Walls
        for r in range(self.height):
            for c in range(self.width):
                if self._walls[r, c] == 1:
                    x1 = x_offset + c * cell_size
                    y1 = y_offset + r * cell_size
                    x2 = x1 + cell_size - 1
                    y2 = y1 + cell_size - 1
                    # Neon-style walls
                    draw.rectangle([x1, y1, x2, y2], fill=(10, 10, 30), outline=(25, 25, 255), width=2 * self.SF)

        # 2. Draw Pellets
        for pos in self.pellets:
            x, y = cell_center(pos)
            rad = 4 * self.SF
            draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=(255, 255, 150))

        # 3. Draw Pacman
        cx, cy = cell_center(self.pacman_pos)
        rad = 13 * self.SF

        # Draw full yellow circle
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(255, 255, 0))
        # Draw mouth if open
        if (self.steps // 3) % 2 == 0:
            angle_map = {
                0: -math.pi / 2,  # UP
                1: math.pi / 2,   # DOWN
                2: math.pi,       # LEFT
                3: 0,             # RIGHT
            }
            center_angle = angle_map[self.pacman_dir]
            a1 = center_angle - math.pi / 6
            a2 = center_angle + math.pi / 6
            p1 = (cx, cy)
            p2 = (cx + rad * 1.3 * math.cos(a1), cy + rad * 1.3 * math.sin(a1))
            p3 = (cx + rad * 1.3 * math.cos(a2), cy + rad * 1.3 * math.sin(a2))
            draw.polygon([p1, p2, p3], fill=(0, 0, 0))

        # 4. Draw Ghosts
        for i, pos in enumerate(self.ghost_positions):
            cx, cy = cell_center(pos)
            g_rad = 13 * self.SF
            color = GHOST_COLORS[i]

            # Draw Ghost shape (top half semicircle, bottom rectangle + spikes)
            draw.chord([cx - g_rad, cy - g_rad, cx + g_rad, cy + g_rad], start=180, end=360, fill=color)
            draw.rectangle([cx - g_rad, cy, cx + g_rad, cy + g_rad - 3 * self.SF], fill=color)

            # Bottom spikes
            w = 2 * g_rad
            h_spike = 3 * self.SF
            y_top = cy + g_rad - h_spike
            y_bot = cy + g_rad
            spikes = [
                (cx - g_rad, y_top),
                (cx - g_rad, y_bot),
                (cx - g_rad + w / 4, y_top),
                (cx - g_rad + w / 2, y_bot),
                (cx - g_rad + 3 * w / 4, y_top),
                (cx + g_rad, y_bot),
                (cx + g_rad, y_top),
            ]
            draw.polygon(spikes, fill=color)

            # Eyes looking towards Pacman
            dy = np.sign(self.pacman_pos[0] - pos[0])
            dx = np.sign(self.pacman_pos[1] - pos[1])
            ex_offset = dx * 2 * self.SF
            ey_offset = dy * 2 * self.SF
            for side in (-1, 1):
                e_cx = cx + side * 5 * self.SF + ex_offset
                e_cy = cy - 3 * self.SF + ey_offset
                draw.ellipse([e_cx - 3 * self.SF, e_cy - 3 * self.SF, e_cx + 3 * self.SF, e_cy + 3 * self.SF], fill=(255, 255, 255))
                p_cx = e_cx + ex_offset / 2
                p_cy = e_cy + ey_offset / 2
                draw.ellipse([p_cx - 1.5 * self.SF, p_cy - 1.5 * self.SF, p_cx + 1.5 * self.SF, p_cy + 1.5 * self.SF], fill=(0, 0, 255))

        # 5. Draw HUD
        hud_y = 8 * self.SF
        draw.text((8 * self.SF, hud_y), f"SCORE: {self.score}", fill=(255, 255, 255), font=self._hud_font)
        draw.text(((width - 8) * self.SF, hud_y), f"LEFT: {len(self.pellets)}", fill=(255, 255, 255), font=self._hud_font, anchor="rt")

        # Downsample using high-quality LANCZOS anti-aliasing
        canvas_resized = canvas.resize((width, height), Image.Resampling.LANCZOS)
        return np.array(canvas_resized, dtype=np.uint8)

    def close(self) -> None:
        """Close the environment."""
        pass

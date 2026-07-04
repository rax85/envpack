"""A Gymnasium environment for two-player simultaneous-move Tron Light Cycles."""

import copy
from typing import Any, Tuple, Dict, Optional, List

import gymnasium as gym
from absl import logging
from gymnasium import spaces
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import numpy.typing as npt

# Action Constants
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# Grid Constants
GRID_SIZE = 30
CELL_PX = 12
PADDING_PX = 8
HEADER_PX = 50
FOOTER_PX = 30

# Cell types
EMPTY = 0
P1_HEAD = 1
P1_TRAIL = 2
P2_HEAD = 3
P2_TRAIL = 4

# Colors
COLOR_BG = (15, 15, 20)
COLOR_GRID = (25, 25, 45)
COLOR_P1_HEAD = (0, 242, 254)       # Bright Cyan
COLOR_P1_TRAIL = (0, 150, 180)
COLOR_P2_HEAD = (255, 107, 0)       # Bright Orange
COLOR_P2_TRAIL = (180, 70, 0)
COLOR_HEADER = (22, 22, 30)
COLOR_FOOTER = (18, 18, 24)
COLOR_TEXT = (236, 240, 241)

CANVAS_SIZE = (
    GRID_SIZE * CELL_PX + 2 * PADDING_PX,
    GRID_SIZE * CELL_PX + 2 * PADDING_PX + HEADER_PX + FOOTER_PX,
)


class GymTronEnv(gym.Env):
    """A Gymnasium environment for two-player simultaneous Tron Light Cycles."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode

        # Font setup
        try:
            font_properties = font_manager.FontProperties(
                family="sans-serif", weight="bold"
            )
            font_file = font_manager.findfont(font_properties)
            self._title_font = ImageFont.truetype(font_file, 14)
            self._stats_font = ImageFont.truetype(font_file, 10)
        except Exception:
            logging.warning("Could not load system sans-serif font. Using default font.")
            self._title_font = ImageFont.load_default()
            self._stats_font = ImageFont.load_default()

        # Spaces
        # Joint action: [P1 action, P2 action]
        self.action_space = spaces.MultiDiscrete([4, 4])
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0, high=4, shape=(GRID_SIZE, GRID_SIZE), dtype=np.int32
                ),
                "valid_mask": spaces.Box(
                    low=0, high=1, shape=(2, 4), dtype=np.int8
                ),
                "total_score": spaces.Box(
                    low=0, high=1000, shape=(2,), dtype=np.int32
                ),
            }
        )

        # Pre-allocate static canvas
        self._base_canvas = Image.new("RGB", CANVAS_SIZE, COLOR_BG)
        draw_base = ImageDraw.Draw(self._base_canvas)
        draw_base.rectangle([0, 0, CANVAS_SIZE[0] - 1, HEADER_PX - 1], fill=COLOR_HEADER)
        draw_base.rectangle(
            [0, CANVAS_SIZE[1] - FOOTER_PX, CANVAS_SIZE[0] - 1, CANVAS_SIZE[1] - 1],
            fill=COLOR_FOOTER,
        )

        # Draw grid lines
        for i in range(GRID_SIZE + 1):
            coord = PADDING_PX + i * CELL_PX
            # Vertical
            draw_base.line(
                [(coord, HEADER_PX + PADDING_PX), (coord, HEADER_PX + PADDING_PX + GRID_SIZE * CELL_PX)],
                fill=COLOR_GRID,
                width=1,
            )
            # Horizontal
            draw_base.line(
                [(PADDING_PX, HEADER_PX + coord - PADDING_PX), (PADDING_PX + GRID_SIZE * CELL_PX, HEADER_PX + coord - PADDING_PX)],
                fill=COLOR_GRID,
                width=1,
            )

        self.reset()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment state."""
        super().reset(seed=seed)

        if options is not None and "state" in options:
            state = options["state"]
            self._grid = state["grid"].copy()
            self._p1_pos = tuple(state["p1_pos"])
            self._p2_pos = tuple(state["p2_pos"])
            self._p1_dir = state["p1_dir"]
            self._p2_dir = state["p2_dir"]
            self._steps = state["steps"]
            self._scores = np.array(state["scores"], dtype=np.int32)
            self._crash_pos = state["crash_pos"]
            return self._create_observation(), {}

        self._grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        
        # P1 starts mid-left moving Right
        self._p1_pos = (GRID_SIZE // 2, 4)
        self._p1_dir = RIGHT
        self._grid[self._p1_pos] = P1_HEAD

        # P2 starts mid-right moving Left
        self._p2_pos = (GRID_SIZE // 2, GRID_SIZE - 5)
        self._p2_dir = LEFT
        self._grid[self._p2_pos] = P2_HEAD

        self._steps = 0
        self._scores = np.zeros(2, dtype=np.int32)
        self._crash_pos: List[Tuple[int, int]] = []

        return self._create_observation(), {}

    def _get_opposite_direction(self, direction: int) -> int:
        """Returns the backward opposite of the current direction."""
        if direction == UP:
            return DOWN
        if direction == DOWN:
            return UP
        if direction == LEFT:
            return RIGHT
        return LEFT

    def _create_observation(self) -> Dict[str, Any]:
        """Create current state observation dictionary."""
        valid_mask = np.ones((2, 4), dtype=np.int8)
        
        # Block opposite directions to prevent suicide
        valid_mask[0, self._get_opposite_direction(self._p1_dir)] = 0
        valid_mask[1, self._get_opposite_direction(self._p2_dir)] = 0

        return {
            "observation": self._grid.copy(),
            "valid_mask": valid_mask,
            "total_score": self._scores.copy(),
        }

    def step(
        self, action: npt.NDArray[np.int32]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Advance one simultaneous step for both light cycles."""
        p1_act, p2_act = action
        if not (0 <= p1_act < 4 and 0 <= p2_act < 4):
            raise ValueError(f"Invalid actions: {action}")

        # If opposite action selected, force override to go straight
        p1_opp = self._get_opposite_direction(self._p1_dir)
        if p1_act == p1_opp:
            p1_act = self._p1_dir
        
        p2_opp = self._get_opposite_direction(self._p2_dir)
        if p2_act == p2_opp:
            p2_act = self._p2_dir

        self._p1_dir = p1_act
        self._p2_dir = p2_act
        self._steps += 1

        # Move vectors
        dirs = {
            UP: (-1, 0),
            DOWN: (1, 0),
            LEFT: (0, -1),
            RIGHT: (0, 1),
        }

        dr1, dc1 = dirs[self._p1_dir]
        dr2, dc2 = dirs[self._p2_dir]

        p1_next = (self._p1_pos[0] + dr1, self._p1_pos[1] + dc1)
        p2_next = (self._p2_pos[0] + dr2, self._p2_pos[1] + dc2)

        # Check crashes
        p1_crash = False
        p2_crash = False

        # Out of bounds
        if not (0 <= p1_next[0] < GRID_SIZE and 0 <= p1_next[1] < GRID_SIZE):
            p1_crash = True
        elif self._grid[p1_next] != EMPTY:
            p1_crash = True

        if not (0 <= p2_next[0] < GRID_SIZE and 0 <= p2_next[1] < GRID_SIZE):
            p2_crash = True
        elif self._grid[p2_next] != EMPTY:
            p2_crash = True

        # Head-on collision check
        if p1_next == p2_next:
            p1_crash = True
            p2_crash = True

        reward = 0.0
        terminated = False

        if p1_crash or p2_crash:
            terminated = True
            if p1_crash and p2_crash:
                # Draw
                reward = 0.0
                if 0 <= p1_next[0] < GRID_SIZE and 0 <= p1_next[1] < GRID_SIZE:
                    self._crash_pos.append(p1_next)
                if 0 <= p2_next[0] < GRID_SIZE and 0 <= p2_next[1] < GRID_SIZE:
                    self._crash_pos.append(p2_next)
            elif p1_crash:
                # Player 2 wins
                reward = -10.0
                self._scores[1] += 1
                if 0 <= p1_next[0] < GRID_SIZE and 0 <= p1_next[1] < GRID_SIZE:
                    self._crash_pos.append(p1_next)
            else:
                # Player 1 wins
                reward = 10.0
                self._scores[0] += 1
                if 0 <= p2_next[0] < GRID_SIZE and 0 <= p2_next[1] < GRID_SIZE:
                    self._crash_pos.append(p2_next)
        else:
            # Shift head to trail, move head
            self._grid[self._p1_pos] = P1_TRAIL
            self._grid[self._p2_pos] = P2_TRAIL
            
            self._p1_pos = p1_next
            self._p2_pos = p2_next
            
            self._grid[self._p1_pos] = P1_HEAD
            self._grid[self._p2_pos] = P2_HEAD

            # Small survival incentive
            reward = 0.01

        truncated = False
        return self._create_observation(), float(reward), terminated, truncated, {"state": self._get_state()}

    def _get_state(self) -> Dict[str, Any]:
        """Return environment internal state dictionary."""
        return {
            "grid": self._grid.copy(),
            "p1_pos": list(self._p1_pos),
            "p2_pos": list(self._p2_pos),
            "p1_dir": self._p1_dir,
            "p2_dir": self._p2_dir,
            "steps": self._steps,
            "scores": list(self._scores),
            "crash_pos": copy.deepcopy(self._crash_pos),
        }

    def render(self) -> Optional[npt.NDArray[np.uint8]]:
        """Render the environment to an RGB canvas array."""
        canvas = self._base_canvas.copy()
        draw = ImageDraw.Draw(canvas)

        # Header Info
        draw.text(
            (PADDING_PX + 5, HEADER_PX // 2),
            "TRON LIGHT CYCLES",
            fill=COLOR_TEXT,
            font=self._title_font,
            anchor="lm",
        )

        draw.text(
            (CANVAS_SIZE[0] - PADDING_PX - 5, HEADER_PX // 2),
            f"P1: {self._scores[0]} | P2: {self._scores[1]}",
            fill=COLOR_TEXT,
            font=self._title_font,
            anchor="rm",
        )

        # Draw heads and trails
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                cell = self._grid[r, c]
                if cell != EMPTY:
                    rx = PADDING_PX + c * CELL_PX
                    ry = HEADER_PX + PADDING_PX + r * CELL_PX

                    if cell == P1_HEAD:
                        draw.rectangle([rx, ry, rx + CELL_PX - 1, ry + CELL_PX - 1], fill=COLOR_P1_HEAD)
                    elif cell == P1_TRAIL:
                        draw.rectangle([rx + 2, ry + 2, rx + CELL_PX - 3, ry + CELL_PX - 3], fill=COLOR_P1_TRAIL)
                    elif cell == P2_HEAD:
                        draw.rectangle([rx, ry, rx + CELL_PX - 1, ry + CELL_PX - 1], fill=COLOR_P2_HEAD)
                    elif cell == P2_TRAIL:
                        draw.rectangle([rx + 2, ry + 2, rx + CELL_PX - 3, ry + CELL_PX - 3], fill=COLOR_P2_TRAIL)

        # Draw crash explosions
        for cr, cc in self._crash_pos:
            rx = PADDING_PX + cc * CELL_PX
            ry = HEADER_PX + PADDING_PX + cr * CELL_PX
            draw.ellipse([rx - 4, ry - 4, rx + CELL_PX + 3, ry + CELL_PX + 3], fill=(231, 76, 60))
            draw.ellipse([rx - 1, ry - 1, rx + CELL_PX, ry + CELL_PX], fill=(241, 196, 15))

        # Footer stats
        draw.text(
            (PADDING_PX + 5, CANVAS_SIZE[1] - FOOTER_PX // 2),
            f"Steps: {self._steps}",
            fill=(180, 180, 180),
            font=self._stats_font,
            anchor="lm",
        )

        return np.array(canvas, dtype=np.uint8)

    def close(self) -> None:
        """Close the environment."""
        pass

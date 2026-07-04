"""A Gym environment for playing Tetris."""

import random
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
LEFT = 0
RIGHT = 1
ROTATE = 2
DOWN = 3
HARD_DROP = 4

# Tetromino shapes (matrix representation)
SHAPES = {
    1: np.array([  # I
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.int32),
    2: np.array([  # O
        [1, 1],
        [1, 1]
    ], dtype=np.int32),
    3: np.array([  # T
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0]
    ], dtype=np.int32),
    4: np.array([  # S
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 0]
    ], dtype=np.int32),
    5: np.array([  # Z
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 0]
    ], dtype=np.int32),
    6: np.array([  # J
        [1, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ], dtype=np.int32),
    7: np.array([  # L
        [0, 0, 1],
        [1, 1, 1],
        [0, 0, 0]
    ], dtype=np.int32),
}

# Tetromino colors
COLORS = {
    0: (30, 30, 30),        # Empty
    1: (52, 152, 219),      # I: Cyan / Light Blue
    2: (241, 196, 15),      # O: Yellow
    3: (155, 89, 182),      # T: Purple
    4: (46, 204, 113),      # S: Green
    5: (231, 76, 60),       # Z: Red
    6: (41, 128, 185),      # J: Blue
    7: (230, 126, 34),      # L: Orange
}

GRID_SIZE = (20, 10)  # rows, cols
CELL_PX = 20
PADDING_PX = 1
HEADER_PX = 60
FOOTER_PX = 40

COLOR_BG = (20, 20, 20)
COLOR_GRID = (35, 35, 35)
COLOR_HEADER = (44, 62, 80)
COLOR_FOOTER = (52, 73, 94)
COLOR_TEXT_LIGHT = (236, 240, 241)

CANVAS_SIZE = (
    GRID_SIZE[1] * (CELL_PX + PADDING_PX) + PADDING_PX,
    GRID_SIZE[0] * (CELL_PX + PADDING_PX) + PADDING_PX + HEADER_PX + FOOTER_PX,
)


class GymTetrisEnv(gym.Env):
    """A Gym environment for playing Tetris."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode

        # Font setup
        font_properties = font_manager.FontProperties(
            family="sans-serif", weight="bold"
        )
        font_file = font_manager.findfont(font_properties)
        self._font = ImageFont.truetype(font_file, 14)
        self._score_font = ImageFont.truetype(font_file, 16)
        self._stats_font = ImageFont.truetype(font_file, 11)

        # Spaces
        n_actions = 5
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0, high=8, shape=GRID_SIZE, dtype=np.int32
                ),
                "valid_mask": spaces.Box(
                    low=0, high=1, shape=[n_actions], dtype=np.int32
                ),
                "total_score": spaces.Box(
                    low=0, high=1000000, shape=(1,), dtype=np.int32
                ),
            }
        )

        # Pre-calculated backgrounds
        self._background = np.full(
            (CANVAS_SIZE[1], CANVAS_SIZE[0], 3), COLOR_BG, dtype=np.uint8
        )
        self._background[0:HEADER_PX, :] = COLOR_HEADER
        self._background[CANVAS_SIZE[1] - FOOTER_PX:, :] = COLOR_FOOTER

        self.reset()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to the initial state."""
        super().reset(seed=seed)

        if options is not None and "state" in options:
            state = options["state"]
            self._board = state["board"].copy()
            self._score = state["score"]
            self._lines_cleared = state["lines_cleared"]
            self._total_moves = state["total_moves"]
            self._current_type = state["current_type"]
            self._current_shape = np.array(state["current_shape"], dtype=np.int32)
            self._current_pos = tuple(state["current_pos"])
            self._next_type = state["next_type"]
            self._move_history = copy.deepcopy(state["move_history"])

            observation, _ = self._create_observation()
            return observation, {}

        self._board = np.zeros(GRID_SIZE, dtype=np.int32)
        self._score = 0
        self._lines_cleared = 0
        self._total_moves = 0
        self._move_history: List[Tuple[int, bool]] = []

        # Spawn initial piece and preview piece
        self._next_type = random.choice(list(SHAPES.keys()))
        self._spawn_piece()

        observation, _ = self._create_observation()
        return observation, {}

    def _spawn_piece(self) -> bool:
        """Spawn the next piece. Returns False if spawning fails (Game Over)."""
        self._current_type = self._next_type
        self._current_shape = SHAPES[self._current_type].copy()
        
        # Spawn at top center
        start_x = (GRID_SIZE[1] - self._current_shape.shape[1]) // 2
        self._current_pos = (0, start_x)

        # Set next piece type
        self._next_type = random.choice(list(SHAPES.keys()))

        # Check if spawned piece collides immediately
        if self._check_collision(self._board, self._current_shape, self._current_pos):
            return False
        return True

    def _check_collision(self, board: npt.NDArray[np.int32], shape: npt.NDArray[np.int32], pos: Tuple[int, int]) -> bool:
        """Check if shape collides with walls or locked blocks on the board."""
        py, px = pos
        for r in range(shape.shape[0]):
            for c in range(shape.shape[1]):
                if shape[r, c] != 0:
                    by, bx = py + r, px + c
                    # Out of bounds checks
                    if by < 0 or by >= GRID_SIZE[0] or bx < 0 or bx >= GRID_SIZE[1]:
                        return True
                    # Collision check
                    if board[by, bx] != 0:
                        return True
        return False

    def _get_valid_mask(self) -> npt.NDArray[np.int32]:
        """Compute valid actions mask."""
        mask = np.zeros(5, dtype=np.int32)

        # LEFT (0)
        pos_left = (self._current_pos[0], self._current_pos[1] - 1)
        if not self._check_collision(self._board, self._current_shape, pos_left):
            mask[LEFT] = 1

        # RIGHT (1)
        pos_right = (self._current_pos[0], self._current_pos[1] + 1)
        if not self._check_collision(self._board, self._current_shape, pos_right):
            mask[RIGHT] = 1

        # ROTATE (2)
        shape_rot = np.rot90(self._current_shape, k=-1)
        if not self._check_collision(self._board, shape_rot, self._current_pos):
            mask[ROTATE] = 1

        # DOWN (3)
        pos_down = (self._current_pos[0] + 1, self._current_pos[1])
        if not self._check_collision(self._board, self._current_shape, pos_down):
            mask[DOWN] = 1

        # HARD_DROP (4)
        # Hard drop is always valid as long as current piece exists (it will drop at least 0 spaces)
        mask[HARD_DROP] = 1

        return mask

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Perform one step in the environment."""
        self._total_moves += 1
        valid_mask = self._get_valid_mask()
        
        # If action is invalid (except HARD_DROP which is always valid), ignore it
        is_valid = valid_mask[action] == 1
        self._move_history.append((action, is_valid))
        if len(self._move_history) > 8:
            self._move_history.pop(0)

        # Apply action
        if is_valid:
            if action == LEFT:
                self._current_pos = (self._current_pos[0], self._current_pos[1] - 1)
            elif action == RIGHT:
                self._current_pos = (self._current_pos[0], self._current_pos[1] + 1)
            elif action == ROTATE:
                self._current_shape = np.rot90(self._current_shape, k=-1)
            elif action == DOWN:
                self._current_pos = (self._current_pos[0] + 1, self._current_pos[1])

        terminated = False
        reward = 0.01  # Survival reward

        # If HARD_DROP: drop all the way down instantly
        if action == HARD_DROP:
            drop_dist = 0
            while not self._check_collision(self._board, self._current_shape, (self._current_pos[0] + 1, self._current_pos[1])):
                self._current_pos = (self._current_pos[0] + 1, self._current_pos[1])
                drop_dist += 1
            reward += 0.01 * drop_dist  # Bonus reward for dropping height
            # Lock instantly
            lines, game_over = self._lock_and_update()
            reward += self._get_line_reward(lines)
            if game_over:
                terminated = True
                reward = -1.0
        else:
            # Gravity: piece automatically slides down 1 block every step (even after left/right/rotate action)
            pos_gravity = (self._current_pos[0] + 1, self._current_pos[1])
            if self._check_collision(self._board, self._current_shape, pos_gravity):
                # Lock in place
                lines, game_over = self._lock_and_update()
                reward += self._get_line_reward(lines)
                if game_over:
                    terminated = True
                    reward = -1.0
            else:
                self._current_pos = pos_gravity

        truncated = False
        observation, _ = self._create_observation()
        return observation, float(reward), terminated, truncated, {"state": self._get_state()}

    def _get_line_reward(self, lines: int) -> float:
        """Returns reward based on lines cleared."""
        if lines == 0:
            return 0.0
        elif lines == 1:
            return 0.1
        elif lines == 2:
            return 0.3
        elif lines == 3:
            return 0.5
        else:
            return 1.0  # Tetris!

    def _lock_and_update(self) -> Tuple[int, bool]:
        """Locks the active piece, clears lines, updates score. Returns (lines_cleared, game_over)."""
        # Burn piece onto board
        py, px = self._current_pos
        for r in range(self._current_shape.shape[0]):
            for c in range(self._current_shape.shape[1]):
                if self._current_shape[r, c] != 0:
                    by, bx = py + r, px + c
                    if 0 <= by < GRID_SIZE[0] and 0 <= bx < GRID_SIZE[1]:
                        self._board[by, bx] = self._current_type

        # Check for completed lines
        cleared = 0
        r = GRID_SIZE[0] - 1
        while r >= 0:
            if np.all(self._board[r] != 0):
                self._board = np.delete(self._board, r, axis=0)
                self._board = np.insert(self._board, 0, np.zeros(GRID_SIZE[1], dtype=np.int32), axis=0)
                cleared += 1
            else:
                r -= 1

        self._lines_cleared += cleared
        # Classical scoring logic
        if cleared == 1:
            self._score += 100
        elif cleared == 2:
            self._score += 300
        elif cleared == 3:
            self._score += 500
        elif cleared == 4:
            self._score += 800

        # Spawn next piece
        spawn_success = self._spawn_piece()
        return cleared, not spawn_success

    def _get_state(self) -> Dict[str, Any]:
        """Return the current internal state of the environment."""
        return {
            "board": self._board.copy(),
            "score": self._score,
            "lines_cleared": self._lines_cleared,
            "total_moves": self._total_moves,
            "current_type": self._current_type,
            "current_shape": self._current_shape.tolist(),
            "current_pos": list(self._current_pos),
            "next_type": self._next_type,
            "move_history": copy.deepcopy(self._move_history),
        }

    def _create_observation(self) -> Tuple[Dict[str, Any], bool]:
        """Create the observation dictionary."""
        grid = self._board.copy()
        
        # Overlay the active falling piece as `8` in the observation grid
        py, px = self._current_pos
        for r in range(self._current_shape.shape[0]):
            for c in range(self._current_shape.shape[1]):
                if self._current_shape[r, c] != 0:
                    by, bx = py + r, px + c
                    if 0 <= by < GRID_SIZE[0] and 0 <= bx < GRID_SIZE[1]:
                        grid[by, bx] = 8

        valid_mask = self._get_valid_mask()
        done = not self._check_collision(self._board, self._current_shape, self._current_pos)
        
        return {
            "observation": grid,
            "valid_mask": valid_mask,
            "total_score": np.array([self._score], dtype=np.int32),
        }, not done

    def _draw_arrow(
        self, draw: ImageDraw.ImageDraw, x: int, y: int, action: int, color: Tuple[int, int, int]
    ) -> None:
        """Draw an action symbol in the footer."""
        radius = 5
        if action == LEFT:
            # Arrow Left
            points = [(x - radius, y), (x, y - 4), (x, y - 1.5), (x + radius, y - 1.5), (x + radius, y + 1.5), (x, y + 1.5), (x, y + 4)]
            draw.polygon(points, fill=color)
        elif action == RIGHT:
            # Arrow Right
            points = [(x + radius, y), (x, y - 4), (x, y - 1.5), (x - radius, y - 1.5), (x - radius, y + 1.5), (x, y + 1.5), (x, y + 4)]
            draw.polygon(points, fill=color)
        elif action == ROTATE:
            # Circular rotate arrow approximation
            draw.arc([x - 5, y - 5, x + 5, y + 5], start=0, end=270, fill=color, width=2)
            draw.polygon([(x + 5, y), (x + 2, y + 3), (x + 8, y + 3)], fill=color)
        elif action == DOWN:
            # Arrow Down
            points = [(x, y + radius), (x + 4, y), (x + 1.5, y), (x + 1.5, y - radius), (x - 1.5, y - radius), (x - 1.5, y), (x - 4, y)]
            draw.polygon(points, fill=color)
        elif action == HARD_DROP:
            # Hard drop represented by two arrows down or a solid double-arrow down
            points1 = [(x, y + radius), (x + 4, y), (x + 1.5, y), (x + 1.5, y - radius), (x - 1.5, y - radius), (x - 1.5, y), (x - 4, y)]
            draw.polygon(points1, fill=color)
            # Second tip lower
            points2 = [(x, y + radius + 4), (x + 4, y + 4), (x - 4, y + 4)]
            draw.polygon(points2, fill=color)

    def _render(self) -> None:
        """Update the current observation image."""
        canvas = Image.fromarray(self._background)
        draw = ImageDraw.Draw(canvas)

        # Draw Header
        draw.text(
            (10, HEADER_PX // 2),
            "TETRIS",
            fill=COLOR_TEXT_LIGHT,
            font=self._score_font,
            anchor="lm",
        )
        
        # Draw Score
        draw.text(
            (CANVAS_SIZE[0] - 65, HEADER_PX // 2),
            f"SC:{self._score}",
            fill=COLOR_TEXT_LIGHT,
            font=self._score_font,
            anchor="rm",
        )

        # Draw Next Piece Preview box in Header
        # Let's draw a tiny grid bounding box for Next
        next_box_x = CANVAS_SIZE[0] - 45
        next_box_y = HEADER_PX // 2 - 16
        draw.rectangle(
            [next_box_x, next_box_y, next_box_x + 32, next_box_y + 32],
            outline=COLOR_GRID,
            width=1
        )
        next_shape = SHAPES[self._next_type]
        # Draw Next Tetromino in the center of next_box
        # Offset to center
        cell_size = 6
        shape_w = next_shape.shape[1] * cell_size
        shape_h = next_shape.shape[0] * cell_size
        offset_x = next_box_x + (32 - shape_w) // 2
        offset_y = next_box_y + (32 - shape_h) // 2

        for r in range(next_shape.shape[0]):
            for c in range(next_shape.shape[1]):
                if next_shape[r, c] != 0:
                    cx = offset_x + c * cell_size
                    cy = offset_y + r * cell_size
                    draw.rectangle(
                        [cx, cy, cx + cell_size - 1, cy + cell_size - 1],
                        fill=COLORS[self._next_type],
                    )

        # Draw Grid background cell squares
        for y in range(GRID_SIZE[0]):
            for x in range(GRID_SIZE[1]):
                rx = x * (CELL_PX + PADDING_PX) + PADDING_PX
                ry = y * (CELL_PX + PADDING_PX) + PADDING_PX + HEADER_PX
                draw.rectangle(
                    [rx, ry, rx + CELL_PX - 1, ry + CELL_PX - 1],
                    fill=COLOR_GRID,
                )

        # Draw Static Board pieces
        for y in range(GRID_SIZE[0]):
            for x in range(GRID_SIZE[1]):
                val = self._board[y, x]
                if val != 0:
                    rx = x * (CELL_PX + PADDING_PX) + PADDING_PX
                    ry = y * (CELL_PX + PADDING_PX) + PADDING_PX + HEADER_PX
                    draw.rectangle(
                        [rx + 1, ry + 1, rx + CELL_PX - 2, ry + CELL_PX - 2],
                        fill=COLORS[val],
                    )

        # Draw Active Piece
        py, px = self._current_pos
        for r in range(self._current_shape.shape[0]):
            for c in range(self._current_shape.shape[1]):
                if self._current_shape[r, c] != 0:
                    by, bx = py + r, px + c
                    if 0 <= by < GRID_SIZE[0] and 0 <= bx < GRID_SIZE[1]:
                        rx = bx * (CELL_PX + PADDING_PX) + PADDING_PX
                        ry = by * (CELL_PX + PADDING_PX) + PADDING_PX + HEADER_PX
                        # Active piece is drawn slightly brighter or with a light border
                        draw.rectangle(
                            [rx + 1, ry + 1, rx + CELL_PX - 2, ry + CELL_PX - 2],
                            fill=COLORS[self._current_type],
                            outline=(255, 255, 255),
                            width=1,
                        )

        # Draw Footer Statistics
        stats_text = f"Lines: {self._lines_cleared}"
        draw.text(
            (10, CANVAS_SIZE[1] - FOOTER_PX + 12),
            stats_text,
            fill=COLOR_TEXT_LIGHT,
            font=self._stats_font,
        )

        # Draw move history symbols
        arrow_y = CANVAS_SIZE[1] - FOOTER_PX // 2
        arrow_x_start = CANVAS_SIZE[0] - 118
        arrow_spacing = 14
        for i, (action, is_valid) in enumerate(self._move_history):
            color = (46, 204, 113) if is_valid else (231, 76, 60)
            self._draw_arrow(draw, arrow_x_start + i * arrow_spacing, arrow_y, action, color)

        self._current_observation = np.array(canvas).astype(np.float32) / 256.0

    def render(self) -> Optional[npt.NDArray[np.uint8]]:
        """Return the current observation as an RGB array."""
        self._render()
        return (self._current_observation * 256).astype(np.uint8)

    def close(self) -> None:
        """Close the environment."""
        pass

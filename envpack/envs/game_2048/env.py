"""A Gym environment for playing the game 2048."""

# pylint: disable=too-many-instance-attributes, duplicate-code
import random
import copy

from typing import Any, Tuple, Dict, Optional, List

import gymnasium as gym
from absl import logging
from gymnasium import spaces
from matplotlib import font_manager
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
import numpy.typing as npt
import numba

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

GRID_SIZE = (4, 4)
PADDING_PX = 8
SQUARE_PX = 64
HEADER_PX = 80
FOOTER_PX = 64

RECT_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}

TEXT_COLOR_DARK = (119, 110, 101)
TEXT_COLOR_LIGHT = (249, 246, 242)

BACKGROUND_COLOR = (187, 173, 160)
FOOTER_COLOR = (60, 50, 50)

VALID_COLOR = (119, 175, 87)
INVALID_COLOR = (215, 84, 85)

CANVAS_SIZE = (
    GRID_SIZE[0] * (SQUARE_PX + PADDING_PX),
    GRID_SIZE[1] * (SQUARE_PX + PADDING_PX) + HEADER_PX + FOOTER_PX,
)


@numba.jit(nopython=True)
def _pack_jit(a: npt.NDArray[np.int32]) -> Tuple[npt.NDArray[np.int32], int]:
    non_zeros = a[a != 0]
    out = np.zeros(a.shape, dtype=a.dtype)
    reward = 0
    idx = 0
    i = 0
    length = len(non_zeros)
    while i < length:
        val = non_zeros[i]
        if i + 1 < length and val == non_zeros[i + 1]:
            merged_val = val * 2
            out[idx] = merged_val
            reward += merged_val
            i += 2
        else:
            out[idx] = val
            i += 1
        idx += 1
    return out, reward


@numba.jit(nopython=True)
def _can_pack_or_slide_jit(a: npt.NDArray[np.int32]) -> bool:
    length = len(a)
    last_nonzero_val = -1
    for i in range(length):
        val = a[i]
        if val != 0:
            if last_nonzero_val == val:
                return True
            last_nonzero_val = val

    has_nonzero = False
    for i in range(length):
        val = a[i]
        if val != 0:
            has_nonzero = True
        else:
            if has_nonzero:
                return True
    return False


@numba.jit(nopython=True)
def _get_valid_moves_jit(grid: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    moves = np.zeros(4, dtype=np.int32)

    # UP (0)
    can_up = False
    for i in range(4):
        col = grid[:, i]  # View of column i
        col_flipped = col[::-1]
        if _can_pack_or_slide_jit(col_flipped):
            can_up = True
            break
    moves[0] = 1 if can_up else 0

    # DOWN (1)
    can_down = False
    for i in range(4):
        col = grid[:, i]
        if _can_pack_or_slide_jit(col):
            can_down = True
            break
    moves[1] = 1 if can_down else 0

    # LEFT (2)
    can_left = False
    for i in range(4):
        row = grid[i]
        row_flipped = row[::-1]
        if _can_pack_or_slide_jit(row_flipped):
            can_left = True
            break
    moves[2] = 1 if can_left else 0

    # RIGHT (3)
    can_right = False
    for i in range(4):
        row = grid[i]
        if _can_pack_or_slide_jit(row):
            can_right = True
            break
    moves[3] = 1 if can_right else 0

    return moves


@numba.jit(nopython=True)
def _merge_jit(grid: npt.NDArray[np.int32], action: int) -> int:
    reward = 0
    if action == 0:  # UP
        for i in range(4):
            col = grid[:, i]
            packed, r = _pack_jit(col)
            reward += r
            grid[:, i] = packed
    elif action == 1:  # DOWN
        for i in range(4):
            col = grid[:, i]
            col_flipped = col[::-1]
            packed, r = _pack_jit(col_flipped)
            reward += r
            grid[:, i] = packed[::-1]
    elif action == 2:  # LEFT
        for i in range(4):
            row = grid[i]
            packed, r = _pack_jit(row)
            reward += r
            grid[i] = packed
    elif action == 3:  # RIGHT
        for i in range(4):
            row = grid[i]
            row_flipped = row[::-1]
            packed, r = _pack_jit(row_flipped)
            reward += r
            grid[i] = packed[::-1]
    return reward


class Gym2048Env(gym.Env):
    """A Gym environment for playing the game 2048."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        font_properties = font_manager.FontProperties(
            family="sans-serif", weight="bold"
        )
        font_file = font_manager.findfont(font_properties)
        logging.info("Loading font from %s", font_file)
        self._font = ImageFont.truetype(font_file, 24)
        self._score_label_font = ImageFont.truetype(font_file, 16)
        self._score_font = ImageFont.truetype(font_file, 24)
        self._stats_font = ImageFont.truetype(font_file, 12)
        self._game_over_font = ImageFont.truetype(font_file, 20)

        self._render_cache = {}
        for val in RECT_COLORS:
            self._generate_tile(val)

        n_actions = 4  # up, down, left, right.
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0, high=2**31 - 1, shape=GRID_SIZE, dtype=np.int32
                ),
                "valid_mask": spaces.Box(
                    low=0, high=1, shape=[n_actions], dtype=np.int32
                ),
                "total_score": spaces.Box(
                    low=0, high=2**31 - 1, shape=(1,), dtype=np.int32
                ),
            }
        )

        # Pre-calculate slices for rendering
        self._grid_slices = []
        for y in range(GRID_SIZE[1]):
            for x in range(GRID_SIZE[0]):
                rx = int(x * (SQUARE_PX + PADDING_PX) + PADDING_PX / 2)
                ry = int(y * (SQUARE_PX + PADDING_PX) + PADDING_PX / 2 + HEADER_PX)
                self._grid_slices.append(
                    (slice(ry, ry + SQUARE_PX), slice(rx, rx + SQUARE_PX))
                )

        # Pre-calculate positions for spawning
        self._all_positions = [
            (y, x) for y in range(GRID_SIZE[0]) for x in range(GRID_SIZE[1])
        ]

        self._background = (
            np.full(
                (CANVAS_SIZE[1], CANVAS_SIZE[0], 3), BACKGROUND_COLOR, dtype=np.uint8
            ).astype(np.float32)
            / 256.0
        )
        self._current_observation = self._background.copy()
        self.reset()
        logging.info("Canvas size is %s", CANVAS_SIZE)

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Perform one step in the environment."""
        reward = 0
        valid_moves = _get_valid_moves_jit(self._grid)
        is_valid = valid_moves[action] == 1
        self._total_moves += 1
        if is_valid:
            self._valid_moves += 1
            reward = _merge_jit(self._grid, action)
            self._score += reward
            self._random_spawn()
        else:
            self._invalid_moves += 1
            reward = -32

        self._move_history.append((action, is_valid))
        if len(self._move_history) > 8:
            self._move_history.pop(0)

        observation, terminated = self._create_observation()
        truncated = False
        return observation, float(reward), terminated, truncated, {"state": self._get_state()}

    def _get_state(self) -> Dict[str, Any]:
        """Return the current internal state of the environment."""
        return {
            "grid": self._grid.copy(),
            "score": self._score,
            "total_moves": self._total_moves,
            "valid_moves": self._valid_moves,
            "invalid_moves": self._invalid_moves,
            "move_history": copy.deepcopy(self._move_history),
        }

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to the initial state."""
        super().reset(seed=seed)

        if options is not None and "state" in options:
            state = options["state"]
            self._grid = state["grid"].copy()
            self._score = state["score"]
            self._total_moves = state["total_moves"]
            self._valid_moves = state["valid_moves"]
            self._invalid_moves = state["invalid_moves"]
            self._move_history = copy.deepcopy(state["move_history"])

            observation, _ = self._create_observation()
            return observation, {}

        self._grid = np.zeros(GRID_SIZE, dtype=np.int32)
        self._score = 0
        self._total_moves = 0
        self._valid_moves = 0
        self._invalid_moves = 0
        self._move_history: List[Tuple[int, bool]] = []
        spawn = True
        if options is not None and "nospawn" in options and options["nospawn"]:
            spawn = False
        if spawn:
            self._random_spawn()
            self._random_spawn()
        observation, _ = self._create_observation()
        return observation, {}

    def _random_spawn(self) -> bool:
        """Spawn a new tile at a random empty position."""
        candidates = [pos for pos in self._all_positions if self._grid[pos] == 0]
        if len(candidates) > 0:
            y, x = random.choice(candidates)
            self._grid[y, x] = random.choice([2, 4])
        return len(candidates) == 0

    def _generate_tile(self, val: int) -> None:
        """Generate the image for a tile value."""
        color = RECT_COLORS[val]
        img = Image.new("RGB", (SQUARE_PX, SQUARE_PX), color=color)
        draw = ImageDraw.Draw(img)
        if val > 0:
            text = f"{val}"
            text_color = TEXT_COLOR_DARK if val <= 4 else TEXT_COLOR_LIGHT
            draw.text(
                (SQUARE_PX / 2, SQUARE_PX / 2),
                text,
                fill=text_color,
                font=self._font,
                anchor="mm",
            )
        self._render_cache[val] = np.array(img).astype(np.float32) / 256.0

    def _draw_arrow(
        self, draw: ImageDraw.ImageDraw, x: int, y: int, action: int, color: Tuple[int, int, int]
    ) -> None:
        """Draw an arrow representing a move with a shaft and head."""
        # Arrow dimensions
        # Total size reduced: fits within roughly 16x16 box (radius 8)
        radius = 8
        shaft_w = 4
        # Points relative to center (x, y)
        
        if action == UP:
            # Pointing Up
            # Head tip at top
            points = [
                (x, y - radius),                   # Tip
                (x + 5, y),                        # Head Right Base
                (x + 2, y),                        # Shaft Top Right
                (x + 2, y + radius),               # Shaft Bottom Right
                (x - 2, y + radius),               # Shaft Bottom Left
                (x - 2, y),                        # Shaft Top Left
                (x - 5, y)                         # Head Left Base
            ]
        elif action == DOWN:
            # Pointing Down
            points = [
                (x, y + radius),                   # Tip
                (x + 5, y),                        # Head Right Base
                (x + 2, y),                        # Shaft Top Right
                (x + 2, y - radius),               # Shaft Bottom Right
                (x - 2, y - radius),               # Shaft Bottom Left
                (x - 2, y),                        # Shaft Top Left
                (x - 5, y)                         # Head Left Base
            ]
        elif action == LEFT:
            # Pointing Left
            points = [
                (x - radius, y),                   # Tip
                (x, y - 5),                        # Head Top Base
                (x, y - 2),                        # Shaft Top Left
                (x + radius, y - 2),               # Shaft Top Right
                (x + radius, y + 2),               # Shaft Bottom Right
                (x, y + 2),                        # Shaft Bottom Left
                (x, y + 5)                         # Head Bottom Base
            ]
        elif action == RIGHT:
            # Pointing Right
            points = [
                (x + radius, y),                   # Tip
                (x, y - 5),                        # Head Top Base
                (x, y - 2),                        # Shaft Top Left
                (x - radius, y - 2),               # Shaft Top Right
                (x - radius, y + 2),               # Shaft Bottom Right
                (x, y + 2),                        # Shaft Bottom Left
                (x, y + 5)                         # Head Bottom Base
            ]
        else:
            return
        draw.polygon(points, fill=color)

    def _render(self) -> None:
        """Update the current observation image."""
        self._current_observation = self._background.copy()

        # Draw Score Header
        header = Image.new("RGB", (CANVAS_SIZE[0], HEADER_PX), color=BACKGROUND_COLOR)
        draw = ImageDraw.Draw(header)

        # Check for Game Over
        if np.count_nonzero(_get_valid_moves_jit(self._grid)) == 0:
            go_box_width = 120
            go_box_height = 60
            margin = 10
            x_box_go = margin
            y_box_go = (HEADER_PX - go_box_height) // 2

            draw.text(
                (x_box_go + go_box_width / 2, y_box_go + 30),
                "GAME OVER",
                fill=(0, 0, 128),
                font=self._game_over_font,
                anchor="mm",
            )

        # Draw Score Box
        score_box_width = 120
        score_box_height = 60
        margin = 10
        x_box = CANVAS_SIZE[0] - score_box_width - margin
        y_box = (HEADER_PX - score_box_height) // 2

        draw.rectangle(
            [x_box, y_box, x_box + score_box_width, y_box + score_box_height],
            fill=(205, 193, 180),
        )

        # Draw "SCORE" label
        draw.text(
            (x_box + score_box_width / 2, y_box + 8),
            "SCORE",
            fill=(238, 228, 218),
            font=self._score_label_font,
            anchor="mt",
        )

        # Draw Score Value
        draw.text(
            (x_box + score_box_width / 2, y_box + 38),
            str(self._score),
            fill=(255, 255, 255),
            font=self._score_font,
            anchor="mm",
        )

        # Paste header
        header_arr = np.array(header).astype(np.float32) / 256.0
        self._current_observation[0:HEADER_PX, :] = header_arr

        for val, (s_y, s_x) in zip(self._grid.flat, self._grid_slices):
            self._current_observation[s_y, s_x] = self._render_cache[val]

        # Draw Footer
        footer_y_start = CANVAS_SIZE[1] - FOOTER_PX
        footer = Image.new("RGB", (CANVAS_SIZE[0], FOOTER_PX), color=FOOTER_COLOR)
        draw_footer = ImageDraw.Draw(footer)

        # Stats text
        stats_text = (
            f"Total: {self._total_moves}  "
            f"Valid: {self._valid_moves}  "
            f"Invalid: {self._invalid_moves}"
        )
        
        # Draw "Total" in light color, others in their colors
        # Need to draw them one by one for different colors
        x_off = 10
        y_off = 10
        
        txt = f"Total: {self._total_moves}"
        draw_footer.text((x_off, y_off), txt, fill=TEXT_COLOR_LIGHT, font=self._stats_font)
        x_off += draw_footer.textbbox((x_off, y_off), txt, font=self._stats_font)[2] - x_off + 15
        
        txt = f"Valid: {self._valid_moves}"
        draw_footer.text((x_off, y_off), txt, fill=VALID_COLOR, font=self._stats_font)
        x_off += draw_footer.textbbox((x_off, y_off), txt, font=self._stats_font)[2] - x_off + 15
        
        txt = f"Invalid: {self._invalid_moves}"
        draw_footer.text((x_off, y_off), txt, fill=INVALID_COLOR, font=self._stats_font)

        # Move history arrows
        arrow_y = y_off + 30
        arrow_x_start = 20
        arrow_spacing = 30
        for i, (action, is_valid) in enumerate(self._move_history):
            color = VALID_COLOR if is_valid else INVALID_COLOR
            self._draw_arrow(draw_footer, arrow_x_start + i * arrow_spacing, arrow_y, action, color)

        footer_arr = np.array(footer).astype(np.float32) / 256.0
        self._current_observation[footer_y_start:, :] = footer_arr

    def _create_observation(
        self, valid_moves: Optional[npt.NDArray[np.int32]] = None
    ) -> Tuple[Dict[str, Any], bool]:
        """Create the observation dictionary and check for termination."""
        if valid_moves is None:
            valid_moves = _get_valid_moves_jit(self._grid)
        done = (np.count_nonzero(valid_moves) == 0) or (np.any(self._grid == 2048))
        return {
            "observation": self._grid.copy(),
            "valid_mask": valid_moves,
            "total_score": np.array([self._score], dtype=np.int32),
        }, done

    def render(self) -> Optional[npt.NDArray[np.uint8]]:
        """Return the current observation as an RGB array."""
        self._render()
        return (self._current_observation * 256).astype(np.uint8)

    def _pack(self, a: npt.ArrayLike) -> Tuple[npt.NDArray[np.int32], int]:
        """Pack a row or column."""
        return _pack_jit(np.asarray(a))

    def _can_pack_or_slide(self, a: npt.ArrayLike) -> bool:
        """Check if a row or column can be packed or slid."""
        return _can_pack_or_slide_jit(np.asarray(a))

    def _can_move(self, action: int) -> bool:
        """Check if a specific move is valid."""
        return _get_valid_moves_jit(self._grid)[action] == 1

    def close(self) -> None:
        """Close the environment."""
"""A Gymnasium environment for Sudoku."""

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
ROW = 0
COL = 1
VAL = 2

# Grid Constants
GRID_SIZE = (9, 9)
CELL_PX = 48
PADDING_PX = 8
HEADER_PX = 80
FOOTER_PX = 40

# Colors
COLOR_BG = (30, 30, 30)
COLOR_GRID = (65, 65, 65)
COLOR_BOX_BORDER = (180, 180, 180) # Thick 3x3 borders
COLOR_CELL_BG = (45, 45, 45)
COLOR_GIVEN_TEXT = (236, 240, 241)     # Given/fixed numbers (White)
COLOR_CORRECT_TEXT = (46, 204, 113)    # Correctly placed (Green)
COLOR_CONFLICT_TEXT = (231, 76, 60)    # Violates constraint (Red)
COLOR_INCORRECT_TEXT = (241, 196, 15)  # Incorrect but non-conflicting (Yellow)
COLOR_HEADER = (52, 73, 94)
COLOR_FOOTER = (44, 62, 80)
COLOR_TEXT_LIGHT = (236, 240, 241)

CANVAS_SIZE = (
    GRID_SIZE[1] * CELL_PX + 2 * PADDING_PX,
    GRID_SIZE[0] * CELL_PX + 2 * PADDING_PX + HEADER_PX + FOOTER_PX,
)


class GymSudokuEnv(gym.Env):
    """A Gymnasium environment for solving Sudoku puzzles."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: Optional[str] = None, clues: int = 30) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.clues = clues

        # Font setup
        try:
            font_properties = font_manager.FontProperties(
                family="sans-serif", weight="bold"
            )
            font_file = font_manager.findfont(font_properties)
            self._cell_font = ImageFont.truetype(font_file, 22)
            self._score_font = ImageFont.truetype(font_file, 20)
            self._stats_font = ImageFont.truetype(font_file, 12)
        except Exception:
            logging.warning("Could not load system sans-serif font. Using default font.")
            self._cell_font = ImageFont.load_default()
            self._score_font = ImageFont.load_default()
            self._stats_font = ImageFont.load_default()

        # Spaces
        # Action space: MultiDiscrete([9, 9, 10]) representing [row, col, value]
        # value: 0 is clear, 1-9 are numbers placed on the board.
        self.action_space = spaces.MultiDiscrete([9, 9, 10])
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0, high=9, shape=GRID_SIZE, dtype=np.int32
                ),
                "given_mask": spaces.Box(
                    low=0, high=1, shape=GRID_SIZE, dtype=np.int8
                ),
                "valid_mask": spaces.Box(
                    low=0, high=1, shape=(9, 9, 10), dtype=np.int8
                ),
                "total_score": spaces.Box(
                    low=0, high=81, shape=(1,), dtype=np.int32
                ),
            }
        )

        # Pre-calculate base background
        self._background = np.full(
            (CANVAS_SIZE[1], CANVAS_SIZE[0], 3), COLOR_BG, dtype=np.uint8
        )
        self._background[0:HEADER_PX, :] = COLOR_HEADER
        self._background[CANVAS_SIZE[1] - FOOTER_PX:, :] = COLOR_FOOTER

        self._current_observation = self._background.copy()
        self.reset()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to the initial state."""
        super().reset(seed=seed)

        if options is not None and "state" in options:
            state = options["state"]
            self._grid = state["grid"].copy()
            self._solved_grid = state["solved_grid"].copy()
            self._given_mask = state["given_mask"].copy()
            self._score = state["score"]
            self._total_moves = state["total_moves"]
            self._invalid_moves = state["invalid_moves"]
            self._move_history = copy.deepcopy(state["move_history"])

            observation = self._create_observation()
            return observation, {}

        self._generate_puzzle()
        self._score = self._calculate_score()
        self._total_moves = 0
        self._invalid_moves = 0
        self._move_history: List[Tuple[Tuple[int, int, int], bool]] = []

        observation = self._create_observation()
        return observation, {}

    def _generate_puzzle(self) -> None:
        """Generate a valid Sudoku puzzle using self.np_random."""
        solved = np.zeros((9, 9), dtype=np.int32)
        
        # Fill diagonal 3x3 boxes randomly
        for i in range(0, 9, 3):
            nums = self.np_random.permutation(np.arange(1, 10))
            solved[i:i+3, i:i+3] = nums.reshape((3, 3))

        # Helper to check validity during grid generation
        def is_valid_gen(grid: npt.NDArray[np.int32], r: int, c: int, val: int) -> bool:
            if val in grid[r, :]:
                return False
            if val in grid[:, c]:
                return False
            br, bc = 3 * (r // 3), 3 * (c // 3)
            if val in grid[br:br+3, bc:bc+3]:
                return False
            return True

        # Backtracking solver to fill the rest of the board
        def fill(grid: npt.NDArray[np.int32]) -> bool:
            for r in range(9):
                for c in range(9):
                    if grid[r, c] == 0:
                        vals = self.np_random.permutation(np.arange(1, 10))
                        for val in vals:
                            if is_valid_gen(grid, r, c, val):
                                grid[r, c] = val
                                if fill(grid):
                                    return True
                                grid[r, c] = 0
                        return False
            return True

        fill(solved)
        self._solved_grid = solved

        # Remove elements to build starting clues
        puzzle = solved.copy()
        indices = list(range(81))
        self.np_random.shuffle(indices)
        
        to_remove = 81 - self.clues
        for idx in indices:
            if to_remove <= 0:
                break
            r, c = idx // 9, idx % 9
            puzzle[r, c] = 0
            to_remove -= 1

        self._grid = puzzle
        self._given_mask = (puzzle != 0).astype(np.int8)

    def _calculate_score(self) -> int:
        """Returns the number of cells matching the solved grid."""
        return int(np.sum(self._grid == self._solved_grid))

    def _has_conflict(self, r: int, c: int, val: int) -> bool:
        """Checks if placing val at (r, c) violates Sudoku row, col, or block constraints."""
        if val == 0:
            return False
        # Row check (excluding cell itself)
        if np.sum(self._grid[r, :] == val) - (1 if self._grid[r, c] == val else 0) > 0:
            return True
        # Col check
        if np.sum(self._grid[:, c] == val) - (1 if self._grid[r, c] == val else 0) > 0:
            return True
        # 3x3 block check
        br, bc = 3 * (r // 3), 3 * (c // 3)
        block = self._grid[br:br+3, bc:bc+3]
        if np.sum(block == val) - (1 if self._grid[r, c] == val else 0) > 0:
            return True
        return False

    def _get_valid_mask(self) -> npt.NDArray[np.int8]:
        """Compute values mask of shape (9, 9, 10)."""
        mask = np.zeros((9, 9, 10), dtype=np.int8)
        for r in range(9):
            for c in range(9):
                if self._given_mask[r, c] == 1:
                    continue  # Given cells cannot be edited (all values mask = 0)
                
                mask[r, c, 0] = 1  # Clearing/delete is always allowed
                
                # Check each value 1..9
                for v in range(1, 10):
                    # Check constraint violation
                    if not self._has_conflict_static(self._grid, r, c, v):
                        mask[r, c, v] = 1
        return mask

    @staticmethod
    def _has_conflict_static(grid: npt.NDArray[np.int32], r: int, c: int, val: int) -> bool:
        """Static conflict checker for valid mask calculation."""
        # Row check
        if np.sum(grid[r, :] == val) - (1 if grid[r, c] == val else 0) > 0:
            return True
        # Col check
        if np.sum(grid[:, c] == val) - (1 if grid[r, c] == val else 0) > 0:
            return True
        # 3x3 block check
        br, bc = 3 * (r // 3), 3 * (c // 3)
        block = grid[br:br+3, bc:bc+3]
        if np.sum(block == val) - (1 if grid[r, c] == val else 0) > 0:
            return True
        return False

    def step(
        self, action: npt.NDArray[np.int32]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Perform one step in the environment."""
        r, c, val = action
        if not (0 <= r < 9 and 0 <= c < 9 and 0 <= val < 10):
            raise ValueError(f"Invalid action: {action}")

        self._total_moves += 1

        # Check if cell is editable
        is_given = self._given_mask[r, c] == 1
        reward = -0.01  # Small step penalty to encourage speed

        if is_given:
            self._invalid_moves += 1
            self._move_history.append(((r, c, val), False))
        else:
            # Check if this move violates constraints (before applying to history)
            conflict_before = self._has_conflict(r, c, val)
            
            # Apply move
            old_val = self._grid[r, c]
            self._grid[r, c] = val
            
            new_score = self._calculate_score()
            score_diff = new_score - self._score
            self._score = new_score
            
            # Reward is proportional to score difference
            reward += float(score_diff)
            
            # Additional penalty if they create a constraint conflict
            if val != 0 and self._has_conflict(r, c, val):
                reward -= 0.1
                self._move_history.append(((r, c, val), False))
            else:
                self._move_history.append(((r, c, val), True))

        if len(self._move_history) > 8:
            self._move_history.pop(0)

        terminated = self._score == 81
        if terminated:
            reward += 10.0  # Big bonus for solving the puzzle!

        truncated = False
        observation = self._create_observation()
        return observation, float(reward), terminated, truncated, {"state": self._get_state()}

    def _get_state(self) -> Dict[str, Any]:
        """Return the current internal state of the environment."""
        return {
            "grid": self._grid.copy(),
            "solved_grid": self._solved_grid.copy(),
            "given_mask": self._given_mask.copy(),
            "score": self._score,
            "total_moves": self._total_moves,
            "invalid_moves": self._invalid_moves,
            "move_history": copy.deepcopy(self._move_history),
        }

    def _create_observation(self) -> Dict[str, Any]:
        """Create the observation dictionary."""
        return {
            "observation": self._grid.copy(),
            "given_mask": self._given_mask.copy(),
            "valid_mask": self._get_valid_mask(),
            "total_score": np.array([self._score], dtype=np.int32),
        }

    def _draw_symbol(
        self, draw: ImageDraw.ImageDraw, x: int, y: int, action: Tuple[int, int, int], is_valid: bool
    ) -> None:
        """Draw a representation of a move in the footer history."""
        r, c, val = action
        text = f"({r},{c})={val}"
        color = COLOR_CORRECT_TEXT if is_valid else COLOR_CONFLICT_TEXT
        draw.text((x, y), text, fill=color, font=self._stats_font, anchor="lm")

    def _render(self) -> None:
        """Update the current observation image."""
        canvas = Image.fromarray(self._background)
        draw = ImageDraw.Draw(canvas)

        # Draw Header Texts
        draw.text(
            (PADDING_PX + 5, HEADER_PX // 2),
            "SUDOKU",
            fill=COLOR_TEXT_LIGHT,
            font=self._score_font,
            anchor="lm",
        )

        draw.text(
            (CANVAS_SIZE[0] - PADDING_PX - 5, HEADER_PX // 2),
            f"SCORE: {self._score}/81",
            fill=COLOR_TEXT_LIGHT,
            font=self._score_font,
            anchor="rm",
        )

        # Draw Cells
        for r in range(9):
            for c in range(9):
                rx = PADDING_PX + c * CELL_PX
                ry = HEADER_PX + PADDING_PX + r * CELL_PX
                
                # Draw cell background
                draw.rectangle(
                    [rx, ry, rx + CELL_PX - 1, ry + CELL_PX - 1],
                    fill=COLOR_CELL_BG,
                )

                # Draw cell borders
                draw.rectangle(
                    [rx, ry, rx + CELL_PX - 1, ry + CELL_PX - 1],
                    outline=COLOR_GRID,
                    width=1,
                )

                # Draw values
                val = self._grid[r, c]
                if val != 0:
                    # Decide color
                    if self._given_mask[r, c] == 1:
                        color = COLOR_GIVEN_TEXT
                    elif val == self._solved_grid[r, c]:
                        color = COLOR_CORRECT_TEXT
                    elif self._has_conflict(r, c, val):
                        color = COLOR_CONFLICT_TEXT
                    else:
                        color = COLOR_INCORRECT_TEXT
                    
                    draw.text(
                        (rx + CELL_PX // 2, ry + CELL_PX // 2),
                        str(val),
                        fill=color,
                        font=self._cell_font,
                        anchor="mm",
                    )

        # Highlight 3x3 block borders
        for i in range(4):
            # Vertical thick borders
            vx = PADDING_PX + i * 3 * CELL_PX
            draw.line(
                [(vx, HEADER_PX + PADDING_PX), (vx, HEADER_PX + PADDING_PX + 9 * CELL_PX)],
                fill=COLOR_BOX_BORDER,
                width=3 if 0 < i < 3 else 1,
            )
            # Horizontal thick borders
            hy = HEADER_PX + PADDING_PX + i * 3 * CELL_PX
            draw.line(
                [(PADDING_PX, hy), (PADDING_PX + 9 * CELL_PX, hy)],
                fill=COLOR_BOX_BORDER,
                width=3 if 0 < i < 3 else 1,
            )

        # Draw Footer Statistics
        stats_text = f"Moves: {self._total_moves}  Invalid: {self._invalid_moves}"
        draw.text(
            (PADDING_PX + 5, CANVAS_SIZE[1] - FOOTER_PX + 12),
            stats_text,
            fill=COLOR_TEXT_LIGHT,
            font=self._stats_font,
        )

        # Draw move history
        arrow_y = CANVAS_SIZE[1] - FOOTER_PX // 2
        arrow_x_start = CANVAS_SIZE[0] - 250
        arrow_spacing = 60
        for i, (action, is_valid) in enumerate(self._move_history):
            self._draw_symbol(draw, arrow_x_start + i * arrow_spacing, arrow_y, action, is_valid)

        self._current_observation = np.array(canvas, dtype=np.uint8)

    def render(self) -> Optional[npt.NDArray[np.uint8]]:
        """Return the current observation as an RGB array."""
        self._render()
        return self._current_observation.copy()

    def close(self) -> None:
        """Close the environment."""
        pass

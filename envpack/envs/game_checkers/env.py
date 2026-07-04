"""A Gymnasium environment for two-player Checkers (Draughts)."""

import copy
from typing import Any, Tuple, Dict, Optional, List

import gymnasium as gym
from absl import logging
from gymnasium import spaces
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import numpy.typing as npt

# Action Space: MultiDiscrete([8, 8, 8, 8]) representing [from_row, from_col, to_row, to_col]
# Grid Constants
GRID_ROWS = 8
GRID_COLS = 8
CELL_PX = 48
PADDING_PX = 8
HEADER_PX = 60
FOOTER_PX = 40

# Cell types
EMPTY = 0
P1_NORMAL = 1
P1_KING = 2
P2_NORMAL = 3
P2_KING = 4

# Colors
COLOR_BG = (30, 30, 30)
COLOR_DARK_SQUARE = (45, 45, 45)    # Playable
COLOR_LIGHT_SQUARE = (70, 70, 70)   # Unplayable
COLOR_GRID = (55, 55, 55)
COLOR_P1 = (52, 152, 219)           # Cyan
COLOR_P2 = (155, 89, 182)          # Magenta
COLOR_CROWN = (241, 196, 15)        # Gold
COLOR_HEADER = (52, 73, 94)
COLOR_FOOTER = (44, 62, 80)
COLOR_TEXT = (236, 240, 241)

CANVAS_SIZE = (
    GRID_COLS * CELL_PX + 2 * PADDING_PX,
    GRID_ROWS * CELL_PX + 2 * PADDING_PX + HEADER_PX + FOOTER_PX,
)


class GymCheckersEnv(gym.Env):
    """A Gymnasium environment for two-player Checkers (Draughts) under standard rules."""

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
            self._title_font = ImageFont.truetype(font_file, 16)
            self._stats_font = ImageFont.truetype(font_file, 11)
        except Exception:
            logging.warning("Could not load system sans-serif font. Using default font.")
            self._title_font = ImageFont.load_default()
            self._stats_font = ImageFont.load_default()

        # Spaces
        # Action space: [from_row, from_col, to_row, to_col]
        self.action_space = spaces.MultiDiscrete([8, 8, 8, 8])
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0, high=4, shape=(8, 8), dtype=np.int32
                ),
                "valid_mask": spaces.Box(
                    low=0, high=1, shape=(8, 8, 8, 8), dtype=np.int8
                ),
                "current_player": spaces.Discrete(3),  # 1 or 2
            }
        )

        self._background = np.full(
            (CANVAS_SIZE[1], CANVAS_SIZE[0], 3), COLOR_BG, dtype=np.uint8
        )
        self._background[0:HEADER_PX, :] = COLOR_HEADER
        self._background[CANVAS_SIZE[1] - FOOTER_PX:, :] = COLOR_FOOTER

        self.reset()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to the starting state."""
        super().reset(seed=seed)

        if options is not None and "state" in options:
            state = options["state"]
            self._grid = state["grid"].copy()
            self._current_player = state["current_player"]
            self._active_jumper = state["active_jumper"]
            self._total_moves = state["total_moves"]
            self._draw_counter = state["draw_counter"]
            self._move_history = copy.deepcopy(state["move_history"])

            return self._create_observation(), {}

        # Reset board
        self._grid = np.zeros((8, 8), dtype=np.int32)
        
        # Populate Player 2 pieces (White/Magenta) at top (rows 0..2, only on dark squares)
        # Dark squares are those where (r + c) % 2 == 1
        for r in range(3):
            for c in range(8):
                if (r + c) % 2 == 1:
                    self._grid[r, c] = P2_NORMAL

        # Populate Player 1 pieces (Cyan) at bottom (rows 5..7, only on dark squares)
        for r in range(5, 8):
            for c in range(8):
                if (r + c) % 2 == 1:
                    self._grid[r, c] = P1_NORMAL

        self._current_player = 1
        self._active_jumper: Optional[Tuple[int, int]] = None
        self._total_moves = 0
        self._draw_counter = 0
        self._move_history: List[Tuple[Tuple[int, int, int, int], bool]] = []

        return self._create_observation(), {}

    def _get_pieces_count(self) -> Tuple[int, int]:
        """Returns the number of pieces remaining for (Player 1, Player 2)."""
        p1 = int(np.sum((self._grid == P1_NORMAL) | (self._grid == P1_KING)))
        p2 = int(np.sum((self._grid == P2_NORMAL) | (self._grid == P2_KING)))
        return p1, p2

    def _get_jumps_for_piece(self, r: int, c: int) -> List[Tuple[int, int, int, int]]:
        """Returns list of valid jump moves for a specific piece at (r, c)."""
        piece = self._grid[r, c]
        if piece == EMPTY:
            return []

        jumps = []
        is_p1 = piece in (P1_NORMAL, P1_KING)
        is_king = piece in (P1_KING, P2_KING)
        
        # Directions: List of (dr, dc)
        # Normal pieces move/jump forward. King moves/jumps any direction.
        dirs = []
        if is_king:
            dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        elif is_p1:
            dirs = [(-1, -1), (-1, 1)]  # P1 moves up (decreasing row)
        else:
            dirs = [ (1, -1),  (1, 1)]  # P2 moves down (increasing row)

        for dr, dc in dirs:
            mid_r, mid_c = r + dr, c + dc
            end_r, end_c = r + 2 * dr, c + 2 * dc

            if 0 <= end_r < 8 and 0 <= end_c < 8:
                mid_piece = self._grid[mid_r, mid_c]
                end_piece = self._grid[end_r, end_c]

                if end_piece == EMPTY and mid_piece != EMPTY:
                    # Check if mid_piece belongs to opponent
                    mid_is_p1 = mid_piece in (P1_NORMAL, P1_KING)
                    if is_p1 != mid_is_p1:
                        jumps.append((r, c, end_r, end_c))

        return jumps

    def _get_normal_moves_for_piece(self, r: int, c: int) -> List[Tuple[int, int, int, int]]:
        """Returns list of regular single-step moves for a specific piece at (r, c)."""
        piece = self._grid[r, c]
        if piece == EMPTY:
            return []

        moves = []
        is_p1 = piece in (P1_NORMAL, P1_KING)
        is_king = piece in (P1_KING, P2_KING)

        dirs = []
        if is_king:
            dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        elif is_p1:
            dirs = [(-1, -1), (-1, 1)]
        else:
            dirs = [(1, -1), (1, 1)]

        for dr, dc in dirs:
            end_r, end_c = r + dr, c + dc
            if 0 <= end_r < 8 and 0 <= end_c < 8:
                if self._grid[end_r, end_c] == EMPTY:
                    moves.append((r, c, end_r, end_c))

        return moves

    def _get_valid_moves(self, player: int) -> List[Tuple[int, int, int, int]]:
        """Returns list of all valid moves for the specified player."""
        # 1. Multi-jump restriction
        if self._active_jumper is not None:
            jr, jc = self._active_jumper
            return self._get_jumps_for_piece(jr, jc)

        # Gather all pieces belonging to current player
        p1_turn = player == 1
        pieces_coords = []
        for r in range(8):
            for c in range(8):
                piece = self._grid[r, c]
                if piece != EMPTY:
                    piece_is_p1 = piece in (P1_NORMAL, P1_KING)
                    if p1_turn == piece_is_p1:
                        pieces_coords.append((r, c))

        # 2. Check for mandatory jump captures
        all_jumps = []
        for r, c in pieces_coords:
            all_jumps.extend(self._get_jumps_for_piece(r, c))

        if len(all_jumps) > 0:
            return all_jumps  # Only jumps are valid if at least one is available!

        # 3. Otherwise, normal moves
        all_moves = []
        for r, c in pieces_coords:
            all_moves.extend(self._get_normal_moves_for_piece(r, c))

        return all_moves

    def _create_observation(self) -> Dict[str, Any]:
        """Create observation dictionary and valid actions mask."""
        valid_mask = np.zeros((8, 8, 8, 8), dtype=np.int8)
        
        valid_moves = self._get_valid_moves(self._current_player)
        for fr, fc, tr, tc in valid_moves:
            valid_mask[fr, fc, tr, tc] = 1

        return {
            "observation": self._grid.copy(),
            "valid_mask": valid_mask,
            "current_player": self._current_player,
        }

    def step(
        self, action: npt.NDArray[np.int32]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Perform one move step in checkers."""
        fr, fc, tr, tc = action
        if not (0 <= fr < 8 and 0 <= fc < 8 and 0 <= tr < 8 and 0 <= tc < 8):
            raise ValueError(f"Action out of bounds: {action}")

        # Check validity
        valid_moves = self._get_valid_moves(self._current_player)
        is_valid = (fr, fc, tr, tc) in valid_moves

        reward = -0.01  # Small step penalty to encourage speed
        self._total_moves += 1

        if not is_valid:
            # Invalid move penalty and action is ignored
            self._draw_counter += 1
            reward -= 0.1
            self._move_history.append(((fr, fc, tr, tc), False))
            if len(self._move_history) > 6:
                self._move_history.pop(0)

            # Check if draw due to stalemate / infinite invalid loops
            truncated = self._draw_counter >= 100
            terminated = False
            return self._create_observation(), float(reward), terminated, truncated, {"state": self._get_state()}

        # Perform move
        self._draw_counter = 0
        piece = self._grid[fr, fc]
        is_jump = abs(tr - fr) == 2

        # Move piece on grid
        self._grid[tr, tc] = piece
        self._grid[fr, fc] = EMPTY

        # If jump, remove captured piece
        if is_jump:
            mid_r = (fr + tr) // 2
            mid_c = (fc + tc) // 2
            self._grid[mid_r, mid_c] = EMPTY
            
            # Zero-sum capture rewards
            if self._current_player == 1:
                reward += 1.0
            else:
                reward -= 1.0

        # Handle King Promotion
        promoted = False
        if self._current_player == 1 and tr == 0 and piece == P1_NORMAL:
            self._grid[tr, tc] = P1_KING
            promoted = True
            reward += 0.5
        elif self._current_player == 2 and tr == 7 and piece == P2_NORMAL:
            self._grid[tr, tc] = P2_KING
            promoted = True
            reward -= 0.5

        # Check for multi-jump
        # Only valid if we just jumped, didn't promote on this step, and that piece has more jumps
        has_more_jumps = False
        if is_jump and not promoted:
            more_jumps = self._get_jumps_for_piece(tr, tc)
            if len(more_jumps) > 0:
                has_more_jumps = True
                self._active_jumper = (tr, tc)

        # Update move history
        self._move_history.append(((fr, fc, tr, tc), True))
        if len(self._move_history) > 6:
            self._move_history.pop(0)

        # Switch turns if no multi-jumps are available
        if not has_more_jumps:
            self._active_jumper = None
            self._current_player = 2 if self._current_player == 1 else 1

        # Check win/loss/draw conditions
        p1_cnt, p2_cnt = self._get_pieces_count()
        terminated = False
        
        if p1_cnt == 0:
            terminated = True
            reward = -10.0  # Player 2 wins
        elif p2_cnt == 0:
            terminated = True
            reward = 10.0   # Player 1 wins
        else:
            # Check if current player is blocked
            curr_valid_moves = self._get_valid_moves(self._current_player)
            if len(curr_valid_moves) == 0:
                terminated = True
                # If current player is blocked, they lose.
                reward = -10.0 if self._current_player == 1 else 10.0

        truncated = False
        observation = self._create_observation()
        return observation, float(reward), terminated, truncated, {"state": self._get_state()}

    def _get_state(self) -> Dict[str, Any]:
        """Return full internal state of the environment."""
        return {
            "grid": self._grid.copy(),
            "current_player": self._current_player,
            "active_jumper": self._active_jumper,
            "total_moves": self._total_moves,
            "draw_counter": self._draw_counter,
            "move_history": copy.deepcopy(self._move_history),
        }

    def _draw_symbol(
        self, draw: ImageDraw.ImageDraw, x: int, y: int, action: Tuple[int, int, int, int], is_valid: bool
    ) -> None:
        """Draw a text representation of a move in the footer history."""
        fr, fc, tr, tc = action
        text = f"({fr},{fc})→({tr},{tc})"
        color = COLOR_P1 if is_valid else (231, 76, 60)
        draw.text((x, y), text, fill=color, font=self._stats_font, anchor="lm")

    def _render(self) -> None:
        """Draw visuals representing the current board state."""
        canvas = Image.fromarray(self._background)
        draw = ImageDraw.Draw(canvas)

        # Draw Header
        draw.text(
            (PADDING_PX + 5, HEADER_PX // 2),
            "CHECKERS",
            fill=COLOR_TEXT,
            font=self._title_font,
            anchor="lm",
        )

        p1_cnt, p2_cnt = self._get_pieces_count()
        draw.text(
            (CANVAS_SIZE[0] - PADDING_PX - 5, HEADER_PX // 2),
            f"P1: {p1_cnt} | P2: {p2_cnt} | Turn: P{self._current_player}",
            fill=COLOR_TEXT,
            font=self._title_font,
            anchor="rm",
        )

        # Draw board cells
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                rx = PADDING_PX + c * CELL_PX
                ry = HEADER_PX + PADDING_PX + r * CELL_PX
                
                # Check square color
                is_dark = (r + c) % 2 == 1
                bg_color = COLOR_DARK_SQUARE if is_dark else COLOR_LIGHT_SQUARE
                
                # Draw square cell
                draw.rectangle(
                    [rx, ry, rx + CELL_PX - 1, ry + CELL_PX - 1],
                    fill=bg_color,
                    outline=COLOR_GRID,
                    width=1,
                )

                # Draw piece if present
                piece = self._grid[r, c]
                if piece != EMPTY:
                    is_p1 = piece in (P1_NORMAL, P1_KING)
                    is_king = piece in (P1_KING, P2_KING)
                    color = COLOR_P1 if is_p1 else COLOR_P2
                    
                    # Draw circle disc
                    offset = 4
                    draw.ellipse(
                        [
                            rx + offset,
                            ry + offset,
                            rx + CELL_PX - offset - 1,
                            ry + CELL_PX - offset - 1,
                        ],
                        fill=color,
                    )

                    # Highlight King piece
                    if is_king:
                        # Draw crown shape or star dot
                        draw.ellipse(
                            [
                                rx + CELL_PX // 2 - 4,
                                ry + CELL_PX // 2 - 4,
                                rx + CELL_PX // 2 + 4,
                                ry + CELL_PX // 2 + 4,
                            ],
                            fill=COLOR_CROWN,
                        )

        # Draw Footer Statistics
        stats_text = f"Total Moves: {self._total_moves}"
        draw.text(
            (PADDING_PX + 5, CANVAS_SIZE[1] - FOOTER_PX // 2),
            stats_text,
            fill=COLOR_TEXT,
            font=self._stats_font,
            anchor="lm",
        )

        # Draw move history (displays last 3 moves with sequence arrows in footer)
        arrow_y = CANVAS_SIZE[1] - FOOTER_PX // 2
        visible_history = self._move_history[-3:]
        if visible_history:
            x_pos = CANVAS_SIZE[0] - 220
            draw.text((x_pos, arrow_y), "Moves: ", fill=(150, 150, 150), font=self._stats_font, anchor="lm")
            x_pos += 40
            for idx, (action, is_valid) in enumerate(visible_history):
                self._draw_symbol(draw, x_pos, arrow_y, action, is_valid)
                x_pos += 50
                if idx < len(visible_history) - 1:
                    draw.text((x_pos, arrow_y), "→", fill=(100, 100, 100), font=self._stats_font, anchor="lm")
                    x_pos += 12

        self._current_observation = np.array(canvas, dtype=np.uint8)

    def render(self) -> Optional[npt.NDArray[np.uint8]]:
        """Return visually drawn board frame."""
        self._render()
        return self._current_observation.copy()

    def close(self) -> None:
        """Close the environment."""
        pass

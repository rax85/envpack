"""A Gymnasium environment for two-player Battleship."""

import copy
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from absl import logging
from gymnasium import spaces
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import numpy.typing as npt

# Canvas Constants
WIDTH = 400
HEIGHT = 300
HEADER_PX = 40
FOOTER_PX = 20

# Colors
COLOR_BG = (15, 23, 42)           # Dark slate blue
COLOR_HEADER = (30, 41, 59)       # Header slate blue
COLOR_FOOTER = (15, 23, 42)
COLOR_TEXT = (248, 250, 252)
COLOR_GRID_LINE = (51, 65, 85)
COLOR_WATER = (30, 58, 138)       # Deep blue
COLOR_SHIP = (100, 116, 139)       # Grey steel
COLOR_HIT = (239, 68, 68)         # Vibrant red
COLOR_MISS = (148, 163, 184)       # Light grey/slate
COLOR_HIGHLIGHT = (56, 189, 248)  # Cyan highlights

SHIP_SIZES = [5, 4, 3, 3, 2]      # Carrier, Battleship, Destroyer, Submarine, Patrol Boat


class GymBattleshipEnv(gym.Env):
    """A Gymnasium environment for two-player turn-based Battleship."""

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
        # Action space: [row, col] to shoot on opponent's board
        self.action_space = spaces.MultiDiscrete([8, 8])

        # Observation space (perspective-based for current player)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0, high=5, shape=(8, 8), dtype=np.int32
                ),
                "valid_mask": spaces.Box(
                    low=0, high=1, shape=(8, 8), dtype=np.int8
                ),
                "current_player": spaces.Discrete(3),  # 1 or 2
                "ships_left": spaces.Box(
                    low=0, high=17, shape=(2,), dtype=np.int32
                ),
            }
        )

        # Base background canvas
        self._background = np.full((HEIGHT, WIDTH, 3), COLOR_BG, dtype=np.uint8)
        self._background[0:HEADER_PX, :] = COLOR_HEADER
        self._background[HEIGHT - FOOTER_PX :, :] = COLOR_FOOTER

        self.reset()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to the initial state."""
        super().reset(seed=seed)

        if options is not None and "state" in options:
            state = options["state"]
            self._p1_board = state["p1_board"].copy()
            self._p2_board = state["p2_board"].copy()
            self._p1_shots = state["p1_shots"].copy()
            self._p2_shots = state["p2_shots"].copy()
            self._current_player = state["current_player"]
            self._total_moves = state["total_moves"]
            self._draw_counter = state["draw_counter"]
            return self._create_observation(), {}

        # Reset boards: 0 = empty, 1 = ship
        self._p1_board = np.zeros((8, 8), dtype=np.int32)
        self._p2_board = np.zeros((8, 8), dtype=np.int32)

        # Reset shots: 0 = none, 1 = hit, 2 = miss
        self._p1_shots = np.zeros((8, 8), dtype=np.int32)
        self._p2_shots = np.zeros((8, 8), dtype=np.int32)

        # Randomly place ships for both players
        self._place_ships_randomly(self._p1_board)
        self._place_ships_randomly(self._p2_board)

        self._current_player = 1
        self._total_moves = 0
        self._draw_counter = 0

        return self._create_observation(), {}

    def _place_ships_randomly(self, board: npt.NDArray[np.int32]) -> None:
        """Place all ships randomly without overlapping or extending out of bounds."""
        for size in SHIP_SIZES:
            placed = False
            while not placed:
                orientation = self.np_random.choice(["H", "V"])
                if orientation == "H":
                    r = self.np_random.integers(0, 8)
                    c = self.np_random.integers(0, 8 - size + 1)
                    # Check overlap
                    if np.sum(board[r, c : c + size]) == 0:
                        board[r, c : c + size] = 1
                        placed = True
                else:
                    r = self.np_random.integers(0, 8 - size + 1)
                    c = self.np_random.integers(0, 8)
                    # Check overlap
                    if np.sum(board[r : r + size, c]) == 0:
                        board[r : r + size, c] = 1
                        placed = True

    def _get_ships_remaining(self) -> Tuple[int, int]:
        """Returns the number of remaining ship cells for Player 1 and Player 2."""
        # Remaining cells = Total ship cells (17) - Hits on those ships
        p1_hits = np.sum((self._p1_board == 1) & (self._p2_shots == 1))
        p2_hits = np.sum((self._p2_board == 1) & (self._p1_shots == 1))
        return int(17 - p1_hits), int(17 - p2_hits)

    def _create_observation(self) -> Dict[str, Any]:
        """Create the perspective observation dict for the current active player."""
        p1_rem, p2_rem = self._get_ships_remaining()

        # valid_mask: where current player hasn't shot yet
        active_shots = self._p1_shots if self._current_player == 1 else self._p2_shots
        valid_mask = np.where(active_shots == 0, 1, 0).astype(np.int8)

        # Observation grid representing current player's perspective:
        # Cell values:
        # 0: Empty / Unknown (opponent's side, or empty on own side)
        # 1: Own Ship (undamaged)
        # 2: Own Ship Hit (damaged by opponent)
        # 3: Own Empty Hit (miss by opponent)
        # 4: Opponent Ship Hit (hit by current player)
        # 5: Opponent Empty Hit (miss by current player)
        obs_grid = np.zeros((8, 8), dtype=np.int32)

        if self._current_player == 1:
            # Own board (Player 1)
            for r in range(8):
                for c in range(8):
                    is_ship = self._p1_board[r, c] == 1
                    shot_type = self._p2_shots[r, c] # opponent's shots
                    if is_ship:
                        obs_grid[r, c] = 2 if shot_type == 1 else 1
                    else:
                        obs_grid[r, c] = 3 if shot_type == 2 else 0

            # Opponent board (Player 2) -> overlay onto cells that have been shot by Player 1
            for r in range(8):
                for c in range(8):
                    my_shot = self._p1_shots[r, c]
                    if my_shot == 1:   # Hit
                        obs_grid[r, c] = 4
                    elif my_shot == 2: # Miss
                        obs_grid[r, c] = 5
        else:
            # Own board (Player 2)
            for r in range(8):
                for c in range(8):
                    is_ship = self._p2_board[r, c] == 1
                    shot_type = self._p1_shots[r, c] # opponent's shots
                    if is_ship:
                        obs_grid[r, c] = 2 if shot_type == 1 else 1
                    else:
                        obs_grid[r, c] = 3 if shot_type == 2 else 0

            # Opponent board (Player 1) -> overlay onto cells that have been shot by Player 2
            for r in range(8):
                for c in range(8):
                    my_shot = self._p2_shots[r, c]
                    if my_shot == 1:   # Hit
                        obs_grid[r, c] = 4
                    elif my_shot == 2: # Miss
                        obs_grid[r, c] = 5

        return {
            "observation": obs_grid,
            "valid_mask": valid_mask,
            "current_player": self._current_player,
            "ships_left": np.array([p1_rem, p2_rem], dtype=np.int32),
        }

    def step(
        self, action: npt.NDArray[np.int32]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Process one shot step in turn-based Battleship."""
        r, c = action[0], action[1]
        if not (0 <= r < 8 and 0 <= c < 8):
            raise ValueError(f"Action coordinates out of bounds: {action}")

        # Current player signs for zero-sum perspective (Player 1 = +1, Player 2 = -1)
        player_sign = 1.0 if self._current_player == 1 else -1.0

        # Check if already shot
        active_shots = self._p1_shots if self._current_player == 1 else self._p2_shots
        opponent_board = self._p2_board if self._current_player == 1 else self._p1_board

        is_valid = active_shots[r, c] == 0
        reward = 0.0

        self._total_moves += 1

        if not is_valid:
            # Penalty for invalid action (shooting already targeted cell)
            self._draw_counter += 1
            reward -= 0.5 * player_sign
            truncated = self._draw_counter >= 100
            return self._create_observation(), float(reward), False, truncated, {"state": self._get_state()}

        self._draw_counter = 0

        # Fire shot
        is_hit = opponent_board[r, c] == 1
        if is_hit:
            active_shots[r, c] = 1 # Mark hit
            reward += 1.0 * player_sign
        else:
            active_shots[r, c] = 2 # Mark miss
            reward -= 0.1 * player_sign  # Small miss penalty to encourage accuracy

        # Check Win Conditions
        p1_rem, p2_rem = self._get_ships_remaining()
        terminated = False
        if p2_rem <= 0:   # Player 1 won
            reward += 10.0
            terminated = True
        elif p1_rem <= 0: # Player 2 won
            reward -= 10.0
            terminated = True

        # Switch players if game not terminated
        if not terminated:
            self._current_player = 2 if self._current_player == 1 else 1

        return self._create_observation(), float(reward), terminated, False, {"state": self._get_state()}

    def _get_state(self) -> Dict[str, Any]:
        """Return full state dictionary."""
        return {
            "p1_board": self._p1_board.copy(),
            "p2_board": self._p2_board.copy(),
            "p1_shots": self._p1_shots.copy(),
            "p2_shots": self._p2_shots.copy(),
            "current_player": self._current_player,
            "total_moves": self._total_moves,
            "draw_counter": self._draw_counter,
        }

    def render(self) -> np.ndarray:
        """Render the Battleship game boards side by side."""
        image = Image.fromarray(self._background.copy())
        draw = ImageDraw.Draw(image)

        # Header Text
        draw.text((10, 10), "BATTLESHIP", fill=COLOR_TEXT, font=self._title_font)
        p1_rem, p2_rem = self._get_ships_remaining()
        draw.text(
            (160, 12),
            f"Turn: P{self._current_player}   P1 Ships: {p1_rem}   P2 Ships: {p2_rem}",
            fill=COLOR_TEXT,
            font=self._stats_font,
        )

        # Draw left board (P1 board showing P2's shots)
        draw.text((50, 48), "Player 1 Board", fill=COLOR_HIGHLIGHT, font=self._stats_font)
        self._draw_grid(draw, 36, 70, self._p1_board, self._p2_shots, is_targeting=False)

        # Draw right board (P2 board showing P1's shots)
        draw.text((250, 48), "Player 2 Board", fill=COLOR_HIGHLIGHT, font=self._stats_font)
        self._draw_grid(draw, 236, 70, self._p2_board, self._p1_shots, is_targeting=False)

        # Draw footer info
        draw.text((10, HEIGHT - 15), "TARGET AND SHOOT COORDINATES: [ROW, COL]", fill=COLOR_TEXT, font=self._stats_font)

        return np.array(image, dtype=np.uint8)

    def _draw_grid(
        self,
        draw: ImageDraw.Draw,
        start_x: int,
        start_y: int,
        board: npt.NDArray[np.int32],
        shots: npt.NDArray[np.int32],
        is_targeting: bool = False,
    ) -> None:
        """Draw an 8x8 battleship board grid."""
        cell_size = 16
        # Draw cells
        for r in range(8):
            for c in range(8):
                cx = start_x + c * cell_size
                cy = start_y + r * cell_size

                is_ship = board[r, c] == 1
                shot = shots[r, c]

                # Select fill color
                if shot == 1:     # Hit
                    color = COLOR_HIT
                elif shot == 2:   # Miss
                    color = COLOR_MISS
                elif is_ship and not is_targeting:
                    color = COLOR_SHIP
                else:
                    color = COLOR_WATER

                draw.rectangle(
                    [cx, cy, cx + cell_size - 1, cy + cell_size - 1],
                    fill=color,
                    outline=COLOR_GRID_LINE,
                )

        # Draw row and col index markers
        for i in range(8):
            # Col index (0..7) on top of grid
            draw.text(
                (start_x + i * cell_size + 5, start_y - 12),
                str(i),
                fill=COLOR_TEXT,
                font=self._stats_font,
            )
            # Row index (0..7) on left of grid
            draw.text(
                (start_x - 12, start_y + i * cell_size + 2),
                str(i),
                fill=COLOR_TEXT,
                font=self._stats_font,
            )

    def close(self) -> None:
        """Clean up resources."""
        pass

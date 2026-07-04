"""A Gym environment for playing the game Snake."""

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
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

OPPOSITES = {
    UP: DOWN,
    DOWN: UP,
    LEFT: RIGHT,
    RIGHT: LEFT,
}

# Grid Constants
GRID_SIZE = (10, 10)
CELL_PX = 32
PADDING_PX = 2
HEADER_PX = 60
FOOTER_PX = 40

# Colors
COLOR_BG = (30, 30, 30)
COLOR_GRID = (45, 45, 45)
COLOR_HEAD = (46, 204, 113)  # Bright Green
COLOR_BODY = (39, 174, 96)   # Darker Green
COLOR_FOOD = (231, 76, 60)   # Vibrant Red
COLOR_HEADER = (52, 73, 94)  # Slate Blue
COLOR_FOOTER = (44, 62, 80)   # Darker Slate Blue
COLOR_TEXT_LIGHT = (236, 240, 241)

CANVAS_SIZE = (
    GRID_SIZE[0] * (CELL_PX + PADDING_PX) + PADDING_PX,
    GRID_SIZE[1] * (CELL_PX + PADDING_PX) + PADDING_PX + HEADER_PX + FOOTER_PX,
)


class GymSnakeEnv(gym.Env):
    """A Gym environment for playing the game Snake."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.grid_size = GRID_SIZE

        # Font setup
        try:
            font_properties = font_manager.FontProperties(
                family="sans-serif", weight="bold"
            )
            font_file = font_manager.findfont(font_properties)
            self._score_font = ImageFont.truetype(font_file, 20)
            self._stats_font = ImageFont.truetype(font_file, 12)
        except Exception:
            logging.warning("Could not load system sans-serif font. Using default font.")
            self._score_font = ImageFont.load_default()
            self._stats_font = ImageFont.load_default()

        # Spaces
        n_actions = 4
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0, high=3, shape=(GRID_SIZE[1], GRID_SIZE[0]), dtype=np.int32
                ),
                "valid_mask": spaces.Box(
                    low=0, high=1, shape=(n_actions,), dtype=np.int32
                ),
                "total_score": spaces.Box(
                    low=0, high=1000, shape=(1,), dtype=np.int32
                ),
            }
        )

        # Background base setup
        self._background = np.full(
            (CANVAS_SIZE[1], CANVAS_SIZE[0], 3), COLOR_BG, dtype=np.uint8
        )
        # Pre-draw static header/footer backgrounds
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
            self.snake = list(map(tuple, state["snake"]))
            self.direction = state["direction"]
            self.food = tuple(state["food"])
            self.score = state["score"]
            self.total_moves = state["total_moves"]
            self.steps_since_eating = state["steps_since_eating"]
            self.move_history = copy.deepcopy(state["move_history"])

            observation = self._create_observation()
            return observation, {}

        # Reset snake at the center of the grid moving RIGHT
        center_y = GRID_SIZE[1] // 2
        center_x = GRID_SIZE[0] // 2
        self.snake = [(center_y, center_x)]
        self.direction = RIGHT
        self.score = 0
        self.total_moves = 0
        self.steps_since_eating = 0
        self.move_history: List[Tuple[int, bool]] = []

        self._spawn_food()

        observation = self._create_observation()
        return observation, {}

    def _spawn_food(self) -> None:
        """Spawn a food tile at a random empty position."""
        snake_set = set(self.snake)
        candidates = [
            (y, x)
            for y in range(GRID_SIZE[1])
            for x in range(GRID_SIZE[0])
            if (y, x) not in snake_set
        ]
        if candidates:
            idx = self.np_random.integers(len(candidates))
            self.food = candidates[idx]
        else:
            self.food = (-1, -1)  # Grid full (win)

    def _get_valid_mask(self) -> npt.NDArray[np.int32]:
        """Compute the mask of valid moves (cannot fold backward into itself)."""
        mask = np.ones(4, dtype=np.int32)
        if len(self.snake) > 1:
            opposite_action = OPPOSITES[self.direction]
            mask[opposite_action] = 0
        return mask

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Perform one step in the environment."""
        if not (0 <= action <= 3):
            raise ValueError(f"Invalid action: {action}")

        self.total_moves += 1
        self.steps_since_eating += 1

        # Resolve action
        valid_mask = self._get_valid_mask()
        # If agent attempts an invalid move (e.g. going backward), ignore it and continue in current direction
        if valid_mask[action] == 0:
            action = self.direction

        self.direction = action

        # Calculate new head position
        head_y, head_x = self.snake[0]
        if action == UP:
            new_head = (head_y - 1, head_x)
        elif action == DOWN:
            new_head = (head_y + 1, head_x)
        elif action == LEFT:
            new_head = (head_y, head_x - 1)
        else:
            new_head = (head_y, head_x + 1)

        # Check for collision with boundaries or self (excluding tail segment since it will move)
        collided = (
            new_head[0] < 0
            or new_head[0] >= GRID_SIZE[1]
            or new_head[1] < 0
            or new_head[1] >= GRID_SIZE[0]
            or new_head in self.snake[:-1]
        )

        terminated = False
        truncated = False
        reward = -0.01  # Step penalty to encourage direct pathing

        if collided:
            terminated = True
            reward = -1.0
            self.move_history.append((action, False))
        else:
            # Move snake head forward
            self.snake.insert(0, new_head)

            # Check if food eaten
            if new_head == self.food:
                self.score += 1
                reward = 1.0
                self.steps_since_eating = 0
                self.move_history.append((action, True))
                self._spawn_food()
                if self.food == (-1, -1):
                    # No space left to spawn food (Perfect Game!)
                    terminated = True
                    reward += 5.0
            else:
                self.snake.pop()  # Remove tail since we didn't eat
                self.move_history.append((action, False))

        if len(self.move_history) > 8:
            self.move_history.pop(0)

        # Truncate if steps since eating exceeds a limit to prevent infinite loops
        if self.steps_since_eating >= 200:
            truncated = True

        observation = self._create_observation()
        return observation, float(reward), terminated, truncated, {"state": self._get_state()}

    def _get_state(self) -> Dict[str, Any]:
        """Return the current internal state of the environment."""
        return {
            "snake": copy.deepcopy(self.snake),
            "direction": self.direction,
            "food": self.food,
            "score": self.score,
            "total_moves": self.total_moves,
            "steps_since_eating": self.steps_since_eating,
            "move_history": copy.deepcopy(self.move_history),
        }

    def _create_observation(self) -> Dict[str, Any]:
        """Create the observation dictionary."""
        grid = np.zeros((GRID_SIZE[1], GRID_SIZE[0]), dtype=np.int32)
        # Populate grid
        # 0: Empty, 1: Food, 2: Head, 3: Body
        if self.food != (-1, -1):
            grid[self.food[0], self.food[1]] = 1
        for y, x in self.snake[1:]:
            grid[y, x] = 3
        if self.snake:
            grid[self.snake[0][0], self.snake[0][1]] = 2

        valid_mask = self._get_valid_mask()
        
        return {
            "observation": grid,
            "valid_mask": valid_mask,
            "total_score": np.array([self.score], dtype=np.int32),
        }

    def _draw_arrow(
        self, draw: ImageDraw.ImageDraw, x: int, y: int, action: int, color: Tuple[int, int, int]
    ) -> None:
        """Draw an arrow representing a move with a shaft and head."""
        radius = 6
        if action == UP:
            points = [
                (x, y - radius),
                (x + 4, y),
                (x + 1.5, y),
                (x + 1.5, y + radius),
                (x - 1.5, y + radius),
                (x - 1.5, y),
                (x - 4, y)
            ]
        elif action == DOWN:
            points = [
                (x, y + radius),
                (x + 4, y),
                (x + 1.5, y),
                (x + 1.5, y - radius),
                (x - 1.5, y - radius),
                (x - 1.5, y),
                (x - 4, y)
            ]
        elif action == LEFT:
            points = [
                (x - radius, y),
                (x, y - 4),
                (x, y - 1.5),
                (x + radius, y - 1.5),
                (x + radius, y + 1.5),
                (x, y + 1.5),
                (x, y + 4)
            ]
        elif action == RIGHT:
            points = [
                (x + radius, y),
                (x, y - 4),
                (x, y - 1.5),
                (x - radius, y - 1.5),
                (x - radius, y + 1.5),
                (x, y + 1.5),
                (x, y + 4)
            ]
        else:
            return
        draw.polygon(points, fill=color)

    def _render(self) -> None:
        """Update the current observation image."""
        # Start with static background
        canvas = Image.fromarray(self._background)
        draw = ImageDraw.Draw(canvas)

        # Draw Header Texts
        draw.text(
            (10, HEADER_PX // 2),
            "SNAKE",
            fill=COLOR_TEXT_LIGHT,
            font=self._score_font,
            anchor="lm",
        )

        draw.text(
            (CANVAS_SIZE[0] - 10, HEADER_PX // 2),
            f"SCORE: {self.score}",
            fill=COLOR_TEXT_LIGHT,
            font=self._score_font,
            anchor="rm",
        )

        # Draw Grid background cell squares
        for y in range(GRID_SIZE[1]):
            for x in range(GRID_SIZE[0]):
                rx = x * (CELL_PX + PADDING_PX) + PADDING_PX
                ry = y * (CELL_PX + PADDING_PX) + PADDING_PX + HEADER_PX
                draw.rectangle(
                    [rx, ry, rx + CELL_PX - 1, ry + CELL_PX - 1],
                    fill=COLOR_GRID,
                )

        # Draw Food
        if self.food != (-1, -1):
            fy, fx = self.food
            rx = fx * (CELL_PX + PADDING_PX) + PADDING_PX
            ry = fy * (CELL_PX + PADDING_PX) + PADDING_PX + HEADER_PX
            draw.rounded_rectangle(
                [rx + 2, ry + 2, rx + CELL_PX - 3, ry + CELL_PX - 3],
                radius=4,
                fill=COLOR_FOOD,
            )

        # Draw Snake Body
        for idx, (y, x) in enumerate(self.snake):
            rx = x * (CELL_PX + PADDING_PX) + PADDING_PX
            ry = y * (CELL_PX + PADDING_PX) + PADDING_PX + HEADER_PX
            color = COLOR_HEAD if idx == 0 else COLOR_BODY
            
            # Slightly draw smaller head/body segments for rounded aesthetic
            margin = 1 if idx == 0 else 2
            draw.rounded_rectangle(
                [rx + margin, ry + margin, rx + CELL_PX - 1 - margin, ry + CELL_PX - 1 - margin],
                radius=4,
                fill=color,
            )

        # Draw Footer Statistics
        stats_text = f"Moves: {self.total_moves}  Eaten: {self.score}"
        draw.text(
            (10, CANVAS_SIZE[1] - FOOTER_PX + 12),
            stats_text,
            fill=COLOR_TEXT_LIGHT,
            font=self._stats_font,
        )

        # Draw move history arrows
        arrow_y = CANVAS_SIZE[1] - FOOTER_PX // 2
        arrow_x_start = CANVAS_SIZE[0] - 135
        arrow_spacing = 16
        for i, (action, is_eaten) in enumerate(self.move_history):
            color = COLOR_HEAD if is_eaten else COLOR_TEXT_LIGHT
            self._draw_arrow(draw, arrow_x_start + i * arrow_spacing, arrow_y, action, color)

        self._current_observation = np.array(canvas)

    def render(self) -> Optional[npt.NDArray[np.uint8]]:
        """Return the current observation as an RGB array."""
        self._render()
        return self._current_observation.copy()

    def close(self) -> None:
        """Close the environment."""
        pass

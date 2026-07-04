"""A Gymnasium environment for a Raptor-inspired vertical scrolling shooter."""

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
STAY = 0
LEFT = 1
RIGHT = 2
UP = 3
DOWN = 4

# Grid Constants
GRID_ROWS = 20
GRID_COLS = 15
CELL_PX = 24
PADDING_PX = 8
HEADER_PX = 60
FOOTER_PX = 40

# Grid Cell Types
EMPTY = 0
PLAYER = 1
LASER = 2
ENEMY_BASIC = 3
ENEMY_SHOOTER = 4
ENEMY_BULLET = 5
COIN = 6

# Colors
COLOR_BG = (15, 15, 25) # Deep space blue-black
COLOR_GRID = (30, 30, 45)
COLOR_PLAYER = (52, 152, 219)       # Metallic blue
COLOR_LASER = (46, 204, 113)        # Green laser
COLOR_ENEMY_BASIC = (231, 76, 60)   # Red basic enemy
COLOR_ENEMY_SHOOTER = (155, 89, 182) # Purple shooter enemy
COLOR_BULLET = (241, 196, 15)       # Yellow plasma bullet
COLOR_COIN = (243, 156, 18)         # Gold coin
COLOR_HEADER = (44, 62, 80)
COLOR_FOOTER = (52, 73, 94)
COLOR_TEXT = (236, 240, 241)

CANVAS_SIZE = (
    GRID_COLS * CELL_PX + 2 * PADDING_PX,
    GRID_ROWS * CELL_PX + 2 * PADDING_PX + HEADER_PX + FOOTER_PX,
)


class GymRaptorEnv(gym.Env):
    """A Gymnasium environment for a classic vertical scrolling shooter."""

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
        # Actions: 0: Stay, 1: Left, 2: Right, 3: Up, 4: Down
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0, high=6, shape=(GRID_ROWS, GRID_COLS), dtype=np.int32
                ),
                "valid_mask": spaces.Box(
                    low=0, high=1, shape=(5,), dtype=np.int8
                ),
                "total_score": spaces.Box(
                    low=0, high=100000, shape=(1,), dtype=np.int32
                ),
                "shield": spaces.Box(
                    low=0, high=100, shape=(1,), dtype=np.int32
                ),
            }
        )

        # Background stars for parallax scrolling
        self._background = np.full(
            (CANVAS_SIZE[1], CANVAS_SIZE[0], 3), COLOR_BG, dtype=np.uint8
        )
        self._background[0:HEADER_PX, :] = COLOR_HEADER
        self._background[CANVAS_SIZE[1] - FOOTER_PX:, :] = COLOR_FOOTER

        self.reset()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment state."""
        super().reset(seed=seed)

        if options is not None and "state" in options:
            state = options["state"]
            self._player_pos = state["player_pos"]
            self._shield = state["shield"]
            self._score = state["score"]
            self._money = state["money"]
            self._step_count = state["step_count"]
            self._enemies = copy.deepcopy(state["enemies"])
            self._lasers = copy.deepcopy(state["lasers"])
            self._bullets = copy.deepcopy(state["bullets"])
            self._coins = copy.deepcopy(state["coins"])
            self._stars = copy.deepcopy(state["stars"])
            self._move_history = copy.deepcopy(state["move_history"])

            return self._create_observation(), {}

        # Reset player to bottom middle
        self._player_pos = [GRID_ROWS - 2, GRID_COLS // 2]
        self._shield = 100
        self._score = 0
        self._money = 0
        self._step_count = 0

        # Entities lists containing coords [r, c]
        self._enemies: List[Tuple[List[int], int]] = []  # [ [r, c], type ]
        self._lasers: List[List[int]] = []              # [ [r, c] ]
        self._bullets: List[List[int]] = []             # [ [r, c] ]
        self._coins: List[List[int]] = []               # [ [r, c] ]

        # Star field: list of [x, y, speed]
        self._stars: List[List[float]] = []
        for _ in range(30):
            self._stars.append([
                self.np_random.uniform(PADDING_PX, CANVAS_SIZE[0] - PADDING_PX),
                self.np_random.uniform(HEADER_PX + PADDING_PX, CANVAS_SIZE[1] - FOOTER_PX - PADDING_PX),
                self.np_random.uniform(1.0, 3.0)
            ])

        self._move_history: List[Tuple[int, bool]] = []

        return self._create_observation(), {}

    def _create_observation(self) -> Dict[str, Any]:
        """Build grid observation and valid mask."""
        grid = np.full((GRID_ROWS, GRID_COLS), EMPTY, dtype=np.int32)

        # Place entities on grid
        for (r, c), etype in self._enemies:
            if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
                grid[r, c] = etype

        for r, c in self._lasers:
            if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
                grid[r, c] = LASER

        for r, c in self._bullets:
            if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
                grid[r, c] = ENEMY_BULLET

        for r, c in self._coins:
            if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
                grid[r, c] = COIN

        # Place player
        pr, pc = self._player_pos
        grid[pr, pc] = PLAYER

        # Valid mask
        # 0: Stay (always valid)
        # 1: Left (valid if pc > 0)
        # 2: Right (valid if pc < GRID_COLS - 1)
        # 3: Up (valid if pr > GRID_ROWS // 2)
        # 4: Down (valid if pr < GRID_ROWS - 1)
        mask = np.zeros(5, dtype=np.int8)
        mask[STAY] = 1
        if pc > 0:
            mask[LEFT] = 1
        if pc < GRID_COLS - 1:
            mask[RIGHT] = 1
        if pr > GRID_ROWS // 2:
            mask[UP] = 1
        if pr < GRID_ROWS - 1:
            mask[DOWN] = 1

        return {
            "observation": grid,
            "valid_mask": mask,
            "total_score": np.array([self._score], dtype=np.int32),
            "shield": np.array([self._shield], dtype=np.int32),
        }

    def _has_collision(self, pos1: List[int], pos2: List[int]) -> bool:
        """Returns True if two position lists overlap."""
        return pos1[0] == pos2[0] and pos1[1] == pos2[1]

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Advance game by one step."""
        if not (0 <= action < 5):
            raise ValueError(f"Invalid action: {action}")

        self._step_count += 1
        reward = 0.05  # Constant survival reward

        # Move player
        pr, pc = self._player_pos
        moved = False
        if action == LEFT and pc > 0:
            self._player_pos[1] -= 1
            moved = True
        elif action == RIGHT and pc < GRID_COLS - 1:
            self._player_pos[1] += 1
            moved = True
        elif action == UP and pr > GRID_ROWS // 2:
            self._player_pos[0] -= 1
            moved = True
        elif action == DOWN and pr < GRID_ROWS - 1:
            self._player_pos[0] += 1
            moved = True

        self._move_history.append((action, moved or action == STAY))
        if len(self._move_history) > 6:
            self._move_history.pop(0)

        # Automatic firing: Fire a laser every 3 steps
        if self._step_count % 3 == 0:
            self._lasers.append([self._player_pos[0] - 1, self._player_pos[1]])

        # 1. Update stars (visual movement)
        for star in self._stars:
            star[1] += star[2]
            if star[1] >= CANVAS_SIZE[1] - FOOTER_PX - PADDING_PX:
                star[1] = HEADER_PX + PADDING_PX
                star[0] = self.np_random.uniform(PADDING_PX, CANVAS_SIZE[0] - PADDING_PX)

        # 2. Move Player Lasers (move up 1 row)
        next_lasers = []
        for r, c in self._lasers:
            nr = r - 1
            if nr >= 0:
                next_lasers.append([nr, c])
        self._lasers = next_lasers

        # 3. Move Enemies (move down every 2 steps to make them slower than lasers)
        if self._step_count % 2 == 0:
            next_enemies = []
            for (r, c), etype in self._enemies:
                nr = r + 1
                if nr < GRID_ROWS:
                    next_enemies.append(([nr, c], etype))
                    # Shooter enemies shoot downwards occasionally
                    if etype == ENEMY_SHOOTER and self.np_random.random() < 0.2:
                        self._bullets.append([nr + 1, c])
            self._enemies = next_enemies

            # Move Enemy Bullets
            next_bullets = []
            for r, c in self._bullets:
                nr = r + 1
                if nr < GRID_ROWS:
                    next_bullets.append([nr, c])
            self._bullets = next_bullets

            # Move Coins
            next_coins = []
            for r, c in self._coins:
                nr = r + 1
                if nr < GRID_ROWS:
                    next_coins.append([nr, c])
            self._coins = next_coins

        # 4. Handle Spawning (spawns Basic Enemy, Shooter Enemy, or Coin at row 0)
        if self._step_count % 4 == 0:
            spawn_col = self.np_random.integers(0, GRID_COLS)
            spawn_roll = self.np_random.random()
            if spawn_roll < 0.4:
                # Basic enemy
                self._enemies.append(([0, spawn_col], ENEMY_BASIC))
            elif spawn_roll < 0.6:
                # Shooter enemy
                self._enemies.append(([0, spawn_col], ENEMY_SHOOTER))
            elif spawn_roll < 0.75:
                # Coin
                self._coins.append([0, spawn_col])

        # 5. Check Collisions: Laser vs Enemy
        rem_lasers = []
        for lr, lc in self._lasers:
            hit = False
            for idx, ((er, ec), etype) in enumerate(self._enemies):
                if er == lr and ec == lc:
                    hit = True
                    # Destroy enemy
                    self._enemies.pop(idx)
                    self._score += 100 if etype == ENEMY_BASIC else 250
                    self._money += 10 if etype == ENEMY_BASIC else 25
                    reward += 1.0 if etype == ENEMY_BASIC else 2.5
                    # Chance to drop coin
                    if self.np_random.random() < 0.5:
                        self._coins.append([er, ec])
                    break
            if not hit:
                rem_lasers.append([lr, lc])
        self._lasers = rem_lasers

        # 6. Check Collisions: Player vs Enemy Ship / Bullet / Coin
        # Enemy collision deals heavy damage
        for idx, ((er, ec), etype) in enumerate(self._enemies):
            if self._has_collision(self._player_pos, [er, ec]):
                self._enemies.pop(idx)
                self._shield = max(0, self._shield - 30)
                reward -= 5.0
                break

        # Bullet collision deals moderate damage
        for idx, (br, bc) in enumerate(self._bullets):
            if self._has_collision(self._player_pos, [br, bc]):
                self._bullets.pop(idx)
                self._shield = max(0, self._shield - 10)
                reward -= 1.5
                break

        # Coin collision grants money and points
        for idx, (cr, cc) in enumerate(self._coins):
            if self._has_collision(self._player_pos, [cr, cc]):
                self._coins.pop(idx)
                self._money += 50
                self._score += 500
                reward += 2.0
                break

        terminated = self._shield <= 0
        if terminated:
            reward -= 10.0  # Death penalty

        truncated = False
        return self._create_observation(), float(reward), terminated, truncated, {"state": self._get_state()}

    def _get_state(self) -> Dict[str, Any]:
        """Return full internal state of the environment."""
        return {
            "player_pos": list(self._player_pos),
            "shield": self._shield,
            "score": self._score,
            "money": self._money,
            "step_count": self._step_count,
            "enemies": copy.deepcopy(self._enemies),
            "lasers": copy.deepcopy(self._lasers),
            "bullets": copy.deepcopy(self._bullets),
            "coins": copy.deepcopy(self._coins),
            "stars": copy.deepcopy(self._stars),
            "move_history": copy.deepcopy(self._move_history),
        }

    def _render(self) -> None:
        """Create RGB visual grid visualization canvas."""
        canvas = Image.fromarray(self._background)
        draw = ImageDraw.Draw(canvas)

        # Draw Header
        draw.text(
            (PADDING_PX + 5, HEADER_PX // 2),
            "RAPTOR",
            fill=COLOR_TEXT,
            font=self._title_font,
            anchor="lm",
        )

        draw.text(
            (CANVAS_SIZE[0] - PADDING_PX - 5, HEADER_PX // 2),
            f"SCORE: {self._score}  CREDITS: ${self._money}",
            fill=COLOR_TEXT,
            font=self._title_font,
            anchor="rm",
        )

        # Draw Stars background
        for x, y, _ in self._stars:
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(120, 120, 150))

        # Draw Grid Cells
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                rx = PADDING_PX + c * CELL_PX
                ry = HEADER_PX + PADDING_PX + r * CELL_PX
                
                # Check what cell contains
                # We draw grid cells with outline
                draw.rectangle(
                    [rx, ry, rx + CELL_PX - 1, ry + CELL_PX - 1],
                    outline=COLOR_GRID,
                    width=1,
                )

        # Draw Entities on top of background
        # 1. Lasers
        for r, c in self._lasers:
            rx = PADDING_PX + c * CELL_PX + CELL_PX // 2
            ry = HEADER_PX + PADDING_PX + r * CELL_PX
            draw.line([(rx, ry), (rx, ry + CELL_PX - 1)], fill=COLOR_LASER, width=3)

        # 2. Bullets
        for r, c in self._bullets:
            rx = PADDING_PX + c * CELL_PX + CELL_PX // 2
            ry = HEADER_PX + PADDING_PX + r * CELL_PX + CELL_PX // 2
            draw.ellipse([rx - 4, ry - 4, rx + 4, ry + 4], fill=COLOR_BULLET)

        # 3. Coins
        for r, c in self._coins:
            rx = PADDING_PX + c * CELL_PX + CELL_PX // 2
            ry = HEADER_PX + PADDING_PX + r * CELL_PX + CELL_PX // 2
            draw.ellipse([rx - 5, ry - 5, rx + 5, ry + 5], fill=COLOR_COIN)

        # 4. Enemies
        for (r, c), etype in self._enemies:
            rx = PADDING_PX + c * CELL_PX
            ry = HEADER_PX + PADDING_PX + r * CELL_PX
            color = COLOR_ENEMY_BASIC if etype == ENEMY_BASIC else COLOR_ENEMY_SHOOTER
            # Draw triangle for enemy facing down
            draw.polygon(
                [
                    (rx + CELL_PX // 2, ry + CELL_PX - 2),
                    (rx + 2, ry + 2),
                    (rx + CELL_PX - 3, ry + 2),
                ],
                fill=color,
            )

        # 5. Player
        pr, pc = self._player_pos
        px = PADDING_PX + pc * CELL_PX
        py = HEADER_PX + PADDING_PX + pr * CELL_PX
        # Draw ship triangle facing up
        draw.polygon(
            [
                (px + CELL_PX // 2, py + 2),
                (px + 2, py + CELL_PX - 3),
                (px + CELL_PX - 3, py + CELL_PX - 3),
            ],
            fill=COLOR_PLAYER,
        )
        # Thruster fire
        draw.line(
            [(px + CELL_PX // 2, py + CELL_PX - 2), (px + CELL_PX // 2, py + CELL_PX + 2)],
            fill=(230, 126, 34),
            width=2,
        )

        # Draw Footer Statistics
        # Shield Health Bar
        shield_text = f"SHIELD: {self._shield}%"
        draw.text(
            (PADDING_PX + 5, CANVAS_SIZE[1] - FOOTER_PX // 2),
            shield_text,
            fill=COLOR_TEXT,
            font=self._stats_font,
            anchor="lm",
        )

        # Health bar box
        bar_x = PADDING_PX + 90
        bar_y = CANVAS_SIZE[1] - FOOTER_PX // 2 - 5
        bar_w = 80
        bar_h = 10
        draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], outline=(100, 100, 100), width=1)
        fill_w = int(bar_w * (self._shield / 100.0))
        if fill_w > 0:
            fill_color = (46, 204, 113) if self._shield > 40 else (231, 76, 60)
            draw.rectangle([bar_x + 1, bar_y + 1, bar_x + fill_w - 1, bar_y + bar_h - 1], fill=fill_color)

        # History
        hist_x = CANVAS_SIZE[0] - PADDING_PX - 120
        hist_y = CANVAS_SIZE[1] - FOOTER_PX // 2
        visible_hist = self._move_history[-4:]
        if visible_hist:
            draw.text((hist_x, hist_y), "History: ", fill=(150, 150, 150), font=self._stats_font, anchor="lm")
            hist_x += 45
            for action, success in visible_hist:
                arrow = "•"
                if action == LEFT:
                    arrow = "←"
                elif action == RIGHT:
                    arrow = "→"
                elif action == UP:
                    arrow = "↑"
                elif action == DOWN:
                    arrow = "↓"
                color = COLOR_TEXT if success else (231, 76, 60)
                draw.text((hist_x, hist_y), arrow, fill=color, font=self._stats_font, anchor="lm")
                hist_x += 12

        self._current_observation = np.array(canvas, dtype=np.uint8)

    def render(self) -> Optional[npt.NDArray[np.uint8]]:
        """Return visually drawn game frame."""
        self._render()
        return self._current_observation.copy()

    def close(self) -> None:
        """Close the environment."""
        pass

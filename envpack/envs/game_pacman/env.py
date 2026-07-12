"""A Gymnasium environment for Pacman."""

import copy
import math
import random
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

OPPOSITES = {
    UP: DOWN,
    DOWN: UP,
    LEFT: RIGHT,
    RIGHT: LEFT,
}

DIR_VECTORS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}

# 15x15 Maze Layout
# 1 represents wall, 0 represents path
MAZE_LAYOUT = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 0
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], # 1
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1], # 2
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1], # 3
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 4
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1], # 5
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1], # 6
    [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1], # 7
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1], # 8
    [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1], # 9
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 10
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1], # 11
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1], # 12
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 13
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 14
]

# Spawn settings
PACMAN_START = (13, 7)
BLINKY_START = (7, 7)
PINKY_START = (7, 6)
INKY_START = (7, 8)
CLYDE_START = (6, 7)

POWER_PELLETS_START = {(1, 1), (1, 13), (13, 1), (13, 13)}

WIDTH = 400
HEIGHT = 300


class GymPacmanEnv(gym.Env):
    """A Gymnasium environment for Pacman."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.SF = 3  # Scale factor for SSAA

        # Action Space: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = spaces.Discrete(4)

        # Observation Space: Dict with observation (300, 400, 3), valid_mask, score, lives
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0, high=255, shape=(300, 400, 3), dtype=np.uint8
                ),
                "valid_mask": spaces.Box(
                    low=0, high=1, shape=(4,), dtype=np.int8
                ),
                "score": spaces.Box(
                    low=0, high=100000, shape=(1,), dtype=np.int32
                ),
                "lives": spaces.Box(
                    low=0, high=3, shape=(1,), dtype=np.int32
                ),
            }
        )

        # Font setup (scaled by SF for high-quality downsampling)
        try:
            font_properties = font_manager.FontProperties(
                family="sans-serif", weight="bold"
            )
            font_file = font_manager.findfont(font_properties)
            self._hud_font = ImageFont.truetype(font_file, 14 * self.SF)
        except Exception:
            logging.warning("Could not load system sans-serif font. Using default font.")
            try:
                self._hud_font = ImageFont.load_default(size=14 * self.SF)
            except Exception:
                self._hud_font = ImageFont.load_default()

        self.reset()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to the initial state."""
        super().reset(seed=seed)

        if options is not None and "state" in options:
            state = options["state"]
            self.pacman_pos = tuple(state["pacman_pos"])
            self.pacman_dir = state["pacman_dir"]
            self.ghosts = copy.deepcopy(state["ghosts"])
            self.dots = set(map(tuple, state["dots"]))
            self.power_pellets = set(map(tuple, state["power_pellets"]))
            self.score = state["score"]
            self.lives = state["lives"]
            self.frightened_timer = state["frightened_timer"]
            self.steps = state.get("steps", 0)
        else:
            self.pacman_pos = PACMAN_START
            self.pacman_dir = RIGHT
            self.score = 0
            self.lives = 3
            self.frightened_timer = 0
            self.steps = 0
            self._reset_level_entities()

        obs = self._get_obs()
        return obs, {}

    def _reset_level_entities(self) -> None:
        """Reset Pacman and ghost positions and regenerate dots/pellets."""
        self.pacman_pos = PACMAN_START
        self.pacman_dir = RIGHT
        
        # Initialize ghosts
        self.ghosts = [
            {"name": "Blinky", "pos": BLINKY_START, "dir": UP, "start_pos": BLINKY_START},
            {"name": "Pinky", "pos": PINKY_START, "dir": UP, "start_pos": PINKY_START},
            {"name": "Inky", "pos": INKY_START, "dir": UP, "start_pos": INKY_START},
            {"name": "Clyde", "pos": CLYDE_START, "dir": UP, "start_pos": CLYDE_START},
        ]

        # Reset dots and power pellets
        self.dots = set()
        self.power_pellets = set(POWER_PELLETS_START)
        for r in range(15):
            for c in range(15):
                if MAZE_LAYOUT[r][c] == 0:
                    pos = (r, c)
                    if pos not in self.power_pellets and pos != PACMAN_START:
                        self.dots.add(pos)

    def _get_valid_mask(self) -> np.ndarray:
        """Calculate valid directions from current Pacman position."""
        mask = np.zeros((4,), dtype=np.int8)
        py, px = self.pacman_pos
        for d in range(4):
            dy, dx = DIR_VECTORS[d]
            ny, nx = py + dy, px + dx
            if 0 <= ny < 15 and 0 <= nx < 15 and MAZE_LAYOUT[ny][nx] == 0:
                mask[d] = 1
        return mask

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Generate observation dict."""
        return {
            "observation": self._render_frame(),
            "valid_mask": self._get_valid_mask(),
            "score": np.array([self.score], dtype=np.int32),
            "lives": np.array([self.lives], dtype=np.int32),
        }

    def _get_ghost_next_move(self, ghost: Dict[str, Any], target: Tuple[int, int]) -> Tuple[Tuple[int, int], int]:
        """Determine next cell and direction for a ghost using standard Pacman AI."""
        gy, gx = ghost["pos"]
        current_dir = ghost["dir"]
        
        # Evaluate all 4 possible movements
        allowed = []
        for d in range(4):
            dy, dx = DIR_VECTORS[d]
            ny, nx = gy + dy, gx + dx
            
            # Check boundaries and walls
            if 0 <= ny < 15 and 0 <= nx < 15 and MAZE_LAYOUT[ny][nx] == 0:
                # Ghost cannot reverse unless forced
                if d != OPPOSITES[current_dir]:
                    allowed.append(((ny, nx), d))
        
        if not allowed:
            # Reversing allowed as last resort
            for d in range(4):
                dy, dx = DIR_VECTORS[d]
                ny, nx = gy + dy, gx + dx
                if 0 <= ny < 15 and 0 <= nx < 15 and MAZE_LAYOUT[ny][nx] == 0:
                    allowed.append(((ny, nx), d))
                    
        if not allowed:
            return (gy, gx), current_dir

        if self.frightened_timer > 0:
            # Frightened mode: wander randomly
            idx = self.np_random.integers(len(allowed))
            return allowed[idx]
        else:
            # Normal mode: choose move that minimizes Euclidean distance to target
            ty, tx = target
            best_idx = 0
            min_dist = float("inf")
            # Tie breaker ordering: UP (0), LEFT (2), DOWN (1), RIGHT (3)
            tie_breaker = [UP, LEFT, DOWN, RIGHT]
            
            for i, ((ny, nx), d) in enumerate(allowed):
                dist = (ny - ty) ** 2 + (nx - tx) ** 2
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
                elif dist == min_dist:
                    # Resolve tie
                    d_curr = allowed[i][1]
                    d_best = allowed[best_idx][1]
                    if tie_breaker.index(d_curr) < tie_breaker.index(d_best):
                        best_idx = i
            return allowed[best_idx]

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one step."""
        if not (0 <= action <= 3):
            raise ValueError(f"Invalid action: {action}")

        self.steps += 1
        reward = -0.01  # Small step penalty
        
        # 1. Update Pacman Position
        old_pacman_pos = self.pacman_pos
        py, px = self.pacman_pos
        dy, dx = DIR_VECTORS[action]
        ny, nx = py + dy, px + dx
        
        # Check if the requested direction is valid
        if 0 <= ny < 15 and 0 <= nx < 15 and MAZE_LAYOUT[ny][nx] == 0:
            self.pacman_dir = action
            self.pacman_pos = (ny, nx)
        else:
            # Continue in current direction if possible
            cdy, cdx = DIR_VECTORS[self.pacman_dir]
            cny, cnx = py + cdy, px + cdx
            if 0 <= cny < 15 and 0 <= cnx < 15 and MAZE_LAYOUT[cny][cnx] == 0:
                self.pacman_pos = (cny, cnx)

        # 2. Update frightened mode timer
        if self.frightened_timer > 0:
            self.frightened_timer -= 1

        # 3. Move Ghosts
        old_ghost_positions = {}
        for g in self.ghosts:
            old_ghost_positions[g["name"]] = g["pos"]

        blinky_pos = None
        for g in self.ghosts:
            if g["name"] == "Blinky":
                blinky_pos = g["pos"]
                break
        if blinky_pos is None:
            blinky_pos = BLINKY_START

        for g in self.ghosts:
            # Frightened ghosts move at half speed (only on even step counts)
            if self.frightened_timer > 0 and self.steps % 2 != 0:
                continue

            # Determine target cell
            gname = g["name"]
            gy, gx = g["pos"]
            target = (0, 0)
            
            if gname == "Blinky":
                target = self.pacman_pos
            elif gname == "Pinky":
                pdy, pdx = DIR_VECTORS[self.pacman_dir]
                target = (self.pacman_pos[0] + 2 * pdy, self.pacman_pos[1] + 2 * pdx)
            elif gname == "Inky":
                # Vector from Blinky to 2-cells-ahead of Pacman, doubled
                pdy, pdx = DIR_VECTORS[self.pacman_dir]
                ahead_y = self.pacman_pos[0] + 2 * pdy
                ahead_x = self.pacman_pos[1] + 2 * pdx
                target = (
                    blinky_pos[0] + 2 * (ahead_y - blinky_pos[0]),
                    blinky_pos[1] + 2 * (ahead_x - blinky_pos[1]),
                )
            elif gname == "Clyde":
                dist = math.sqrt((gy - self.pacman_pos[0]) ** 2 + (gx - self.pacman_pos[1]) ** 2)
                if dist < 4.0:
                    target = (13, 1)  # Bottom-left corner
                else:
                    target = self.pacman_pos
                    
            next_pos, next_dir = self._get_ghost_next_move(g, target)
            g["pos"] = next_pos
            g["dir"] = next_dir

        # 4. Check for Collisions
        terminated = False
        
        # Helper to process collisions
        def handle_collisions():
            nonlocal reward, terminated
            for g in self.ghosts:
                old_g_pos = old_ghost_positions.get(g["name"], g["pos"])
                if g["pos"] == self.pacman_pos or (g["pos"] == old_pacman_pos and old_g_pos == self.pacman_pos):
                    if self.frightened_timer > 0:
                        # Eat ghost
                        self.score += 200
                        reward += 200.0
                        g["pos"] = g["start_pos"]
                        g["dir"] = UP
                    else:
                        # Pacman dies
                        self.lives -= 1
                        reward -= 50.0
                        if self.lives <= 0:
                            terminated = True
                        else:
                            # Reset Pacman and ghosts to start, but keep level dots
                            self.pacman_pos = PACMAN_START
                            self.pacman_dir = RIGHT
                            for gh in self.ghosts:
                                gh["pos"] = gh["start_pos"]
                                gh["dir"] = UP
                        return True
            return False

        # Collision check after positions update
        handle_collisions()

        # 5. Check dot/power pellet consumption
        if self.pacman_pos in self.dots:
            self.dots.remove(self.pacman_pos)
            self.score += 10
            reward += 10.0

        elif self.pacman_pos in self.power_pellets:
            self.power_pellets.remove(self.pacman_pos)
            self.score += 50
            reward += 50.0
            self.frightened_timer = 40

        # Check win / level reset
        if not self.dots and not self.power_pellets:
            # Clear level!
            self.score += 500
            reward += 500.0
            self._reset_level_entities()

        obs = self._get_obs()
        info = {
            "score": self.score,
            "lives": self.lives,
            "frightened_timer": self.frightened_timer,
        }

        return obs, float(reward), terminated, False, info

    def render(self) -> Optional[np.ndarray]:
        """Render the environment for Gymnasium interface."""
        return self._render_frame()

    def _render_frame(self) -> np.ndarray:
        """Render the 1200x900 canvas and downsample to 400x300 using LANCZOS."""
        canvas_w = WIDTH * self.SF
        canvas_h = HEIGHT * self.SF
        canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        cell_size = 16 * self.SF
        x_offset = 80 * self.SF
        y_offset = 50 * self.SF

        # 1. Draw Maze Walls
        for r in range(15):
            for c in range(15):
                x1 = x_offset + c * cell_size
                y1 = y_offset + r * cell_size
                x2 = x1 + cell_size - 1
                y2 = y1 + cell_size - 1
                if MAZE_LAYOUT[r][c] == 1:
                    # Neon-style walls
                    draw.rectangle([x1, y1, x2, y2], fill=(10, 10, 30), outline=(25, 25, 255), width=2 * self.SF)

        # 2. Draw Dots
        for r, c in self.dots:
            x = x_offset + c * cell_size + cell_size // 2
            y = y_offset + r * cell_size + cell_size // 2
            rad = 3 * self.SF
            draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=(255, 255, 150))

        # 3. Draw Power Pellets
        for r, c in self.power_pellets:
            x = x_offset + c * cell_size + cell_size // 2
            y = y_offset + r * cell_size + cell_size // 2
            # Flashing effect
            if (self.steps // 5) % 2 == 0:
                rad = 7 * self.SF
                draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=(255, 180, 0))

        # 4. Draw Pacman
        py, px = self.pacman_pos
        cx = x_offset + px * cell_size + cell_size // 2
        cy = y_offset + py * cell_size + cell_size // 2
        rad = 8 * self.SF - 2

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

        # 5. Draw Ghosts
        for g in self.ghosts:
            gy, gx = g["pos"]
            cx = x_offset + gx * cell_size + cell_size // 2
            cy = y_offset + gy * cell_size + cell_size // 2
            g_rad = 8 * self.SF - 2
            
            # Determine color
            if self.frightened_timer > 0:
                # Warning flash when frightened mode runs low (last 10 steps)
                if self.frightened_timer < 10 and (self.steps // 5) % 2 == 0:
                    color = (255, 255, 255)  # Flash white
                    eye_color = (255, 0, 0)
                else:
                    color = (50, 50, 255)   # Frightened blue
                    eye_color = (255, 165, 0)
            else:
                eye_color = None
                if g["name"] == "Blinky":
                    color = (255, 0, 0)
                elif g["name"] == "Pinky":
                    color = (255, 105, 180)
                elif g["name"] == "Inky":
                    color = (0, 255, 255)
                else:
                    color = (255, 165, 0)

            # Draw Ghost shape (top half semicircle, bottom rectangle + spikes)
            draw.chord([cx - g_rad, cy - g_rad, cx + g_rad, cy + g_rad], start=180, end=360, fill=color)
            draw.rectangle([cx - g_rad, cy, cx + g_rad, cy + g_rad - 2 * self.SF], fill=color)
            
            # Bottom spikes
            w = 2 * g_rad
            h_spike = 2 * self.SF
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

            # Eyes
            if self.frightened_timer > 0:
                # Orange worried eyes
                draw.ellipse([cx - 4 * self.SF, cy - 2 * self.SF, cx - 1 * self.SF, cy + 1 * self.SF], fill=eye_color)
                draw.ellipse([cx + 1 * self.SF, cy - 2 * self.SF, cx + 4 * self.SF, cy + 1 * self.SF], fill=eye_color)
                # Wavy worried mouth
                draw.line(
                    [
                        (cx - 5 * self.SF, cy + 4 * self.SF),
                        (cx - 2.5 * self.SF, cy + 2.5 * self.SF),
                        (cx, cy + 4 * self.SF),
                        (cx + 2.5 * self.SF, cy + 2.5 * self.SF),
                        (cx + 5 * self.SF, cy + 4 * self.SF),
                    ],
                    fill=eye_color,
                    width=1 * self.SF,
                )
            else:
                # Normal eyes looking in ghost's movement direction
                gdy, gdx = DIR_VECTORS[g["dir"]]
                ex_offset = gdx * 2 * self.SF
                ey_offset = gdy * 2 * self.SF
                
                # Left eye
                el_cx = cx - 4 * self.SF + ex_offset
                el_cy = cy - 2 * self.SF + ey_offset
                draw.ellipse([el_cx - 2 * self.SF, el_cy - 2 * self.SF, el_cx + 2 * self.SF, el_cy + 2 * self.SF], fill=(255, 255, 255))
                draw.ellipse([el_cx - 1 * self.SF + ex_offset / 2, el_cy - 1 * self.SF + ey_offset / 2, el_cx + 1 * self.SF + ex_offset / 2, el_cy + 1 * self.SF + ey_offset / 2], fill=(0, 0, 255))
                
                # Right eye
                er_cx = cx + 4 * self.SF + ex_offset
                er_cy = cy - 2 * self.SF + ey_offset
                draw.ellipse([er_cx - 2 * self.SF, er_cy - 2 * self.SF, er_cx + 2 * self.SF, er_cy + 2 * self.SF], fill=(255, 255, 255))
                draw.ellipse([er_cx - 1 * self.SF + ex_offset / 2, er_cy - 1 * self.SF + ey_offset / 2, er_cx + 1 * self.SF + ex_offset / 2, er_cy + 1 * self.SF + ey_offset / 2], fill=(0, 0, 255))

        # 6. Draw HUD
        hud_y = 15 * self.SF
        draw.text((15 * self.SF, hud_y), f"SCORE: {self.score}", fill=(255, 255, 255), font=self._hud_font)
        draw.text((385 * self.SF, hud_y), f"LIVES: {self.lives}", fill=(255, 255, 255), font=self._hud_font, anchor="rt")

        # Downsample using high-quality LANCZOS anti-aliasing
        canvas_resized = canvas.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
        return np.array(canvas_resized, dtype=np.uint8)

    def close(self) -> None:
        """Close the environment."""
        pass

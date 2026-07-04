"""A Gymnasium environment for continuous 2D physics-based Air Hockey."""

import copy
from typing import Any, Tuple, Dict, Optional, List

import gymnasium as gym
from absl import logging
from gymnasium import spaces
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import numpy.typing as npt

# Physical Constants
TABLE_WIDTH = 300
TABLE_HEIGHT = 500
GOAL_WIDTH = 100
MALLET_RADIUS = 18
PUCK_RADIUS = 10
MAX_MALLET_SPEED = 8.0
PUCK_FRICTION = 0.99
MAX_PUCK_SPEED = 18.0
PADDING_PX = 8

# Render colors
COLOR_BG = (15, 23, 42)             # Dark slate
COLOR_GRID = (30, 41, 59)
COLOR_BORDER = (71, 85, 105)
COLOR_CENTERLINE = (51, 65, 85)
COLOR_P1 = (14, 165, 233)           # Glowing light blue
COLOR_P2 = (249, 115, 22)           # Glowing orange
COLOR_PUCK = (239, 68, 68)          # Bright red
COLOR_GOAL = (15, 23, 42)
COLOR_TEXT = (248, 250, 252)

HEADER_PX = 50
FOOTER_PX = 30

CANVAS_SIZE = (
    TABLE_WIDTH,
    TABLE_HEIGHT + HEADER_PX + FOOTER_PX,
)


class GymAirHockeyEnv(gym.Env):
    """A Gymnasium environment for two-player continuous Air Hockey."""

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
        # Action space: [P1 mallet movement vector (dx, dy), P2 mallet movement vector (dx, dy)]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2, 2), dtype=np.float32
        )
        # Observation: [p1_x, p1_y, p1_vx, p1_vy, p2_x, p2_y, p2_vx, p2_vy, puck_x, puck_y, puck_vx, puck_vy] (all normalized)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-2.0, high=2.0, shape=(12,), dtype=np.float32
                ),
                "total_score": spaces.Box(
                    low=0, high=10, shape=(2,), dtype=np.int32
                ),
            }
        )

        # Pre-allocate canvas base
        self._base_canvas = Image.new("RGB", CANVAS_SIZE, COLOR_BG)
        draw_base = ImageDraw.Draw(self._base_canvas)
        draw_base.rectangle([0, 0, CANVAS_SIZE[0] - 1, HEADER_PX - 1], fill=COLOR_GRID)
        draw_base.rectangle(
            [0, CANVAS_SIZE[1] - FOOTER_PX, CANVAS_SIZE[0] - 1, CANVAS_SIZE[1] - 1],
            fill=COLOR_GRID,
        )

        # Draw playing board borders and lines
        # Side walls
        draw_base.rectangle(
            [0, HEADER_PX, TABLE_WIDTH - 1, TABLE_HEIGHT + HEADER_PX - 1],
            outline=COLOR_BORDER,
            width=3,
        )
        # Center Line
        draw_base.line(
            [(0, HEADER_PX + TABLE_HEIGHT // 2), (TABLE_WIDTH, HEADER_PX + TABLE_HEIGHT // 2)],
            fill=COLOR_CENTERLINE,
            width=2,
        )
        # Center Circle
        draw_base.ellipse(
            [
                TABLE_WIDTH // 2 - 40,
                HEADER_PX + TABLE_HEIGHT // 2 - 40,
                TABLE_WIDTH // 2 + 40,
                HEADER_PX + TABLE_HEIGHT // 2 + 40,
            ],
            outline=COLOR_CENTERLINE,
            width=2,
        )

        # Goals cutouts (draw them as goal indicators)
        goal_left = (TABLE_WIDTH - GOAL_WIDTH) // 2
        goal_right = goal_left + GOAL_WIDTH
        draw_base.line(
            [(goal_left, HEADER_PX), (goal_right, HEADER_PX)],
            fill=(239, 68, 68),
            width=4,
        )
        draw_base.line(
            [(goal_left, HEADER_PX + TABLE_HEIGHT - 1), (goal_right, HEADER_PX + TABLE_HEIGHT - 1)],
            fill=(239, 68, 68),
            width=4,
        )

        self.reset()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment state."""
        super().reset(seed=seed)

        if options is not None and "state" in options:
            state = options["state"]
            self._p1_pos = np.array(state["p1_pos"], dtype=np.float32)
            self._p1_vel = np.array(state["p1_vel"], dtype=np.float32)
            self._p2_pos = np.array(state["p2_pos"], dtype=np.float32)
            self._p2_vel = np.array(state["p2_vel"], dtype=np.float32)
            self._puck_pos = np.array(state["puck_pos"], dtype=np.float32)
            self._puck_vel = np.array(state["puck_vel"], dtype=np.float32)
            self._scores = np.array(state["scores"], dtype=np.int32)
            self._steps = state["steps"]
            return self._create_observation(), {}

        # Reset P1 (Bottom)
        self._p1_pos = np.array([TABLE_WIDTH / 2, TABLE_HEIGHT * 0.75], dtype=np.float32)
        self._p1_vel = np.zeros(2, dtype=np.float32)

        # Reset P2 (Top)
        self._p2_pos = np.array([TABLE_WIDTH / 2, TABLE_HEIGHT * 0.25], dtype=np.float32)
        self._p2_vel = np.zeros(2, dtype=np.float32)

        # Reset Puck (Center)
        self._puck_pos = np.array([TABLE_WIDTH / 2, TABLE_HEIGHT / 2], dtype=np.float32)
        
        # Initial puck launch direction towards random player
        puck_dir_y = 1.0 if self.np_random.random() < 0.5 else -1.0
        self._puck_vel = np.array([self.np_random.uniform(-1.0, 1.0), puck_dir_y * 3.0], dtype=np.float32)

        self._scores = np.zeros(2, dtype=np.int32)
        self._steps = 0

        return self._create_observation(), {}

    def _create_observation(self) -> Dict[str, Any]:
        """Create normalized features vector for observation."""
        obs = np.array(
            [
                self._p1_pos[0] / TABLE_WIDTH,
                self._p1_pos[1] / TABLE_HEIGHT,
                self._p1_vel[0] / MAX_MALLET_SPEED,
                self._p1_vel[1] / MAX_MALLET_SPEED,
                
                self._p2_pos[0] / TABLE_WIDTH,
                self._p2_pos[1] / TABLE_HEIGHT,
                self._p2_vel[0] / MAX_MALLET_SPEED,
                self._p2_vel[1] / MAX_MALLET_SPEED,
                
                self._puck_pos[0] / TABLE_WIDTH,
                self._puck_pos[1] / TABLE_HEIGHT,
                self._puck_vel[0] / MAX_PUCK_SPEED,
                self._puck_vel[1] / MAX_PUCK_SPEED,
            ],
            dtype=np.float32,
        )
        return {
            "observation": obs,
            "total_score": self._scores.copy(),
        }

    def _resolve_collision(self, mallet_pos: npt.NDArray[np.float32], mallet_vel: npt.NDArray[np.float32]) -> None:
        """Resolve circles intersection collision and apply impulse transfer."""
        d = self._puck_pos - mallet_pos
        dist = np.linalg.norm(d)
        min_dist = MALLET_RADIUS + PUCK_RADIUS

        if dist < min_dist:
            if dist == 0:
                n = np.array([1.0, 0.0], dtype=np.float32)
            else:
                n = d / dist

            # Push out to resolve overlap
            self._puck_pos = mallet_pos + n * min_dist

            # Calculate relative normal velocity
            rel_vel = self._puck_vel - mallet_vel
            vel_along_normal = np.dot(rel_vel, n)

            # Rebound if they are moving towards each other
            if vel_along_normal < 0:
                restitution = 1.0
                impulse = -(1.0 + restitution) * vel_along_normal
                self._puck_vel += impulse * n + mallet_vel * 0.5
                
                # Cap speed
                puck_speed = np.linalg.norm(self._puck_vel)
                if puck_speed > MAX_PUCK_SPEED:
                    self._puck_vel = (self._puck_vel / puck_speed) * MAX_PUCK_SPEED

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Advance the physics simulator by one time step."""
        p1_act = np.clip(action[0], -1.0, 1.0)
        p2_act = np.clip(action[1], -1.0, 1.0)
        self._steps += 1

        # 1. Update Mallets velocity and position
        self._p1_vel = p1_act * MAX_MALLET_SPEED
        self._p2_vel = p2_act * MAX_MALLET_SPEED

        p1_next = self._p1_pos + self._p1_vel
        p2_next = self._p2_pos + self._p2_vel

        # Enforce walls and centerline boundaries
        # Player 1 (Bottom half)
        p1_next[0] = np.clip(p1_next[0], MALLET_RADIUS, TABLE_WIDTH - MALLET_RADIUS)
        p1_next[1] = np.clip(p1_next[1], TABLE_HEIGHT / 2 + MALLET_RADIUS, TABLE_HEIGHT - MALLET_RADIUS)
        self._p1_pos = p1_next

        # Player 2 (Top half)
        p2_next[0] = np.clip(p2_next[0], MALLET_RADIUS, TABLE_WIDTH - MALLET_RADIUS)
        p2_next[1] = np.clip(p2_next[1], MALLET_RADIUS, TABLE_HEIGHT / 2 - MALLET_RADIUS)
        self._p2_pos = p2_next

        # 2. Update Puck Position
        self._puck_pos += self._puck_vel
        self._puck_vel *= PUCK_FRICTION  # Damping friction

        # 3. Handle Puck Boundary collisions
        # Bounce off side walls
        if self._puck_pos[0] <= PUCK_RADIUS:
            self._puck_pos[0] = PUCK_RADIUS
            self._puck_vel[0] *= -1.0
        elif self._puck_pos[0] >= TABLE_WIDTH - PUCK_RADIUS:
            self._puck_pos[0] = TABLE_WIDTH - PUCK_RADIUS
            self._puck_vel[0] *= -1.0

        # Goal coordinates
        goal_left = (TABLE_WIDTH - GOAL_WIDTH) / 2
        goal_right = goal_left + GOAL_WIDTH

        reward = 0.0
        goal_scored = False

        # Top boundary check
        if self._puck_pos[1] <= PUCK_RADIUS:
            if goal_left <= self._puck_pos[0] <= goal_right:
                # Goal for P1!
                self._scores[0] += 1
                reward = 1.0
                goal_scored = True
            else:
                self._puck_pos[1] = PUCK_RADIUS
                self._puck_vel[1] *= -1.0

        # Bottom boundary check
        elif self._puck_pos[1] >= TABLE_HEIGHT - PUCK_RADIUS:
            if goal_left <= self._puck_pos[0] <= goal_right:
                # Goal for P2!
                self._scores[1] += 1
                reward = -1.0
                goal_scored = True
            else:
                self._puck_pos[1] = TABLE_HEIGHT - PUCK_RADIUS
                self._puck_vel[1] *= -1.0

        # 4. Resolve Mallet-Puck Collisions
        self._resolve_collision(self._p1_pos, self._p1_vel)
        self._resolve_collision(self._p2_pos, self._p2_vel)

        # 5. If goal, reset puck position to center
        if goal_scored:
            self._puck_pos = np.array([TABLE_WIDTH / 2, TABLE_HEIGHT / 2], dtype=np.float32)
            puck_dir_y = 1.0 if self.np_random.random() < 0.5 else -1.0
            self._puck_vel = np.array([self.np_random.uniform(-1.0, 1.0), puck_dir_y * 3.0], dtype=np.float32)

        # Terminate if a player reaches 7 goals
        terminated = False
        if self._scores[0] >= 7 or self._scores[1] >= 7:
            terminated = True
            if self._scores[0] >= 7:
                reward += 10.0  # P1 wins
            else:
                reward -= 10.0  # P2 wins

        truncated = self._steps >= 1000

        return self._create_observation(), float(reward), terminated, truncated, {"state": self._get_state()}

    def _get_state(self) -> Dict[str, Any]:
        """Return environment internal state dictionary."""
        return {
            "p1_pos": list(self._p1_pos),
            "p1_vel": list(self._p1_vel),
            "p2_pos": list(self._p2_pos),
            "p2_vel": list(self._p2_vel),
            "puck_pos": list(self._puck_pos),
            "puck_vel": list(self._puck_vel),
            "scores": list(self._scores),
            "steps": self._steps,
        }

    def render(self) -> Optional[npt.NDArray[np.uint8]]:
        """Render the environment to an RGB canvas array."""
        canvas = self._base_canvas.copy()
        draw = ImageDraw.Draw(canvas)

        # Header Info
        draw.text(
            (PADDING_PX + 5, HEADER_PX // 2),
            "AIR HOCKEY 2D",
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

        # Draw Player 1 Mallet (Blue)
        p1_x, p1_y = int(self._p1_pos[0]), int(self._p1_pos[1] + HEADER_PX)
        draw.ellipse(
            [p1_x - MALLET_RADIUS, p1_y - MALLET_RADIUS, p1_x + MALLET_RADIUS, p1_y + MALLET_RADIUS],
            fill=COLOR_P1,
            outline=(255, 255, 255),
            width=2,
        )
        draw.ellipse(
            [p1_x - MALLET_RADIUS // 2, p1_y - MALLET_RADIUS // 2, p1_x + MALLET_RADIUS // 2, p1_y + MALLET_RADIUS // 2],
            outline=(255, 255, 255),
            width=1,
        )

        # Draw Player 2 Mallet (Orange)
        p2_x, p2_y = int(self._p2_pos[0]), int(self._p2_pos[1] + HEADER_PX)
        draw.ellipse(
            [p2_x - MALLET_RADIUS, p2_y - MALLET_RADIUS, p2_x + MALLET_RADIUS, p2_y + MALLET_RADIUS],
            fill=COLOR_P2,
            outline=(255, 255, 255),
            width=2,
        )
        draw.ellipse(
            [p2_x - MALLET_RADIUS // 2, p2_y - MALLET_RADIUS // 2, p2_x + MALLET_RADIUS // 2, p2_y + MALLET_RADIUS // 2],
            outline=(255, 255, 255),
            width=1,
        )

        # Draw Puck (Red)
        puck_x, puck_y = int(self._puck_pos[0]), int(self._puck_pos[1] + HEADER_PX)
        draw.ellipse(
            [puck_x - PUCK_RADIUS, puck_y - PUCK_RADIUS, puck_x + PUCK_RADIUS, puck_y + PUCK_RADIUS],
            fill=COLOR_PUCK,
            outline=(255, 255, 255),
            width=1,
        )

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

"""A Gymnasium environment for continuous 2D physics-based Air Hockey."""

import copy
import math
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

        # Scale Factor for supersampling anti-aliasing
        self.SF = 3

        # Font setup (multiplied by SF for crispness when scaled down)
        try:
            font_properties = font_manager.FontProperties(
                family="sans-serif", weight="bold"
            )
            font_file = font_manager.findfont(font_properties)
            self._title_font = ImageFont.truetype(font_file, 14 * self.SF)
            self._stats_font = ImageFont.truetype(font_file, 10 * self.SF)
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

        # Pre-allocate canvas base at 3x scale
        self._base_canvas = Image.new("RGB", (CANVAS_SIZE[0] * self.SF, CANVAS_SIZE[1] * self.SF), (10, 15, 26))  # Deep space background
        draw_base = ImageDraw.Draw(self._base_canvas)
        draw_base.rectangle([0, 0, CANVAS_SIZE[0] * self.SF - 1, HEADER_PX * self.SF - 1], fill=(15, 23, 42))
        draw_base.rectangle(
            [0, (CANVAS_SIZE[1] - FOOTER_PX) * self.SF, CANVAS_SIZE[0] * self.SF - 1, CANVAS_SIZE[1] * self.SF - 1],
            fill=(10, 15, 28),
        )

        # Draw playing board borders and lines with cyber glow aesthetics at 3x scale
        # Side walls
        draw_base.rectangle(
            [0, HEADER_PX * self.SF, TABLE_WIDTH * self.SF - 1, (TABLE_HEIGHT + HEADER_PX) * self.SF - 1],
            outline=(40, 55, 80),
            width=3 * self.SF,
        )
        # Inner glowing border frame
        draw_base.rectangle(
            [3 * self.SF, (HEADER_PX + 3) * self.SF, (TABLE_WIDTH - 4) * self.SF, (TABLE_HEIGHT + HEADER_PX - 4) * self.SF],
            outline=(24, 70, 110),
            width=1 * self.SF,
        )
        
        # Center Line (neon blue/gray)
        draw_base.line(
            [(0, (HEADER_PX + TABLE_HEIGHT // 2) * self.SF), (TABLE_WIDTH * self.SF, (HEADER_PX + TABLE_HEIGHT // 2) * self.SF)],
            fill=(20, 60, 95),
            width=2 * self.SF,
        )
        # Center Circle (neon blue/gray)
        draw_base.ellipse(
            [
                (TABLE_WIDTH // 2 - 40) * self.SF,
                (HEADER_PX + TABLE_HEIGHT // 2 - 40) * self.SF,
                (TABLE_WIDTH // 2 + 40) * self.SF,
                (HEADER_PX + TABLE_HEIGHT // 2 + 40) * self.SF,
            ],
            outline=(20, 60, 95),
            width=2 * self.SF,
        )

        # Goals cutouts (draw them as glowing red laser gates)
        goal_left = (TABLE_WIDTH - GOAL_WIDTH) // 2
        goal_right = goal_left + GOAL_WIDTH
        
        # Top goal red laser glow
        draw_base.line(
            [(goal_left * self.SF, HEADER_PX * self.SF), (goal_right * self.SF, HEADER_PX * self.SF)],
            fill=(239, 68, 68),
            width=4 * self.SF,
        )
        # Bottom goal red laser glow
        draw_base.line(
            [(goal_left * self.SF, (HEADER_PX + TABLE_HEIGHT - 1) * self.SF), (goal_right * self.SF, (HEADER_PX + TABLE_HEIGHT - 1) * self.SF)],
            fill=(239, 68, 68),
            width=4 * self.SF,
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
        """Render the environment to an RGB canvas array with supersampling anti-aliasing."""
        canvas = self._base_canvas.copy()
        
        # RGBA overlay for soft drop shadows and glowing motion trails at 3x scale
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        ol_draw = ImageDraw.Draw(overlay)
        draw = ImageDraw.Draw(canvas)

        # Header Info at 3x scale
        draw.text(
            ((PADDING_PX + 5) * self.SF, (HEADER_PX // 2) * self.SF),
            "AIR HOCKEY 2D",
            fill=COLOR_TEXT,
            font=self._title_font,
            anchor="lm",
        )

        draw.text(
            ((CANVAS_SIZE[0] - PADDING_PX - 5) * self.SF, (HEADER_PX // 2) * self.SF),
            f"P1: {self._scores[0]} | P2: {self._scores[1]}",
            fill=COLOR_TEXT,
            font=self._title_font,
            anchor="rm",
        )

        # Convert positions to scale
        p1_x = int(self._p1_pos[0] * self.SF)
        p1_y = int((self._p1_pos[1] + HEADER_PX) * self.SF)
        p2_x = int(self._p2_pos[0] * self.SF)
        p2_y = int((self._p2_pos[1] + HEADER_PX) * self.SF)
        puck_x = int(self._puck_pos[0] * self.SF)
        puck_y = int((self._puck_pos[1] + HEADER_PX) * self.SF)

        m_rad = MALLET_RADIUS * self.SF
        p_rad = PUCK_RADIUS * self.SF

        # 1. Draw Drop Shadows (floating effect)
        ol_draw.ellipse([p1_x - m_rad + 3 * self.SF, p1_y - m_rad + 4 * self.SF, p1_x + m_rad + 3 * self.SF, p1_y + m_rad + 4 * self.SF], fill=(0, 0, 0, 90))
        ol_draw.ellipse([p2_x - m_rad + 3 * self.SF, p2_y - m_rad + 4 * self.SF, p2_x + m_rad + 3 * self.SF, p2_y + m_rad + 4 * self.SF], fill=(0, 0, 0, 90))
        ol_draw.ellipse([puck_x - p_rad + 2 * self.SF, puck_y - p_rad + 3 * self.SF, puck_x + p_rad + 2 * self.SF, puck_y + p_rad + 3 * self.SF], fill=(0, 0, 0, 90))

        # 2. Draw Puck motion blur/energy trail based on velocity
        puck_speed = math.hypot(self._puck_vel[0], self._puck_vel[1])
        if puck_speed > 0.5:
            # Draw 3 fading trail segments
            for i in range(1, 4):
                tx = int((self._puck_pos[0] - self._puck_vel[0] * i * 0.5) * self.SF)
                ty = int((self._puck_pos[1] + HEADER_PX - self._puck_vel[1] * i * 0.5) * self.SF)
                t_rad = max(2 * self.SF, p_rad - i * self.SF)
                alpha = int(140 / (i + 1))
                ol_draw.ellipse([tx - t_rad, ty - t_rad, tx + t_rad, ty + t_rad], fill=(239, 68, 68, alpha))

        # Merge shadows/trails overlay onto main table canvas
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(canvas)

        # 3. Draw Player 1 Mallet (Blue Shaded neon disc)
        # Outer base
        draw.ellipse(
            [p1_x - m_rad, p1_y - m_rad, p1_x + m_rad, p1_y + m_rad],
            fill=(10, 110, 160),
            outline=(255, 255, 255),
            width=2 * self.SF,
        )
        # Glowing inner core
        draw.ellipse(
            [p1_x - m_rad + 3 * self.SF, p1_y - m_rad + 3 * self.SF, p1_x + m_rad - 3 * self.SF, p1_y + m_rad - 3 * self.SF],
            fill=COLOR_P1,
        )
        # Inner grip ring
        draw.ellipse(
            [p1_x - m_rad // 2, p1_y - m_rad // 2, p1_x + m_rad // 2, p1_y + m_rad // 2],
            outline=(255, 255, 255),
            width=1 * self.SF,
        )
        # Specular light reflect
        draw.ellipse(
            [p1_x - 6 * self.SF, p1_y - 8 * self.SF, p1_x - 2 * self.SF, p1_y - 5 * self.SF],
            fill=(255, 255, 255),
        )

        # 4. Draw Player 2 Mallet (Orange Shaded neon disc)
        # Outer base
        draw.ellipse(
            [p2_x - m_rad, p2_y - m_rad, p2_x + m_rad, p2_y + m_rad],
            fill=(180, 80, 10),
            outline=(255, 255, 255),
            width=2 * self.SF,
        )
        # Glowing inner core
        draw.ellipse(
            [p2_x - m_rad + 3 * self.SF, p2_y - m_rad + 3 * self.SF, p2_x + m_rad - 3 * self.SF, p2_y + m_rad - 3 * self.SF],
            fill=COLOR_P2,
        )
        # Inner grip ring
        draw.ellipse(
            [p2_x - m_rad // 2, p2_y - m_rad // 2, p2_x + m_rad // 2, p2_y + m_rad // 2],
            outline=(255, 255, 255),
            width=1 * self.SF,
        )
        # Specular light reflect
        draw.ellipse(
            [p2_x - 6 * self.SF, p2_y - 8 * self.SF, p2_x - 2 * self.SF, p2_y - 5 * self.SF],
            fill=(255, 255, 255),
        )

        # 5. Draw Puck (Red Glowing Core)
        # Outer red rim
        draw.ellipse(
            [puck_x - p_rad, puck_y - p_rad, puck_x + p_rad, puck_y + p_rad],
            fill=(180, 30, 30),
            outline=(255, 255, 255),
            width=1 * self.SF,
        )
        # Glowing center
        draw.ellipse(
            [puck_x - p_rad + 2 * self.SF, puck_y - p_rad + 2 * self.SF, puck_x + p_rad - 2 * self.SF, puck_y + p_rad - 2 * self.SF],
            fill=(255, 80, 80),
        )
        # Specular light reflect
        draw.ellipse(
            [puck_x - 3 * self.SF, puck_y - 4 * self.SF, puck_x - 1 * self.SF, puck_y - 2 * self.SF],
            fill=(255, 255, 255),
        )

        # Footer stats
        draw.text(
            ((PADDING_PX + 5) * self.SF, (CANVAS_SIZE[1] - FOOTER_PX // 2) * self.SF),
            f"Steps: {self._steps}",
            fill=(180, 180, 180),
            font=self._stats_font,
            anchor="lm",
        )

        # Downsample using high-quality LANCZOS anti-aliasing
        canvas_resized = canvas.resize(CANVAS_SIZE, Image.Resampling.LANCZOS)
        return np.array(canvas_resized, dtype=np.uint8)

    def close(self) -> None:
        """Close the environment."""
        pass

"""A Gymnasium environment for a 2D Platformer game."""

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from absl import logging
from gymnasium import spaces
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Actions: 0: Idle, 1: Move Left, 2: Move Right, 3: Jump
IDLE = 0
LEFT = 1
RIGHT = 2
JUMP = 3

# Physical constants
STAGE_WIDTH = 800
STAGE_HEIGHT = 300
VIEW_WIDTH = 400
VIEW_HEIGHT = 300

GRAVITY = 0.4
JUMP_SPEED = -8.0
ACCEL = 0.5
MAX_VX = 4.0
FRICTION = 0.85

PLAYER_WIDTH = 16
PLAYER_HEIGHT = 24

# Static level structures at 1x scale
SOLID_BLOCKS = [
    # Ground platforms
    (0, 260, 300, 300),
    (380, 260, 800, 300),
    # Floating platforms
    (200, 200, 280, 215),
    (320, 160, 380, 175),
    (420, 200, 500, 215),
    (550, 140, 680, 155),
]

SPIKES = [
    # Spike pit in the gap
    (300, 280, 380, 300),
    # Hazard on the right ground
    (600, 250, 630, 260),
]

COINS_START = [
    (240, 170),
    (350, 130),
    (460, 170),
    (615, 110),
    (700, 230),
]

FLAG_X = 750
FLAG_Y = 200


class GymPlatformerEnv(gym.Env):
    """A Gymnasium environment for a 2D Platformer game."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.SF = 3  # SSAA scale factor

        # Action Space: 0: Idle, 1: Move Left, 2: Move Right, 3: Jump
        self.action_space = spaces.Discrete(4)

        # Observation Space: Dict with observation (300, 400, 3), valid_mask, level_progress
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0, high=255, shape=(300, 400, 3), dtype=np.uint8
                ),
                "valid_mask": spaces.Box(
                    low=0, high=1, shape=(4,), dtype=np.int8
                ),
                "level_progress": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
            }
        )

        # Font setup (scaled by SF)
        try:
            font_properties = font_manager.FontProperties(
                family="sans-serif", weight="bold"
            )
            font_file = font_manager.findfont(font_properties)
            self._hud_font = ImageFont.truetype(font_file, 12 * self.SF)
        except Exception:
            logging.warning("Could not load system sans-serif font. Using default font.")
            try:
                self._hud_font = ImageFont.load_default(size=12 * self.SF)
            except Exception:
                self._hud_font = ImageFont.load_default()

        # Cache gradient background for faster rendering
        self._gradient_bg = self._create_gradient_backdrop()

        self.reset()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)

        if options is not None and "state" in options:
            state = options["state"]
            self.px = float(state["player_pos"][0])
            self.py = float(state["player_pos"][1])
            self.vx = float(state["player_vel"][0])
            self.vy = float(state["player_vel"][1])
            self.score = int(state["score"])
            self.coins = set(map(tuple, state["coins"]))
            self.on_ground = bool(state.get("on_ground", False))
            self.steps = int(state.get("steps", 0))
        else:
            self.px = 50.0
            self.py = 230.0
            self.vx = 0.0
            self.vy = 0.0
            self.score = 0
            self.coins = set(COINS_START)
            self.on_ground = False
            self.steps = 0

        obs = self._get_obs()
        return obs, {}

    def _create_gradient_backdrop(self) -> Image.Image:
        """Pre-draw sunset gradient backdrop at 3x scale."""
        w = VIEW_WIDTH * self.SF
        h = VIEW_HEIGHT * self.SF
        img = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(img)

        # Draw vertical sunset gradient (midnight blue to dark orange/pink)
        for y in range(h):
            t = y / h
            # Top: Midnight Blue (25, 25, 112)
            # Middle/Bottom: Orange/Pink (255, 110, 60)
            r = int(25 * (1 - t) + 255 * t)
            g = int(25 * (1 - t) + 110 * t)
            b = int(112 * (1 - t) + 60 * t)
            draw.line([(0, y), (w, y)], fill=(r, g, b))
        return img

    def _get_valid_mask(self) -> np.ndarray:
        """Return which actions are valid. Jump is valid only if on ground."""
        mask = np.ones((4,), dtype=np.int8)
        if not self.on_ground:
            mask[JUMP] = 0
        return mask

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Generate the observations dictionary."""
        progress = float(np.clip(self.px / FLAG_X, 0.0, 1.0))
        return {
            "observation": self._render_frame(),
            "valid_mask": self._get_valid_mask(),
            "level_progress": np.array([progress], dtype=np.float32),
        }

    def _check_solid_collision(self, x: float, y: float) -> Optional[Tuple[float, float, float, float]]:
        """Check if character at (x, y) collides with any solid block."""
        pw, ph = PLAYER_WIDTH, PLAYER_HEIGHT
        for bx1, by1, bx2, by2 in SOLID_BLOCKS:
            if x < bx2 and x + pw > bx1 and y < by2 and y + ph > by1:
                return (bx1, by1, bx2, by2)
        return None

    def _check_spike_collision(self, x: float, y: float) -> bool:
        """Check if character at (x, y) overlaps with any spikes."""
        pw, ph = PLAYER_WIDTH, PLAYER_HEIGHT
        for sx1, sy1, sx2, sy2 in SPIKES:
            if x < sx2 and x + pw > sx1 and y < sy2 and y + ph > sy1:
                return True
        return False

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Advance physical simulation by one step."""
        if not (0 <= action <= 3):
            raise ValueError(f"Invalid action: {action}")

        self.steps += 1
        reward = -0.05  # Step time penalty

        # Apply horizontal input
        if action == LEFT:
            self.vx = max(self.vx - ACCEL, -MAX_VX)
        elif action == RIGHT:
            self.vx = min(self.vx + ACCEL, MAX_VX)
        else:
            # Inertia drag
            self.vx *= FRICTION
            if abs(self.vx) < 0.1:
                self.vx = 0.0

        # Apply jump
        if action == JUMP and self.on_ground:
            self.vy = JUMP_SPEED
            self.on_ground = False

        # Apply gravity
        self.vy = min(self.vy + GRAVITY, 10.0)

        # Horizontal Collision Resolution
        self.px += self.vx
        block = self._check_solid_collision(self.px, self.py)
        if block is not None:
            bx1, by1, bx2, by2 = block
            if self.vx > 0:
                self.px = bx1 - PLAYER_WIDTH
            elif self.vx < 0:
                self.px = bx2
            self.vx = 0.0

        # Level bounds clamping (horizontal)
        if self.px < 0:
            self.px = 0.0
            self.vx = 0.0
        elif self.px > STAGE_WIDTH - PLAYER_WIDTH:
            self.px = STAGE_WIDTH - PLAYER_WIDTH
            self.vx = 0.0

        # Vertical Collision Resolution
        self.py += self.vy
        self.on_ground = False
        block = self._check_solid_collision(self.px, self.py)
        if block is not None:
            bx1, by1, bx2, by2 = block
            if self.vy > 0:
                self.py = by1 - PLAYER_HEIGHT
                self.on_ground = True
            elif self.vy < 0:
                self.py = by2
            self.vy = 0.0

        # Level bounds check (vertical)
        terminated = False
        if self.py > STAGE_HEIGHT:
            # Fell in pit, dies
            reward -= 50.0
            self.px = 50.0
            self.py = 230.0
            self.vx = 0.0
            self.vy = 0.0
            self.on_ground = False

        # Check Spikes Collision
        if self._check_spike_collision(self.px, self.py):
            reward -= 50.0
            # Reset position
            self.px = 50.0
            self.py = 230.0
            self.vx = 0.0
            self.vy = 0.0
            self.on_ground = False

        # Check Gold Coin Collection
        pacman_center_x = self.px + PLAYER_WIDTH / 2.0
        pacman_center_y = self.py + PLAYER_HEIGHT / 2.0
        collected_coins = []
        for cx, cy in self.coins:
            dist = math.sqrt((pacman_center_x - cx) ** 2 + (pacman_center_y - cy) ** 2)
            if dist < 24.0:  # Collection range
                collected_coins.append((cx, cy))
                self.score += 10
                reward += 10.0

        for coin in collected_coins:
            self.coins.remove(coin)

        # Check Win Condition
        if self.px >= FLAG_X:
            reward += 100.0
            terminated = True

        obs = self._get_obs()
        info = {
            "score": self.score,
            "coins_remaining": len(self.coins),
            "level_progress": self.px / FLAG_X,
        }

        return obs, float(reward), terminated, False, info

    def render(self) -> Optional[np.ndarray]:
        """Render method."""
        return self._render_frame()

    def _render_frame(self) -> np.ndarray:
        """Render screen frame with camera horizontal tracking and 3x SSAA Lanczos downsampling."""
        # Calculate horizontal camera tracking (view width 400)
        camera_x = (self.px + PLAYER_WIDTH / 2.0) - VIEW_WIDTH / 2.0
        camera_x = max(0.0, min(camera_x, STAGE_WIDTH - VIEW_WIDTH))

        # Start with sunset backdrop
        canvas = self._gradient_bg.copy()
        draw = ImageDraw.Draw(canvas)

        sf = self.SF

        # 1. Draw solid blocks (silhouette dark slate gray)
        block_color = (25, 25, 35)
        for bx1, by1, bx2, by2 in SOLID_BLOCKS:
            # Transform to camera space and scale
            x1 = (bx1 - camera_x) * sf
            y1 = by1 * sf
            x2 = (bx2 - camera_x) * sf
            y2 = by2 * sf
            draw.rectangle([x1, y1, x2, y2], fill=block_color)

        # 2. Draw spikes (silhouette dark slate gray with red tip indicator)
        for sx1, sy1, sx2, sy2 in SPIKES:
            x1_b = (sx1 - camera_x) * sf
            x2_b = (sx2 - camera_x) * sf
            y_base = sy2 * sf
            y_tip = sy1 * sf
            
            # Draw individual spikes
            spike_w = 10 * sf
            curr_x = x1_b
            while curr_x < x2_b:
                draw.polygon(
                    [
                        (curr_x, y_base),
                        (curr_x + spike_w / 2, y_tip),
                        (curr_x + spike_w, y_base),
                    ],
                    fill=(40, 10, 10),
                    outline=(200, 30, 30),
                )
                curr_x += spike_w

        # 3. Draw Coins
        for cx, cy in self.coins:
            x = (cx - camera_x) * sf
            y = cy * sf
            r = 5 * sf
            # Gold circle
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 215, 0), outline=(200, 150, 0), width=1 * sf)

        # 4. Draw Flag
        fx = (FLAG_X - camera_x) * sf
        fy = FLAG_Y * sf
        # Flagpole
        draw.line([(fx, fy), (fx, 260 * sf)], fill=(50, 50, 50), width=2 * sf)
        # Animated waving flag triangle
        wave_offset = math.sin(self.steps * 0.2) * 5 * sf
        draw.polygon(
            [
                (fx, fy),
                (fx + 20 * sf + wave_offset, fy + 7 * sf),
                (fx, fy + 14 * sf),
            ],
            fill=(230, 30, 30),
        )

        # 5. Draw Player Character
        px = (self.px - camera_x) * sf
        py = self.py * sf
        pw = PLAYER_WIDTH * sf
        ph = PLAYER_HEIGHT * sf

        # Animated runner sprite: draw a nice silhouette body
        # Head
        hr = 4 * sf
        draw.ellipse([px + pw/2 - hr, py + hr, px + pw/2 + hr, py + 3*hr], fill=(0, 240, 255))
        # Body
        draw.rounded_rectangle([px + 2*sf, py + 2.5*hr, px + pw - 2*sf, py + ph - 6*sf], radius=3*sf, fill=(0, 100, 230))
        
        # Legs: animated running swing
        leg_y = py + ph - 6*sf
        if self.vx != 0.0 and not self.on_ground:
            # Jump pose
            draw.line([(px + 4*sf, leg_y), (px - 1*sf, py + ph)], fill=(0, 240, 255), width=2*sf)
            draw.line([(px + pw - 4*sf, leg_y), (px + pw + 3*sf, py + ph - 2*sf)], fill=(0, 240, 255), width=2*sf)
        elif self.vx != 0.0:
            # Walk/run leg swing
            swing = math.sin(self.steps * 0.4) * 5 * sf
            draw.line([(px + 4*sf, leg_y), (px + 4*sf + swing, py + ph)], fill=(0, 240, 255), width=2*sf)
            draw.line([(px + pw - 4*sf, leg_y), (px + pw - 4*sf - swing, py + ph)], fill=(0, 240, 255), width=2*sf)
        else:
            # Standing legs
            draw.line([(px + 4*sf, leg_y), (px + 4*sf, py + ph)], fill=(0, 240, 255), width=2*sf)
            draw.line([(px + pw - 4*sf, leg_y), (px + pw - 4*sf, py + ph)], fill=(0, 240, 255), width=2*sf)

        # Eyes (glowing)
        draw.ellipse([px + pw/2 - 2*sf, py + 1.5*hr - 1*sf, px + pw/2, py + 1.5*hr + 1*sf], fill=(255, 255, 0))
        draw.ellipse([px + pw/2 + 1*sf, py + 1.5*hr - 1*sf, px + pw/2 + 3*sf, py + 1.5*hr + 1*sf], fill=(255, 255, 0))

        # 6. HUD (Score and progress bar)
        # Score top left
        draw.text((15 * sf, 15 * sf), f"SCORE: {self.score}", fill=(255, 255, 255), font=self._hud_font)

        # Progress bar at bottom
        bar_w = 200 * sf
        bar_h = 6 * sf
        bar_x = (VIEW_WIDTH / 2) * sf - bar_w / 2
        bar_y = 15 * sf
        # Background bar
        draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], fill=(80, 80, 80), outline=(255, 255, 255), width=1*sf)
        # Active progress fill
        prog_fill = min(self.px / FLAG_X, 1.0) * bar_w
        if prog_fill > 0:
            draw.rectangle([bar_x, bar_y, bar_x + prog_fill, bar_y + bar_h], fill=(0, 255, 100))

        # Downsample using high-quality LANCZOS anti-aliasing
        canvas_resized = canvas.resize((VIEW_WIDTH, VIEW_HEIGHT), Image.Resampling.LANCZOS)
        return np.array(canvas_resized, dtype=np.uint8)

    def close(self) -> None:
        """Close environment."""
        pass

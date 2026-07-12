"""A Gymnasium environment for two-player simultaneous Tank Combat in a grid maze."""

import copy
import math
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import numpy.typing as npt
import gymnasium as gym
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from absl import logging

# Game Constants
PLAY_WIDTH = 400
PLAY_HEIGHT = 400
CELL_SIZE = 40
GRID_ROWS = 10
GRID_COLS = 10

TANK_RADIUS = 12
BULLET_RADIUS = 3
MOVE_SPEED = 2.0
ROTATION_SPEED = 0.1  # radians per step
BULLET_SPEED = 5.0
MAX_HP = 3
SHOOT_COOLDOWN = 15

HEADER_PX = 50
FOOTER_PX = 30
CANVAS_SIZE = (PLAY_WIDTH, PLAY_HEIGHT + HEADER_PX + FOOTER_PX)

COLOR_BG = (15, 23, 42)
COLOR_WALL = (30, 41, 59)
COLOR_WALL_BORDER = (71, 85, 105)
COLOR_P1 = (14, 165, 233)           # Cyan/Blue
COLOR_P2 = (249, 115, 22)           # Orange
COLOR_BULLET = (250, 204, 21)       # Yellow

MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

class GymTankCombatEnv(gym.Env):
    """A Gymnasium environment for two-player simultaneous Tank Combat."""
    
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.SF = 3  # SSAA scale factor

        # Action space: MultiDiscrete([5, 5])
        # P1 action: [0..4], P2 action: [0..4]
        # 0: IDLE, 1: Rotate Left, 2: Rotate Right, 3: Move Forward, 4: Shoot
        self.action_space = spaces.MultiDiscrete([5, 5])

        # Obs space: Dict
        # "observation": Box shape (18,)
        # p1_x, p1_y, cos(p1_angle), sin(p1_angle), p1_hp
        # p2_x, p2_y, cos(p2_angle), sin(p2_angle), p2_hp
        # b1_x, b1_y, b1_vx, b1_vy
        # b2_x, b2_y, b2_vx, b2_vy
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=-2.0, high=2.0, shape=(18,), dtype=np.float32
            ),
            "total_score": spaces.Box(
                low=0, high=100, shape=(2,), dtype=np.int32
            )
        })

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

        # Build base canvas with maze drawn
        self._base_canvas = Image.new("RGB", (CANVAS_SIZE[0] * self.SF, CANVAS_SIZE[1] * self.SF), COLOR_BG)
        draw = ImageDraw.Draw(self._base_canvas)

        # Draw header / footer background
        draw.rectangle([0, 0, CANVAS_SIZE[0] * self.SF - 1, HEADER_PX * self.SF - 1], fill=(15, 23, 42))
        draw.rectangle(
            [0, (CANVAS_SIZE[1] - FOOTER_PX) * self.SF, CANVAS_SIZE[0] * self.SF - 1, CANVAS_SIZE[1] * self.SF - 1],
            fill=(10, 15, 28),
        )

        # Draw maze walls
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if MAZE[r][c] == 1:
                    x0 = c * CELL_SIZE * self.SF
                    y0 = (r * CELL_SIZE + HEADER_PX) * self.SF
                    x1 = (c + 1) * CELL_SIZE * self.SF
                    y1 = ((r + 1) * CELL_SIZE + HEADER_PX) * self.SF
                    draw.rectangle([x0, y0, x1, y1], fill=(30, 41, 59), outline=(71, 85, 105), width=1 * self.SF)

        self.reset()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)

        if options is not None and "state" in options:
            state = options["state"]
            self._p1_pos = np.array(state["p1_pos"], dtype=np.float32)
            self._p1_angle = float(state["p1_angle"])
            self._p1_hp = int(state["p1_hp"])
            self._p1_cooldown = int(state["p1_cooldown"])
            self._p2_pos = np.array(state["p2_pos"], dtype=np.float32)
            self._p2_angle = float(state["p2_angle"])
            self._p2_hp = int(state["p2_hp"])
            self._p2_cooldown = int(state["p2_cooldown"])
            self._bullets = copy.deepcopy(state["bullets"])
            self._scores = np.array(state["scores"], dtype=np.int32)
            self._steps = int(state["steps"])
            return self._create_observation(), {}

        # Default start positions in the maze
        self._p1_pos = np.array([60.0, 60.0], dtype=np.float32)
        self._p1_angle = 0.0
        self._p1_hp = MAX_HP
        self._p1_cooldown = 0

        self._p2_pos = np.array([340.0, 340.0], dtype=np.float32)
        self._p2_angle = math.pi
        self._p2_hp = MAX_HP
        self._p2_cooldown = 0

        self._bullets = []
        self._scores = np.zeros(2, dtype=np.int32)
        self._steps = 0

        return self._create_observation(), {}

    def _check_wall_collision(self, x: float, y: float, radius: float) -> bool:
        min_c = max(0, int((x - radius) // CELL_SIZE))
        max_c = min(GRID_COLS - 1, int((x + radius) // CELL_SIZE))
        min_r = max(0, int((y - radius) // CELL_SIZE))
        max_r = min(GRID_ROWS - 1, int((y + radius) // CELL_SIZE))
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if MAZE[r][c] == 1:
                    cx = max(c * CELL_SIZE, min(x, (c + 1) * CELL_SIZE))
                    cy = max(r * CELL_SIZE, min(y, (r + 1) * CELL_SIZE))
                    dist = math.hypot(x - cx, y - cy)
                    if dist < radius:
                        return True
        return False

    def _get_random_spawn_pos(self, other_x: Optional[float], other_y: Optional[float]) -> Tuple[float, float]:
        empty_cells = []
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if MAZE[r][c] == 0:
                    x = c * CELL_SIZE + CELL_SIZE / 2
                    y = r * CELL_SIZE + CELL_SIZE / 2
                    if other_x is None or math.hypot(x - other_x, y - other_y) > 100:
                        empty_cells.append((x, y))
        if not empty_cells:
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    if MAZE[r][c] == 0:
                        empty_cells.append((c * CELL_SIZE + CELL_SIZE/2, r * CELL_SIZE + CELL_SIZE/2))
        idx = self.np_random.choice(len(empty_cells))
        return empty_cells[idx]

    def _create_observation(self) -> Dict[str, Any]:
        obs = np.zeros(18, dtype=np.float32)
        obs[0] = self._p1_pos[0] / PLAY_WIDTH
        obs[1] = self._p1_pos[1] / PLAY_HEIGHT
        obs[2] = math.cos(self._p1_angle)
        obs[3] = math.sin(self._p1_angle)
        obs[4] = self._p1_hp / MAX_HP

        obs[5] = self._p2_pos[0] / PLAY_WIDTH
        obs[6] = self._p2_pos[1] / PLAY_HEIGHT
        obs[7] = math.cos(self._p2_angle)
        obs[8] = math.sin(self._p2_angle)
        obs[9] = self._p2_hp / MAX_HP

        # Find up to 2 active bullets
        for idx in range(2):
            if idx < len(self._bullets):
                b = self._bullets[idx]
                obs[10 + idx*4] = b["pos"][0] / PLAY_WIDTH
                obs[10 + idx*4 + 1] = b["pos"][1] / PLAY_HEIGHT
                obs[10 + idx*4 + 2] = b["vel"][0] / BULLET_SPEED
                obs[10 + idx*4 + 3] = b["vel"][1] / BULLET_SPEED
            else:
                obs[10 + idx*4 : 10 + idx*4 + 4] = -1.0

        return {
            "observation": obs,
            "total_score": self._scores.copy()
        }

    def _get_state(self) -> Dict[str, Any]:
        return {
            "p1_pos": list(self._p1_pos),
            "p1_angle": self._p1_angle,
            "p1_hp": self._p1_hp,
            "p1_cooldown": self._p1_cooldown,
            "p2_pos": list(self._p2_pos),
            "p2_angle": self._p2_angle,
            "p2_hp": self._p2_hp,
            "p2_cooldown": self._p2_cooldown,
            "bullets": copy.deepcopy(self._bullets),
            "scores": list(self._scores),
            "steps": self._steps,
        }

    def step(self, action: npt.NDArray[np.int32]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self._steps += 1
        p1_act = action[0]
        p2_act = action[1]

        # Handle Cooldowns
        if self._p1_cooldown > 0:
            self._p1_cooldown -= 1
        if self._p2_cooldown > 0:
            self._p2_cooldown -= 1

        # Player 1 Rotation
        if p1_act == 1:
            self._p1_angle -= ROTATION_SPEED
        elif p1_act == 2:
            self._p1_angle += ROTATION_SPEED
        self._p1_angle = self._p1_angle % (2 * math.pi)

        # Player 2 Rotation
        if p2_act == 1:
            self._p2_angle -= ROTATION_SPEED
        elif p2_act == 2:
            self._p2_angle += ROTATION_SPEED
        self._p2_angle = self._p2_angle % (2 * math.pi)

        # Move Forward with sliding
        if p1_act == 3:
            dx = math.cos(self._p1_angle) * MOVE_SPEED
            dy = math.sin(self._p1_angle) * MOVE_SPEED
            new_x = self._p1_pos[0] + dx
            if not self._check_wall_collision(new_x, self._p1_pos[1], TANK_RADIUS) and math.hypot(new_x - self._p2_pos[0], self._p1_pos[1] - self._p2_pos[1]) >= 2 * TANK_RADIUS:
                self._p1_pos[0] = new_x
            new_y = self._p1_pos[1] + dy
            if not self._check_wall_collision(self._p1_pos[0], new_y, TANK_RADIUS) and math.hypot(self._p1_pos[0] - self._p2_pos[0], new_y - self._p2_pos[1]) >= 2 * TANK_RADIUS:
                self._p1_pos[1] = new_y

        if p2_act == 3:
            dx = math.cos(self._p2_angle) * MOVE_SPEED
            dy = math.sin(self._p2_angle) * MOVE_SPEED
            new_x = self._p2_pos[0] + dx
            if not self._check_wall_collision(new_x, self._p2_pos[1], TANK_RADIUS) and math.hypot(new_x - self._p1_pos[0], self._p2_pos[1] - self._p1_pos[1]) >= 2 * TANK_RADIUS:
                self._p2_pos[0] = new_x
            new_y = self._p2_pos[1] + dy
            if not self._check_wall_collision(self._p2_pos[0], new_y, TANK_RADIUS) and math.hypot(self._p2_pos[0] - self._p1_pos[0], new_y - self._p1_pos[1]) >= 2 * TANK_RADIUS:
                self._p2_pos[1] = new_y

        # Clamp positions to keep inside boundaries
        self._p1_pos[0] = np.clip(self._p1_pos[0], TANK_RADIUS, PLAY_WIDTH - TANK_RADIUS)
        self._p1_pos[1] = np.clip(self._p1_pos[1], TANK_RADIUS, PLAY_HEIGHT - TANK_RADIUS)
        self._p2_pos[0] = np.clip(self._p2_pos[0], TANK_RADIUS, PLAY_WIDTH - TANK_RADIUS)
        self._p2_pos[1] = np.clip(self._p2_pos[1], TANK_RADIUS, PLAY_HEIGHT - TANK_RADIUS)

        # Shooting
        if p1_act == 4 and self._p1_cooldown == 0:
            bx = self._p1_pos[0] + math.cos(self._p1_angle) * (TANK_RADIUS + 3)
            by = self._p1_pos[1] + math.sin(self._p1_angle) * (TANK_RADIUS + 3)
            bvx = math.cos(self._p1_angle) * BULLET_SPEED
            bvy = math.sin(self._p1_angle) * BULLET_SPEED
            self._bullets.append({
                "pos": [bx, by],
                "vel": [bvx, bvy],
                "owner": 0,
                "bounces": 0
            })
            self._p1_cooldown = SHOOT_COOLDOWN

        if p2_act == 4 and self._p2_cooldown == 0:
            bx = self._p2_pos[0] + math.cos(self._p2_angle) * (TANK_RADIUS + 3)
            by = self._p2_pos[1] + math.sin(self._p2_angle) * (TANK_RADIUS + 3)
            bvx = math.cos(self._p2_angle) * BULLET_SPEED
            bvy = math.sin(self._p2_angle) * BULLET_SPEED
            self._bullets.append({
                "pos": [bx, by],
                "vel": [bvx, bvy],
                "owner": 1,
                "bounces": 0
            })
            self._p2_cooldown = SHOOT_COOLDOWN

        # Update bullets
        active_bullets = []
        reward = 0.0
        p1_respawn = False
        p2_respawn = False

        for bullet in self._bullets:
            bx, by = bullet["pos"]
            bvx, bvy = bullet["vel"]

            steps = 4
            dx = bvx / steps
            dy = bvy / steps
            destroyed = False
            for _ in range(steps):
                next_bx = bx + dx
                next_by = by + dy

                # Check boundaries
                if next_bx < 0 or next_bx >= PLAY_WIDTH or next_by < 0 or next_by >= PLAY_HEIGHT:
                    bullet["bounces"] += 1
                    if bullet["bounces"] > 2:
                        destroyed = True
                    else:
                        if next_bx < 0 or next_bx >= PLAY_WIDTH:
                            bvx = -bvx
                        if next_by < 0 or next_by >= PLAY_HEIGHT:
                            bvy = -bvy
                    break
                else:
                    r = int(next_by // CELL_SIZE)
                    c = int(next_bx // CELL_SIZE)
                    if MAZE[r][c] == 1:
                        bullet["bounces"] += 1
                        if bullet["bounces"] > 2:
                            destroyed = True
                        else:
                            old_c = int(bx // CELL_SIZE)
                            old_r = int(by // CELL_SIZE)
                            crossed_x = (old_c != c)
                            crossed_y = (old_r != r)
                            if crossed_x:
                                bvx = -bvx
                            if crossed_y:
                                bvy = -bvy
                            if not crossed_x and not crossed_y:
                                bvx = -bvx
                                bvy = -bvy
                        break
                bx = next_bx
                by = next_by

            if destroyed:
                continue

            bullet["pos"] = [bx, by]
            bullet["vel"] = [bvx, bvy]

            # Tank collisions
            dist_p1 = math.hypot(bx - self._p1_pos[0], by - self._p1_pos[1])
            dist_p2 = math.hypot(bx - self._p2_pos[0], by - self._p2_pos[1])

            if dist_p1 <= TANK_RADIUS:
                self._p1_hp -= 1
                reward -= 1.0
                destroyed = True
                if self._p1_hp <= 0:
                    self._scores[1] += 1
                    reward -= 5.0
                    p1_respawn = True
            elif dist_p2 <= TANK_RADIUS:
                self._p2_hp -= 1
                reward += 1.0
                destroyed = True
                if self._p2_hp <= 0:
                    self._scores[0] += 1
                    reward += 5.0
                    p2_respawn = True

            if not destroyed:
                active_bullets.append(bullet)

        self._bullets = active_bullets

        # Handle Respawns
        if p1_respawn:
            p1_pos = self._get_random_spawn_pos(self._p2_pos[0], self._p2_pos[1])
            self._p1_pos = np.array(p1_pos, dtype=np.float32)
            self._p1_hp = MAX_HP
            self._p1_angle = 0.0
            self._p1_cooldown = 0
            self._bullets = []
        if p2_respawn:
            p2_pos = self._get_random_spawn_pos(self._p1_pos[0], self._p1_pos[1])
            self._p2_pos = np.array(p2_pos, dtype=np.float32)
            self._p2_hp = MAX_HP
            self._p2_angle = math.pi
            self._p2_cooldown = 0
            self._bullets = []

        # Win condition (first to 5)
        terminated = False
        if self._scores[0] >= 5 or self._scores[1] >= 5:
            terminated = True
            if self._scores[0] >= 5:
                reward += 10.0
            else:
                reward -= 10.0

        truncated = self._steps >= 1000

        return self._create_observation(), float(reward), terminated, truncated, {"state": self._get_state()}

    def render(self) -> Optional[npt.NDArray[np.uint8]]:
        canvas = self._base_canvas.copy()
        draw = ImageDraw.Draw(canvas)

        # Draw P1 Tank
        p1_x, p1_y = self._p1_pos
        p1_y_v = p1_y + HEADER_PX
        p1_cx, p1_cy = p1_x * self.SF, p1_y_v * self.SF
        r = TANK_RADIUS * self.SF
        
        # P1 Barrel
        bx1 = p1_cx + math.cos(self._p1_angle) * r * 1.5
        by1 = p1_cy + math.sin(self._p1_angle) * r * 1.5
        draw.line([p1_cx, p1_cy, bx1, by1], fill=COLOR_P1, width=4 * self.SF)

        # P1 Body
        draw.ellipse([p1_cx - r, p1_cy - r, p1_cx + r, p1_cy + r], fill=(10, 110, 160), outline=(255, 255, 255), width=2 * self.SF)
        draw.ellipse([p1_cx - r * 0.6, p1_cy - r * 0.6, p1_cx + r * 0.6, p1_cy + r * 0.6], fill=COLOR_P1)

        # Draw P2 Tank
        p2_x, p2_y = self._p2_pos
        p2_y_v = p2_y + HEADER_PX
        p2_cx, p2_cy = p2_x * self.SF, p2_y_v * self.SF
        
        # P2 Barrel
        bx2 = p2_cx + math.cos(self._p2_angle) * r * 1.5
        by2 = p2_cy + math.sin(self._p2_angle) * r * 1.5
        draw.line([p2_cx, p2_cy, bx2, by2], fill=COLOR_P2, width=4 * self.SF)

        # P2 Body
        draw.ellipse([p2_cx - r, p2_cy - r, p2_cx + r, p2_cy + r], fill=(180, 80, 10), outline=(255, 255, 255), width=2 * self.SF)
        draw.ellipse([p2_cx - r * 0.6, p2_cy - r * 0.6, p2_cx + r * 0.6, p2_cy + r * 0.6], fill=COLOR_P2)

        # Draw Bullets
        for b in self._bullets:
            bx, by = b["pos"]
            by_v = by + HEADER_PX
            bcx, bcy = bx * self.SF, by_v * self.SF
            br = BULLET_RADIUS * self.SF
            draw.ellipse([bcx - br, bcy - br, bcx + br, bcy + br], fill=COLOR_BULLET, outline=(255, 255, 255), width=1 * self.SF)

        # Draw HUD Scores & HP
        # P1 HUD
        draw.text(
            (15 * self.SF, (HEADER_PX // 2) * self.SF),
            f"P1: {self._scores[0]}",
            fill=COLOR_P1,
            font=self._title_font,
            anchor="lm",
        )
        p1_hp_str = "♥" * self._p1_hp + "♡" * (MAX_HP - self._p1_hp)
        draw.text(
            (85 * self.SF, (HEADER_PX // 2) * self.SF),
            p1_hp_str,
            fill=(239, 68, 68),
            font=self._title_font,
            anchor="lm",
        )

        # P2 HUD
        draw.text(
            ((PLAY_WIDTH - 15) * self.SF, (HEADER_PX // 2) * self.SF),
            f"P2: {self._scores[1]}",
            fill=COLOR_P2,
            font=self._title_font,
            anchor="rm",
        )
        p2_hp_str = "♥" * self._p2_hp + "♡" * (MAX_HP - self._p2_hp)
        draw.text(
            ((PLAY_WIDTH - 85) * self.SF, (HEADER_PX // 2) * self.SF),
            p2_hp_str,
            fill=(239, 68, 68),
            font=self._title_font,
            anchor="rm",
        )

        # Footer info
        draw.text(
            (15 * self.SF, (CANVAS_SIZE[1] - FOOTER_PX // 2) * self.SF),
            f"Steps: {self._steps}",
            fill=(180, 180, 180),
            font=self._stats_font,
            anchor="lm",
        )

        # Downsample with LANCZOS for 3x SSAA
        canvas_resized = canvas.resize(CANVAS_SIZE, Image.Resampling.LANCZOS)
        return np.array(canvas_resized, dtype=np.uint8)

    def close(self) -> None:
        pass

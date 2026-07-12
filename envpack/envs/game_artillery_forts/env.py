"""A Gymnasium environment for two-player simultaneous real-time Artillery Forts."""

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
PLAY_WIDTH = 600
PLAY_HEIGHT = 350
HEADER_PX = 50
FOOTER_PX = 30
CANVAS_SIZE = (PLAY_WIDTH, PLAY_HEIGHT + HEADER_PX + FOOTER_PX)

FORT_WIDTH = 24
FORT_HEIGHT = 16
MAX_HP = 3
SHOOT_COOLDOWN = 30
MAX_POWER = 12.0
MIN_POWER = 2.0
GRAVITY = 0.15
AIR_RESISTANCE = 0.995
WIND_FORCE_SCALE = 0.02

COLOR_BG = (15, 23, 42)
COLOR_TERRAIN_FILL = (30, 58, 138)    # Deep blue
COLOR_TERRAIN_LINE = (59, 130, 246)   # Neon blue
COLOR_P1 = (14, 165, 233)            # Cyan
COLOR_P2 = (249, 115, 22)            # Orange
COLOR_SHELL = (250, 204, 21)         # Yellow

class GymArtilleryFortsEnv(gym.Env):
    """A Gymnasium environment for two-player simultaneous Artillery Forts."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.SF = 3  # SSAA scale factor

        # Action space: MultiDiscrete([6, 6])
        # 0: IDLE, 1: Aim Up, 2: Aim Down, 3: Power Up, 4: Power Down, 5: Fire
        self.action_space = spaces.MultiDiscrete([6, 6])

        # Obs space: Dict
        # "observation": Box shape (49,)
        # p1: fort_y, angle, power, hp, cooldown (5)
        # p2: fort_y, angle, power, hp, cooldown (5)
        # wind (1)
        # active shells: s1_x, s1_y, s1_vx, s1_vy, s2_x, s2_y, s2_vx, s2_vy (8)
        # sampled terrain heights at 30 points (30)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=-5.0, high=5.0, shape=(49,), dtype=np.float32
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

        # Build base canvas layout
        self._base_canvas = Image.new("RGB", (CANVAS_SIZE[0] * self.SF, CANVAS_SIZE[1] * self.SF), COLOR_BG)
        draw = ImageDraw.Draw(self._base_canvas)
        draw.rectangle([0, 0, CANVAS_SIZE[0] * self.SF - 1, HEADER_PX * self.SF - 1], fill=(15, 23, 42))
        draw.rectangle(
            [0, (CANVAS_SIZE[1] - FOOTER_PX) * self.SF, CANVAS_SIZE[0] * self.SF - 1, CANVAS_SIZE[1] * self.SF - 1],
            fill=(10, 15, 28),
        )

        self.reset()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)

        if options is not None and "state" in options:
            state = options["state"]
            self._terrain = np.array(state["terrain"], dtype=np.float32)
            self._p1_y = float(state["p1_y"])
            self._p1_angle = float(state["p1_angle"])
            self._p1_power = float(state["p1_power"])
            self._p1_hp = int(state["p1_hp"])
            self._p1_cooldown = int(state["p1_cooldown"])

            self._p2_y = float(state["p2_y"])
            self._p2_angle = float(state["p2_angle"])
            self._p2_power = float(state["p2_power"])
            self._p2_hp = int(state["p2_hp"])
            self._p2_cooldown = int(state["p2_cooldown"])

            self._wind_speed = float(state["wind_speed"])
            self._shells = copy.deepcopy(state["shells"])
            self._explosions = copy.deepcopy(state.get("explosions", []))
            self._scores = np.array(state["scores"], dtype=np.int32)
            self._steps = int(state["steps"])
            return self._create_observation(), {}

        # Procedural terrain generation using sines/cosines
        x_vals = np.arange(PLAY_WIDTH, dtype=np.float32)
        base = 230.0
        h1 = 40.0 * np.sin(x_vals * 0.005 + self.np_random.uniform(0, 2*math.pi))
        h2 = 15.0 * np.cos(x_vals * 0.015 + self.np_random.uniform(0, 2*math.pi))
        h3 = 8.0 * np.sin(x_vals * 0.04 + self.np_random.uniform(0, 2*math.pi))
        self._terrain = np.clip(base + h1 + h2 + h3, 120.0, PLAY_HEIGHT - 30.0)

        # Wind Speed
        self._wind_speed = self.np_random.uniform(-1.5, 1.5)

        # Fort position setups
        self._p1_y = self._terrain[80]
        self._p1_angle = -math.pi / 4  # 45 deg up-right
        self._p1_power = 6.0
        self._p1_hp = MAX_HP
        self._p1_cooldown = 0

        self._p2_y = self._terrain[520]
        self._p2_angle = -3 * math.pi / 4  # 45 deg up-left
        self._p2_power = 6.0
        self._p2_hp = MAX_HP
        self._p2_cooldown = 0

        self._shells = []
        self._explosions = []
        self._scores = np.zeros(2, dtype=np.int32)
        self._steps = 0

        return self._create_observation(), {}

    def _create_observation(self) -> Dict[str, Any]:
        obs = np.zeros(49, dtype=np.float32)
        # P1
        obs[0] = self._p1_y / PLAY_HEIGHT
        obs[1] = self._p1_angle / math.pi
        obs[2] = self._p1_power / MAX_POWER
        obs[3] = self._p1_hp / MAX_HP
        obs[4] = self._p1_cooldown / SHOOT_COOLDOWN

        # P2
        obs[5] = self._p2_y / PLAY_HEIGHT
        obs[6] = self._p2_angle / math.pi
        obs[7] = self._p2_power / MAX_POWER
        obs[8] = self._p2_hp / MAX_HP
        obs[9] = self._p2_cooldown / SHOOT_COOLDOWN

        # Wind
        obs[10] = self._wind_speed / 2.0

        # Shells (up to 2)
        for idx in range(2):
            if idx < len(self._shells):
                s = self._shells[idx]
                obs[11 + idx*4] = s["pos"][0] / PLAY_WIDTH
                obs[11 + idx*4 + 1] = s["pos"][1] / PLAY_HEIGHT
                obs[11 + idx*4 + 2] = s["vel"][0] / 15.0
                obs[11 + idx*4 + 3] = s["vel"][1] / 15.0
            else:
                obs[11 + idx*4 : 11 + idx*4 + 4] = -1.0

        # Sampled terrain (30 values)
        sampled_indices = np.linspace(0, PLAY_WIDTH - 1, 30, dtype=np.int32)
        for i, idx_val in enumerate(sampled_indices):
            obs[19 + i] = self._terrain[idx_val] / PLAY_HEIGHT

        return {
            "observation": obs,
            "total_score": self._scores.copy()
        }

    def _get_state(self) -> Dict[str, Any]:
        return {
            "terrain": list(self._terrain),
            "p1_y": self._p1_y,
            "p1_angle": self._p1_angle,
            "p1_power": self._p1_power,
            "p1_hp": self._p1_hp,
            "p1_cooldown": self._p1_cooldown,
            "p2_y": self._p2_y,
            "p2_angle": self._p2_angle,
            "p2_power": self._p2_power,
            "p2_hp": self._p2_hp,
            "p2_cooldown": self._p2_cooldown,
            "wind_speed": self._wind_speed,
            "shells": copy.deepcopy(self._shells),
            "explosions": copy.deepcopy(self._explosions),
            "scores": list(self._scores),
            "steps": self._steps,
        }

    def step(self, action: npt.NDArray[np.int32]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self._steps += 1
        p1_act = action[0]
        p2_act = action[1]

        # Cooldowns
        if self._p1_cooldown > 0:
            self._p1_cooldown -= 1
        if self._p2_cooldown > 0:
            self._p2_cooldown -= 1

        # P1 adjustments
        if p1_act == 1:  # Aim Up
            self._p1_angle = np.clip(self._p1_angle - 0.02, -math.pi/2, 0.0)
        elif p1_act == 2:  # Aim Down
            self._p1_angle = np.clip(self._p1_angle + 0.02, -math.pi/2, 0.0)

        if p1_act == 3:  # Power Up
            self._p1_power = np.clip(self._p1_power + 0.1, MIN_POWER, MAX_POWER)
        elif p1_act == 4:  # Power Down
            self._p1_power = np.clip(self._p1_power - 0.1, MIN_POWER, MAX_POWER)

        # P2 adjustments
        if p2_act == 1:  # Aim Up
            self._p2_angle = np.clip(self._p2_angle + 0.02, -math.pi, -math.pi/2)
        elif p2_act == 2:  # Aim Down
            self._p2_angle = np.clip(self._p2_angle - 0.02, -math.pi, -math.pi/2)

        if p2_act == 3:  # Power Up
            self._p2_power = np.clip(self._p2_power + 0.1, MIN_POWER, MAX_POWER)
        elif p2_act == 4:  # Power Down
            self._p2_power = np.clip(self._p2_power - 0.1, MIN_POWER, MAX_POWER)

        # Firing shells
        if p1_act == 5 and self._p1_cooldown == 0:
            sx = 80.0 + math.cos(self._p1_angle) * 15.0
            sy = self._p1_y - 8.0 + math.sin(self._p1_angle) * 15.0
            vx = math.cos(self._p1_angle) * self._p1_power
            vy = math.sin(self._p1_angle) * self._p1_power
            self._shells.append({
                "pos": [sx, sy],
                "vel": [vx, vy],
                "owner": 0
            })
            self._p1_cooldown = SHOOT_COOLDOWN

        if p2_act == 5 and self._p2_cooldown == 0:
            sx = 520.0 + math.cos(self._p2_angle) * 15.0
            sy = self._p2_y - 8.0 + math.sin(self._p2_angle) * 15.0
            vx = math.cos(self._p2_angle) * self._p2_power
            vy = math.sin(self._p2_angle) * self._p2_power
            self._shells.append({
                "pos": [sx, sy],
                "vel": [vx, vy],
                "owner": 1
            })
            self._p2_cooldown = SHOOT_COOLDOWN

        # Update Shells
        active_shells = []
        reward = 0.0
        p1_respawn = False
        p2_respawn = False

        for shell in self._shells:
            sx, sy = shell["pos"]
            vx, vy = shell["vel"]

            # Physics updates
            vx += self._wind_speed * WIND_FORCE_SCALE
            vx *= AIR_RESISTANCE
            vy *= AIR_RESISTANCE
            vy += GRAVITY

            sx += vx
            sy += vy

            # Check boundaries
            if sx < 0 or sx >= PLAY_WIDTH:
                # out of bounds horizontally, destroy
                continue

            # Check terrain collision
            terrain_y = self._terrain[int(sx)]
            impact = False

            # Box colliders for Forts
            # Fort 1 box: x in [80-12, 80+12], y in [p1_y-16, p1_y]
            # Fort 2 box: x in [520-12, 520+12], y in [p2_y-16, p2_y]
            if 80 - 12 <= sx <= 80 + 12 and self._p1_y - 16 <= sy <= self._p1_y:
                # Direct hit on Fort 1!
                self._p1_hp -= 1
                reward -= 1.0
                impact = True
                if self._p1_hp <= 0:
                    self._scores[1] += 1
                    reward -= 5.0
                    p1_respawn = True
            elif 520 - 12 <= sx <= 520 + 12 and self._p2_y - 16 <= sy <= self._p2_y:
                # Direct hit on Fort 2!
                self._p2_hp -= 1
                reward += 1.0
                impact = True
                if self._p2_hp <= 0:
                    self._scores[0] += 1
                    reward += 5.0
                    p2_respawn = True

            if not impact and sy >= terrain_y:
                # Terrain collision
                impact = True
                impact_x = int(sx)
                impact_y = sy

                # Crater terrain
                crater_radius = 20
                for tx in range(impact_x - crater_radius, impact_x + crater_radius + 1):
                    if 0 <= tx < PLAY_WIDTH:
                        dx = abs(tx - impact_x)
                        depth = math.sqrt(max(0.0, crater_radius**2 - dx**2))
                        self._terrain[tx] = max(self._terrain[tx], impact_y + depth)
                        self._terrain[tx] = min(self._terrain[tx], PLAY_HEIGHT - 10.0)

                # Check proximity splash damage to Forts
                dist_to_p1 = math.hypot(sx - 80.0, sy - self._p1_y)
                dist_to_p2 = math.hypot(sx - 520.0, sy - self._p2_y)

                if dist_to_p1 < 30.0:
                    self._p1_hp -= 1
                    reward -= 1.0
                    if self._p1_hp <= 0:
                        self._scores[1] += 1
                        reward -= 5.0
                        p1_respawn = True
                if dist_to_p2 < 30.0:
                    self._p2_hp -= 1
                    reward += 1.0
                    if self._p2_hp <= 0:
                        self._scores[0] += 1
                        reward += 5.0
                        p2_respawn = True

            if impact:
                # Spawn explosion visual
                self._explosions.append({
                    "pos": [sx, sy],
                    "timer": 6
                })
            else:
                shell["pos"] = [sx, sy]
                shell["vel"] = [vx, vy]
                active_shells.append(shell)

        self._shells = active_shells

        # Update Explosions
        active_explosions = []
        for exp in self._explosions:
            exp["timer"] -= 1
            if exp["timer"] > 0:
                active_explosions.append(exp)
        self._explosions = active_explosions

        # Update Fort Y positions to fall with terrain
        self._p1_y = self._terrain[80]
        self._p2_y = self._terrain[520]

        # Abyss death checks (fort falls below threshold)
        if self._p1_y >= PLAY_HEIGHT - 15.0:
            self._p1_hp = 0
            self._scores[1] += 1
            reward -= 5.0
            p1_respawn = True

        if self._p2_y >= PLAY_HEIGHT - 15.0:
            self._p2_hp = 0
            self._scores[0] += 1
            reward += 5.0
            p2_respawn = True

        # Handle Respawns/New Round resets
        if p1_respawn or p2_respawn:
            # Re-generate terrain for next round
            x_vals = np.arange(PLAY_WIDTH, dtype=np.float32)
            base = 230.0
            h1 = 40.0 * np.sin(x_vals * 0.005 + self.np_random.uniform(0, 2*math.pi))
            h2 = 15.0 * np.cos(x_vals * 0.015 + self.np_random.uniform(0, 2*math.pi))
            h3 = 8.0 * np.sin(x_vals * 0.04 + self.np_random.uniform(0, 2*math.pi))
            self._terrain = np.clip(base + h1 + h2 + h3, 120.0, PLAY_HEIGHT - 30.0)

            self._p1_y = self._terrain[80]
            self._p1_hp = MAX_HP
            self._p1_cooldown = 0
            self._p1_angle = -math.pi / 4

            self._p2_y = self._terrain[520]
            self._p2_hp = MAX_HP
            self._p2_cooldown = 0
            self._p2_angle = -3 * math.pi / 4

            self._wind_speed = self.np_random.uniform(-1.5, 1.5)
            self._shells = []
            self._explosions = []

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

        # Draw procedural terrain
        # We need a filled polygon: points along the terrain, then down to bottom right, to bottom left
        terrain_points = []
        for x in range(PLAY_WIDTH):
            terrain_points.append((x * self.SF, (self._terrain[x] + HEADER_PX) * self.SF))
        # Add corners
        terrain_points.append((PLAY_WIDTH * self.SF, (PLAY_HEIGHT + HEADER_PX) * self.SF))
        terrain_points.append((0, (PLAY_HEIGHT + HEADER_PX) * self.SF))

        # Fill terrain
        draw.polygon(terrain_points, fill=COLOR_TERRAIN_FILL)
        # Outline terrain line
        for x in range(PLAY_WIDTH - 1):
            x1, y1 = x * self.SF, (self._terrain[x] + HEADER_PX) * self.SF
            x2, y2 = (x + 1) * self.SF, (self._terrain[x + 1] + HEADER_PX) * self.SF
            draw.line([x1, y1, x2, y2], fill=COLOR_TERRAIN_LINE, width=2 * self.SF)

        # Helper to draw aiming guidelines
        def draw_aim_line(fx, fy, angle, power, color):
            # Draw a dotted curve showing starting path of shell
            sx = fx + math.cos(angle) * 15.0
            sy = fy - 8.0 + math.sin(angle) * 15.0
            vx, vy = math.cos(angle) * power, math.sin(angle) * power
            pts = []
            cur_x, cur_y = sx, sy
            for _ in range(8):
                vx += self._wind_speed * WIND_FORCE_SCALE
                vx *= AIR_RESISTANCE
                vy *= AIR_RESISTANCE
                vy += GRAVITY
                cur_x += vx
                cur_y += vy
                pts.append((cur_x * self.SF, (cur_y + HEADER_PX) * self.SF))
            for pt in pts:
                draw.ellipse([pt[0]-1*self.SF, pt[1]-1*self.SF, pt[0]+1*self.SF, pt[1]+1*self.SF], fill=color)

        # Draw Aim Lines
        draw_aim_line(80.0, self._p1_y, self._p1_angle, self._p1_power, (103, 232, 249))
        draw_aim_line(520.0, self._p2_y, self._p2_angle, self._p2_power, (253, 186, 116))

        # Draw Fort 1 (Player 1)
        p1_x, p1_y_v = 80.0 * self.SF, (self._p1_y + HEADER_PX) * self.SF
        fw, fh = FORT_WIDTH * self.SF, FORT_HEIGHT * self.SF
        # Body
        draw.rectangle([p1_x - fw//2, p1_y_v - fh, p1_x + fw//2, p1_y_v], fill=(10, 110, 160), outline=(255, 255, 255), width=1 * self.SF)
        # Turret
        bx1 = p1_x + math.cos(self._p1_angle) * 18.0 * self.SF
        by1 = p1_y_v - fh//2 + math.sin(self._p1_angle) * 18.0 * self.SF
        draw.line([p1_x, p1_y_v - fh//2, bx1, by1], fill=COLOR_P1, width=3 * self.SF)

        # Draw Fort 2 (Player 2)
        p2_x, p2_y_v = 520.0 * self.SF, (self._p2_y + HEADER_PX) * self.SF
        # Body
        draw.rectangle([p2_x - fw//2, p2_y_v - fh, p2_x + fw//2, p2_y_v], fill=(180, 80, 10), outline=(255, 255, 255), width=1 * self.SF)
        # Turret
        bx2 = p2_x + math.cos(self._p2_angle) * 18.0 * self.SF
        by2 = p2_y_v - fh//2 + math.sin(self._p2_angle) * 18.0 * self.SF
        draw.line([p2_x, p2_y_v - fh//2, bx2, by2], fill=COLOR_P2, width=3 * self.SF)

        # Draw Shells
        for s in self._shells:
            sx, sy = s["pos"]
            sy_v = sy + HEADER_PX
            draw.ellipse(
                [(sx - 3)*self.SF, (sy_v - 3)*self.SF, (sx + 3)*self.SF, (sy_v + 3)*self.SF],
                fill=COLOR_SHELL, outline=(255, 255, 255), width=1 * self.SF
            )

        # Draw Explosions
        for exp in self._explosions:
            ex, ey = exp["pos"]
            ey_v = ey + HEADER_PX
            timer = exp["timer"]
            rad = (7 - timer) * 4 * self.SF
            draw.ellipse(
                [ex*self.SF - rad, ey_v*self.SF - rad, ex*self.SF + rad, ey_v*self.SF + rad],
                fill=(249, 115, 22, 120), outline=(239, 68, 68), width=1 * self.SF
            )

        # Draw Wind HUD Indicator
        wind_text = f"Wind: {'←' if self._wind_speed < 0 else '→'} {abs(self._wind_speed):.2f}"
        draw.text(
            ((PLAY_WIDTH // 2) * self.SF, (HEADER_PX // 2) * self.SF),
            wind_text,
            fill=(250, 204, 21),
            font=self._title_font,
            anchor="mm",
        )

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

"""A Gymnasium environment for two-player continuous Gravity Duel with central gravity star."""

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
STAR_RADIUS = 20
SHIP_RADIUS = 10
MISSILE_RADIUS = 3

ROTATION_SPEED = 0.08  # rad per step
THRUST_ACCEL = 0.15
DRAG = 0.985
GRAVITY_CONSTANT = 5000.0
MISSILE_SPEED = 4.0
MAX_HP = 3
SHOOT_COOLDOWN = 20
MISSILE_LIFETIME = 150
MAX_VELOCITY = 10.0

HEADER_PX = 50
FOOTER_PX = 30
CANVAS_SIZE = (PLAY_WIDTH, PLAY_HEIGHT + HEADER_PX + FOOTER_PX)

COLOR_BG = (10, 10, 18)
COLOR_STAR_CORE = (255, 255, 220)
COLOR_STAR_GLOW1 = (251, 191, 36)    # Warm yellow
COLOR_STAR_GLOW2 = (249, 115, 22)    # Orange
COLOR_P1 = (14, 165, 233)            # Cyan/Blue
COLOR_P2 = (249, 115, 22)            # Orange
COLOR_MISSILE = (239, 68, 68)        # Red

class GymGravityDuelEnv(gym.Env):
    """A Gymnasium environment for two-player simultaneous Gravity Duel."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.SF = 3  # SSAA scale factor

        # Action space: MultiDiscrete([5, 5])
        # 0: IDLE, 1: Rotate Left, 2: Rotate Right, 3: Thrust, 4: Fire Missile
        self.action_space = spaces.MultiDiscrete([5, 5])

        # Obs space: Dict
        # "observation": Box shape (24,)
        # p1_x, p1_y, p1_vx, p1_vy, cos(p1_angle), sin(p1_angle), p1_hp
        # p2_x, p2_y, p2_vx, p2_vy, cos(p2_angle), sin(p2_angle), p2_hp
        # star_x, star_y
        # m1_x, m1_y, m1_vx, m1_vy
        # m2_x, m2_y, m2_vx, m2_vy
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=-5.0, high=5.0, shape=(24,), dtype=np.float32
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

        # Generate base starfield canvas
        self._base_canvas = Image.new("RGB", (CANVAS_SIZE[0] * self.SF, CANVAS_SIZE[1] * self.SF), COLOR_BG)
        draw = ImageDraw.Draw(self._base_canvas)

        # Draw header / footer background
        draw.rectangle([0, 0, CANVAS_SIZE[0] * self.SF - 1, HEADER_PX * self.SF - 1], fill=(15, 23, 42))
        draw.rectangle(
            [0, (CANVAS_SIZE[1] - FOOTER_PX) * self.SF, CANVAS_SIZE[0] * self.SF - 1, CANVAS_SIZE[1] * self.SF - 1],
            fill=(10, 15, 28),
        )

        # Draw random static stars
        np.random.seed(0)
        for _ in range(50):
            sx = np.random.randint(0, PLAY_WIDTH * self.SF)
            sy = np.random.randint(HEADER_PX * self.SF, (PLAY_HEIGHT + HEADER_PX) * self.SF)
            brightness = np.random.randint(100, 255)
            draw.point((sx, sy), fill=(brightness, brightness, brightness))

        self.reset()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)

        if options is not None and "state" in options:
            state = options["state"]
            self._p1_pos = np.array(state["p1_pos"], dtype=np.float32)
            self._p1_vel = np.array(state["p1_vel"], dtype=np.float32)
            self._p1_angle = float(state["p1_angle"])
            self._p1_hp = int(state["p1_hp"])
            self._p1_cooldown = int(state["p1_cooldown"])
            self._p1_thrust_active = bool(state.get("p1_thrust_active", False))

            self._p2_pos = np.array(state["p2_pos"], dtype=np.float32)
            self._p2_vel = np.array(state["p2_vel"], dtype=np.float32)
            self._p2_angle = float(state["p2_angle"])
            self._p2_hp = int(state["p2_hp"])
            self._p2_cooldown = int(state["p2_cooldown"])
            self._p2_thrust_active = bool(state.get("p2_thrust_active", False))

            self._missiles = copy.deepcopy(state["missiles"])
            self._scores = np.array(state["scores"], dtype=np.int32)
            self._steps = int(state["steps"])
            return self._create_observation(), {}

        # Reset Positions
        self._p1_pos = np.array([80.0, 200.0], dtype=np.float32)
        self._p1_vel = np.zeros(2, dtype=np.float32)
        self._p1_angle = 0.0
        self._p1_hp = MAX_HP
        self._p1_cooldown = 0
        self._p1_thrust_active = False

        self._p2_pos = np.array([320.0, 200.0], dtype=np.float32)
        self._p2_vel = np.zeros(2, dtype=np.float32)
        self._p2_angle = math.pi
        self._p2_hp = MAX_HP
        self._p2_cooldown = 0
        self._p2_thrust_active = False

        self._missiles = []
        self._scores = np.zeros(2, dtype=np.int32)
        self._steps = 0

        return self._create_observation(), {}

    def _apply_gravity(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        star_pos = np.array([PLAY_WIDTH / 2, PLAY_HEIGHT / 2], dtype=np.float32)
        d = star_pos - pos
        dist = np.linalg.norm(d)
        if dist < 5.0:
            dist = 5.0
        # Acceleration direction
        accel_dir = d / dist
        accel_mag = GRAVITY_CONSTANT / (dist ** 2)
        return vel + accel_dir * accel_mag

    def _create_observation(self) -> Dict[str, Any]:
        obs = np.zeros(24, dtype=np.float32)
        # P1
        obs[0] = self._p1_pos[0] / PLAY_WIDTH
        obs[1] = self._p1_pos[1] / PLAY_HEIGHT
        obs[2] = self._p1_vel[0] / MAX_VELOCITY
        obs[3] = self._p1_vel[1] / MAX_VELOCITY
        obs[4] = math.cos(self._p1_angle)
        obs[5] = math.sin(self._p1_angle)
        obs[6] = self._p1_hp / MAX_HP

        # P2
        obs[7] = self._p2_pos[0] / PLAY_WIDTH
        obs[8] = self._p2_pos[1] / PLAY_HEIGHT
        obs[9] = self._p2_vel[0] / MAX_VELOCITY
        obs[10] = self._p2_vel[1] / MAX_VELOCITY
        obs[11] = math.cos(self._p2_angle)
        obs[12] = math.sin(self._p2_angle)
        obs[13] = self._p2_hp / MAX_HP

        # Star
        obs[14] = 0.5
        obs[15] = 0.5

        # Active missiles
        for idx in range(2):
            if idx < len(self._missiles):
                m = self._missiles[idx]
                obs[16 + idx*4] = m["pos"][0] / PLAY_WIDTH
                obs[16 + idx*4 + 1] = m["pos"][1] / PLAY_HEIGHT
                obs[16 + idx*4 + 2] = m["vel"][0] / MAX_VELOCITY
                obs[16 + idx*4 + 3] = m["vel"][1] / MAX_VELOCITY
            else:
                obs[16 + idx*4 : 16 + idx*4 + 4] = -1.0

        return {
            "observation": obs,
            "total_score": self._scores.copy()
        }

    def _get_state(self) -> Dict[str, Any]:
        return {
            "p1_pos": list(self._p1_pos),
            "p1_vel": list(self._p1_vel),
            "p1_angle": self._p1_angle,
            "p1_hp": self._p1_hp,
            "p1_cooldown": self._p1_cooldown,
            "p1_thrust_active": self._p1_thrust_active,
            "p2_pos": list(self._p2_pos),
            "p2_vel": list(self._p2_vel),
            "p2_angle": self._p2_angle,
            "p2_hp": self._p2_hp,
            "p2_cooldown": self._p2_cooldown,
            "p2_thrust_active": self._p2_thrust_active,
            "missiles": copy.deepcopy(self._missiles),
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

        self._p1_thrust_active = False
        self._p2_thrust_active = False

        # P1 actions
        if p1_act == 1:
            self._p1_angle -= ROTATION_SPEED
        elif p1_act == 2:
            self._p1_angle += ROTATION_SPEED
        self._p1_angle = self._p1_angle % (2 * math.pi)

        if p1_act == 3:
            self._p1_thrust_active = True
            self._p1_vel[0] += math.cos(self._p1_angle) * THRUST_ACCEL
            self._p1_vel[1] += math.sin(self._p1_angle) * THRUST_ACCEL

        # P2 actions
        if p2_act == 1:
            self._p2_angle -= ROTATION_SPEED
        elif p2_act == 2:
            self._p2_angle += ROTATION_SPEED
        self._p2_angle = self._p2_angle % (2 * math.pi)

        if p2_act == 3:
            self._p2_thrust_active = True
            self._p2_vel[0] += math.cos(self._p2_angle) * THRUST_ACCEL
            self._p2_vel[1] += math.sin(self._p2_angle) * THRUST_ACCEL

        # Apply gravity to ships
        self._p1_vel = self._apply_gravity(self._p1_pos, self._p1_vel)
        self._p2_vel = self._apply_gravity(self._p2_pos, self._p2_vel)

        # Drag friction
        self._p1_vel *= DRAG
        self._p2_vel *= DRAG

        # Cap velocity
        for vel in [self._p1_vel, self._p2_vel]:
            speed = np.linalg.norm(vel)
            if speed > MAX_VELOCITY:
                vel[:] = (vel / speed) * MAX_VELOCITY

        # Update position with wrap-around
        self._p1_pos += self._p1_vel
        self._p1_pos[0] = self._p1_pos[0] % PLAY_WIDTH
        self._p1_pos[1] = self._p1_pos[1] % PLAY_HEIGHT

        self._p2_pos += self._p2_vel
        self._p2_pos[0] = self._p2_pos[0] % PLAY_WIDTH
        self._p2_pos[1] = self._p2_pos[1] % PLAY_HEIGHT

        # Firing Missiles
        if p1_act == 4 and self._p1_cooldown == 0:
            # Spawn at ship nose
            mx = self._p1_pos[0] + math.cos(self._p1_angle) * (SHIP_RADIUS + 2)
            my = self._p1_pos[1] + math.sin(self._p1_angle) * (SHIP_RADIUS + 2)
            mvx = self._p1_vel[0] + math.cos(self._p1_angle) * MISSILE_SPEED
            mvy = self._p1_vel[1] + math.sin(self._p1_angle) * MISSILE_SPEED
            self._missiles.append({
                "pos": [mx, my],
                "vel": [mvx, mvy],
                "owner": 0,
                "lifetime": MISSILE_LIFETIME,
                "trail": [[mx, my]]
            })
            self._p1_cooldown = SHOOT_COOLDOWN

        if p2_act == 4 and self._p2_cooldown == 0:
            mx = self._p2_pos[0] + math.cos(self._p2_angle) * (SHIP_RADIUS + 2)
            my = self._p2_pos[1] + math.sin(self._p2_angle) * (SHIP_RADIUS + 2)
            mvx = self._p2_vel[0] + math.cos(self._p2_angle) * MISSILE_SPEED
            mvy = self._p2_vel[1] + math.sin(self._p2_angle) * MISSILE_SPEED
            self._missiles.append({
                "pos": [mx, my],
                "vel": [mvx, mvy],
                "owner": 1,
                "lifetime": MISSILE_LIFETIME,
                "trail": [[mx, my]]
            })
            self._p2_cooldown = SHOOT_COOLDOWN

        # Update missiles
        active_missiles = []
        reward = 0.0
        p1_respawn = False
        p2_respawn = False

        star_center = np.array([PLAY_WIDTH / 2, PLAY_HEIGHT / 2], dtype=np.float32)

        for m in self._missiles:
            m_pos = np.array(m["pos"], dtype=np.float32)
            m_vel = np.array(m["vel"], dtype=np.float32)

            # Apply star gravity to missile
            m_vel = self._apply_gravity(m_pos, m_vel)
            m_pos += m_vel
            m_pos[0] = m_pos[0] % PLAY_WIDTH
            m_pos[1] = m_pos[1] % PLAY_HEIGHT

            # Update trail
            m["trail"].append(list(m_pos))
            if len(m["trail"]) > 5:
                m["trail"].pop(0)

            m["pos"] = list(m_pos)
            m["vel"] = list(m_vel)
            m["lifetime"] -= 1

            # Checks
            dist_to_star = np.linalg.norm(m_pos - star_center)
            # Destroyed if absorbed by gravity star or expired
            if dist_to_star <= STAR_RADIUS or m["lifetime"] <= 0:
                continue

            # Check ship hits
            dist_p1 = np.linalg.norm(m_pos - self._p1_pos)
            dist_p2 = np.linalg.norm(m_pos - self._p2_pos)

            hit_registered = False
            if dist_p1 <= SHIP_RADIUS + MISSILE_RADIUS:
                self._p1_hp -= 1
                reward -= 1.0
                hit_registered = True
                if self._p1_hp <= 0:
                    self._scores[1] += 1
                    reward -= 5.0
                    p1_respawn = True
            elif dist_p2 <= SHIP_RADIUS + MISSILE_RADIUS:
                self._p2_hp -= 1
                reward += 1.0
                hit_registered = True
                if self._p2_hp <= 0:
                    self._scores[0] += 1
                    reward += 5.0
                    p2_respawn = True

            if not hit_registered:
                active_missiles.append(m)

        self._missiles = active_missiles

        # Check Star Collision for Ships
        dist_p1_star = np.linalg.norm(self._p1_pos - star_center)
        dist_p2_star = np.linalg.norm(self._p2_pos - star_center)

        if dist_p1_star <= STAR_RADIUS + SHIP_RADIUS:
            self._p1_hp = 0
            self._scores[1] += 1
            reward -= 3.0
            p1_respawn = True

        if dist_p2_star <= STAR_RADIUS + SHIP_RADIUS:
            self._p2_hp = 0
            self._scores[0] += 1
            reward += 3.0
            p2_respawn = True

        # Handle Respawns
        if p1_respawn:
            self._p1_pos = np.array([80.0, 200.0], dtype=np.float32)
            self._p1_vel = np.zeros(2, dtype=np.float32)
            self._p1_hp = MAX_HP
            self._p1_angle = 0.0
            self._p1_cooldown = 0
        if p2_respawn:
            self._p2_pos = np.array([320.0, 200.0], dtype=np.float32)
            self._p2_vel = np.zeros(2, dtype=np.float32)
            self._p2_hp = MAX_HP
            self._p2_angle = math.pi
            self._p2_cooldown = 0

        # Win condition (first to 5 points)
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
        # Draw everything on canvas with 3x scale factor
        canvas = self._base_canvas.copy()
        draw = ImageDraw.Draw(canvas)

        # Draw Glowing Central Gravity Sun with concentric translucent layers
        star_center = np.array([PLAY_WIDTH / 2, PLAY_HEIGHT / 2 + HEADER_PX], dtype=np.float32)
        sc_x, sc_y = star_center * self.SF
        sr = STAR_RADIUS * self.SF

        # Outer orange glow
        draw.ellipse([sc_x - sr - 15*self.SF, sc_y - sr - 15*self.SF, sc_x + sr + 15*self.SF, sc_y + sr + 15*self.SF], fill=(234, 88, 12, 100))
        # Inner warm yellow glow
        draw.ellipse([sc_x - sr - 5*self.SF, sc_y - sr - 5*self.SF, sc_x + sr + 5*self.SF, sc_y + sr + 5*self.SF], fill=COLOR_STAR_GLOW1)
        # Core
        draw.ellipse([sc_x - sr, sc_y - sr, sc_x + sr, sc_y + sr], fill=COLOR_STAR_CORE)

        # Helper to draw a ship as a triangle
        def draw_ship(pos, angle, color, thrust_active):
            cx, cy = pos
            cy_v = cy + HEADER_PX
            cx_sf, cy_sf = cx * self.SF, cy_v * self.SF
            r = SHIP_RADIUS * self.SF

            # Triangle coordinates
            nose = (cx_sf + math.cos(angle) * r, cy_sf + math.sin(angle) * r)
            back_left = (cx_sf + math.cos(angle + 2.4) * r, cy_sf + math.sin(angle + 2.4) * r)
            back_right = (cx_sf + math.cos(angle - 2.4) * r, cy_sf + math.sin(angle - 2.4) * r)

            # Draw thrust flame if thrust is active
            if thrust_active:
                flame_tip = (cx_sf - math.cos(angle) * r * 1.8, cy_sf - math.sin(angle) * r * 1.8)
                flame_l = (cx_sf + math.cos(angle + 2.8) * r * 0.7, cy_sf + math.sin(angle + 2.8) * r * 0.7)
                flame_r = (cx_sf + math.cos(angle - 2.8) * r * 0.7, cy_sf + math.sin(angle - 2.8) * r * 0.7)
                draw.polygon([flame_l, flame_tip, flame_r], fill=(249, 115, 22))

            draw.polygon([nose, back_left, back_right], fill=color, outline=(255, 255, 255), width=1 * self.SF)

        # Draw P1
        draw_ship(self._p1_pos, self._p1_angle, COLOR_P1, self._p1_thrust_active)
        # Draw P2
        draw_ship(self._p2_pos, self._p2_angle, COLOR_P2, self._p2_thrust_active)

        # Draw Missiles with Trails
        for m in self._missiles:
            trail = m["trail"]
            # Draw trail line
            if len(trail) > 1:
                trail_pts = [(pt[0]*self.SF, (pt[1] + HEADER_PX)*self.SF) for pt in trail]
                draw.line(trail_pts, fill=(239, 68, 68, 150), width=1 * self.SF)

            mx, my = m["pos"]
            my_v = my + HEADER_PX
            draw.ellipse(
                [(mx - MISSILE_RADIUS)*self.SF, (my_v - MISSILE_RADIUS)*self.SF, (mx + MISSILE_RADIUS)*self.SF, (my_v + MISSILE_RADIUS)*self.SF],
                fill=COLOR_MISSILE, outline=(255, 255, 255), width=1 * self.SF
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

"""A Gymnasium environment for Asteroids."""

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
# 0: Idle, 1: Rotate Left, 2: Rotate Right, 3: Thrust, 4: Shoot
IDLE = 0
ROTATE_LEFT = 1
ROTATE_RIGHT = 2
THRUST = 3
SHOOT = 4

# Physical / Screen dimensions
WIDTH = 400
HEIGHT = 300

SHIP_RADIUS = 8.0
LASER_RADIUS = 2.0
GEM_RADIUS = 4.0

DRAG = 0.99
THRUST_FORCE = 0.15
ROTATION_SPEED = 0.1
LASER_SPEED = 8.0
LASER_LIFETIME = 40
GEM_LIFETIME = 150
COOLDOWN_MAX = 5


class GymAsteroidsEnv(gym.Env):
    """A Gymnasium environment for Asteroids."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.SF = 3  # SSAA scale factor

        # Action Space: 0: Idle, 1: Rotate Left, 2: Rotate Right, 3: Thrust, 4: Shoot
        self.action_space = spaces.Discrete(5)

        # Observation Space: Dict with observation (300, 400, 3), valid_mask, score, lives
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0, high=255, shape=(300, 400, 3), dtype=np.uint8
                ),
                "valid_mask": spaces.Box(
                    low=0, high=1, shape=(5,), dtype=np.int8
                ),
                "score": spaces.Box(
                    low=0, high=100000, shape=(1,), dtype=np.int32
                ),
                "lives": spaces.Box(
                    low=0, high=3, shape=(1,), dtype=np.int32
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

        # Distant background stars cached for rendering
        self._stars: List[Tuple[float, float, float]] = []

        self.reset()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Distant background stars (randomly generated once)
        if not self._stars:
            self._stars = [
                (
                    self.np_random.uniform(0, WIDTH),
                    self.np_random.uniform(0, HEIGHT),
                    self.np_random.uniform(1.0, 2.5),
                )
                for _ in range(30)
            ]

        if options is not None and "state" in options:
            state = options["state"]
            self.lives = int(state["lives"])
            self.score = int(state["score"])
            
            # Reconstruct ship
            self.ship_px = float(state["ship_pos"][0])
            self.ship_py = float(state["ship_pos"][1])
            self.ship_vx = float(state["ship_vel"][0])
            self.ship_vy = float(state["ship_vel"][1])
            self.ship_angle = float(state["ship_angle"])
            self.shoot_cooldown = int(state.get("shoot_cooldown", 0))
            
            # Reconstruct asteroids
            self.asteroids = []
            for ast in state["asteroids"]:
                self.asteroids.append({
                    "x": float(ast["x"]),
                    "y": float(ast["y"]),
                    "vx": float(ast["vx"]),
                    "vy": float(ast["vy"]),
                    "radius": float(ast["radius"]),
                    "rot": float(ast.get("rot", 0.0)),
                    "rot_speed": float(ast.get("rot_speed", 0.02)),
                    "multipliers": ast.get("multipliers", [1.0] * 8),
                })
                
            # Reconstruct gems
            self.gems = []
            for gem in state["gems"]:
                self.gems.append({
                    "x": float(gem["x"]),
                    "y": float(gem["y"]),
                    "vx": float(gem.get("vx", 0.0)),
                    "vy": float(gem.get("vy", 0.0)),
                    "lifetime": int(gem["lifetime"]),
                })
                
            # Reconstruct lasers
            self.lasers = []
            for las in state["lasers"]:
                self.lasers.append({
                    "x": float(las["x"]),
                    "y": float(las["y"]),
                    "vx": float(las["vx"]),
                    "vy": float(las["vy"]),
                    "lifetime": int(las["lifetime"]),
                })
                
            self.particles = []
            self.steps = state.get("steps", 0)
        else:
            self.lives = 3
            self.score = 0
            self.ship_px = 200.0
            self.ship_py = 150.0
            self.ship_vx = 0.0
            self.ship_vy = 0.0
            self.ship_angle = -math.pi / 2.0  # Face up
            self.shoot_cooldown = 0
            self.asteroids = []
            self.gems = []
            self.lasers = []
            self.particles = []
            self.steps = 0
            
            # Spawn 4 initial large asteroids far from the ship
            self._spawn_initial_asteroids()

        obs = self._get_obs()
        return obs, {}

    def _spawn_initial_asteroids(self) -> None:
        """Spawn large asteroids away from the center start point."""
        for _ in range(4):
            while True:
                ax = self.np_random.uniform(0, WIDTH)
                ay = self.np_random.uniform(0, HEIGHT)
                dist = math.sqrt((ax - 200.0) ** 2 + (ay - 150.0) ** 2)
                if dist > 80.0:
                    break
                    
            speed = self.np_random.uniform(0.5, 1.2)
            angle = self.np_random.uniform(0, 2 * math.pi)
            self._create_asteroid(ax, ay, speed * math.cos(angle), speed * math.sin(angle), 24.0)

    def _create_asteroid(self, x: float, y: float, vx: float, vy: float, radius: float) -> None:
        """Helper to append an asteroid with jagged polygon multipliers."""
        mults = [float(self.np_random.uniform(0.8, 1.2)) for _ in range(8)]
        rot_speed = float(self.np_random.uniform(-0.04, 0.04))
        self.asteroids.append({
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "radius": radius,
            "rot": self.np_random.uniform(0, 2 * math.pi),
            "rot_speed": rot_speed,
            "multipliers": mults,
        })

    def _spawn_explosion_particles(self, x: float, y: float, num_particles: int = 12) -> None:
        """Visual particle explosion effect."""
        for _ in range(num_particles):
            speed = self.np_random.uniform(1.0, 3.0)
            angle = self.np_random.uniform(0, 2 * math.pi)
            self.particles.append({
                "x": x,
                "y": y,
                "vx": speed * math.cos(angle),
                "vy": speed * math.sin(angle),
                "lifetime": int(self.np_random.integers(15, 30)),
                "color": (
                    int(self.np_random.integers(200, 255)),
                    int(self.np_random.integers(100, 200)),
                    50,
                ),
            })

    def _get_valid_mask(self) -> np.ndarray:
        """All Asteroids actions are always valid."""
        return np.ones((5,), dtype=np.int8)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Generate observation dict."""
        return {
            "observation": self._render_frame(),
            "valid_mask": self._get_valid_mask(),
            "score": np.array([self.score], dtype=np.int32),
            "lives": np.array([self.lives], dtype=np.int32),
        }

    def _push_asteroids_away(self) -> None:
        """Push any asteroid within 80px of center further away to prevent immediate spawn deaths."""
        for ast in self.asteroids:
            dx = ast["x"] - 200.0
            dy = ast["y"] - 150.0
            dist = math.sqrt(dx ** 2 + dy ** 2)
            if dist < 80.0:
                ang = math.atan2(dy, dx) if dist > 0 else self.np_random.uniform(0, 2 * math.pi)
                ast["x"] = 200.0 + 85.0 * math.cos(ang)
                ast["y"] = 150.0 + 85.0 * math.sin(ang)

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Advance physical simulation by one step."""
        if not (0 <= action <= 4):
            raise ValueError(f"Invalid action: {action}")

        self.steps += 1
        reward = 0.01  # Small survival incentive

        # 1. Update Cooldowns
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

        # 2. Resolve Action
        if action == ROTATE_LEFT:
            self.ship_angle -= ROTATION_SPEED
        elif action == ROTATE_RIGHT:
            self.ship_angle += ROTATION_SPEED
        elif action == THRUST:
            self.ship_vx += THRUST_FORCE * math.cos(self.ship_angle)
            self.ship_vy += THRUST_FORCE * math.sin(self.ship_angle)
        elif action == SHOOT:
            if self.shoot_cooldown <= 0:
                # Firing laser
                lx = self.ship_px + SHIP_RADIUS * math.cos(self.ship_angle)
                ly = self.ship_py + SHIP_RADIUS * math.sin(self.ship_angle)
                lvx = LASER_SPEED * math.cos(self.ship_angle)
                lvy = LASER_SPEED * math.sin(self.ship_angle)
                self.lasers.append({
                    "x": lx,
                    "y": ly,
                    "vx": lvx,
                    "vy": lvy,
                    "lifetime": LASER_LIFETIME,
                })
                self.shoot_cooldown = COOLDOWN_MAX

        # 3. Apply Space Physics
        # Ship movement & wrap-around
        self.ship_px = (self.ship_px + self.ship_vx) % WIDTH
        self.ship_py = (self.ship_py + self.ship_vy) % HEIGHT
        self.ship_vx *= DRAG
        self.ship_vy *= DRAG

        # Move Lasers
        active_lasers = []
        for las in self.lasers:
            las["x"] = (las["x"] + las["vx"]) % WIDTH
            las["y"] = (las["y"] + las["vy"]) % HEIGHT
            las["lifetime"] -= 1
            if las["lifetime"] > 0:
                active_lasers.append(las)
        self.lasers = active_lasers

        # Move Asteroids
        for ast in self.asteroids:
            ast["x"] = (ast["x"] + ast["vx"]) % WIDTH
            ast["y"] = (ast["y"] + ast["vy"]) % HEIGHT
            ast["rot"] = (ast["rot"] + ast["rot_speed"]) % (2 * math.pi)

        # Move Mineral Gems
        active_gems = []
        for gem in self.gems:
            gem["x"] = (gem["x"] + gem["vx"]) % WIDTH
            gem["y"] = (gem["y"] + gem["vy"]) % HEIGHT
            gem["lifetime"] -= 1
            if gem["lifetime"] > 0:
                active_gems.append(gem)
        self.gems = active_gems

        # Move Particles
        active_parts = []
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["lifetime"] -= 1
            if p["lifetime"] > 0:
                active_parts.append(p)
        self.particles = active_parts

        # 4. Resolve Combat (Lasers vs Asteroids)
        dead_asteroids = set()
        dead_lasers = set()
        new_asteroids = []
        
        for li, las in enumerate(self.lasers):
            for ai, ast in enumerate(self.asteroids):
                if ai in dead_asteroids or li in dead_lasers:
                    continue
                dist = math.sqrt((las["x"] - ast["x"]) ** 2 + (las["y"] - ast["y"]) ** 2)
                if dist <= (LASER_RADIUS + ast["radius"]):
                    # Hit!
                    dead_asteroids.add(ai)
                    dead_lasers.add(li)
                    
                    # Spawn gem
                    gem_vx = self.np_random.uniform(-0.4, 0.4)
                    gem_vy = self.np_random.uniform(-0.4, 0.4)
                    self.gems.append({
                        "x": ast["x"],
                        "y": ast["y"],
                        "vx": gem_vx,
                        "vy": gem_vy,
                        "lifetime": GEM_LIFETIME,
                    })
                    
                    # Spawn splits
                    parent_rad = ast["radius"]
                    if parent_rad == 24.0:
                        # Split to two medium (12.0)
                        self.score += 20
                        reward += 20.0
                        for _ in range(2):
                            angle = self.np_random.uniform(0, 2 * math.pi)
                            vx = ast["vx"] + 1.2 * math.cos(angle)
                            vy = ast["vy"] + 1.2 * math.sin(angle)
                            new_asteroids.append((ast["x"], ast["y"], vx, vy, 12.0))
                    elif parent_rad == 12.0:
                        # Split to two small (6.0)
                        self.score += 50
                        reward += 50.0
                        for _ in range(2):
                            angle = self.np_random.uniform(0, 2 * math.pi)
                            vx = ast["vx"] + 1.6 * math.cos(angle)
                            vy = ast["vy"] + 1.6 * math.sin(angle)
                            new_asteroids.append((ast["x"], ast["y"], vx, vy, 6.0))
                    else:
                        # Small asteroid disintegrates
                        self.score += 100
                        reward += 100.0

                    self._spawn_explosion_particles(ast["x"], ast["y"])

        # Filter active lasers and asteroids
        self.lasers = [l for i, l in enumerate(self.lasers) if i not in dead_lasers]
        self.asteroids = [ast for i, ast in enumerate(self.asteroids) if i not in dead_asteroids]
        
        # Append newly split asteroids
        for ax, ay, vx, vy, rad in new_asteroids:
            self._create_asteroid(ax, ay, vx, vy, rad)

        # 5. Resolve Gem Collection
        collected_gems = []
        for gi, gem in enumerate(self.gems):
            dist = math.sqrt((self.ship_px - gem["x"]) ** 2 + (self.ship_py - gem["y"]) ** 2)
            if dist <= (SHIP_RADIUS + GEM_RADIUS + 3.0):  # margin for easy collection
                collected_gems.append(gi)
                self.score += 50
                reward += 50.0
                
        self.gems = [gem for i, gem in enumerate(self.gems) if i not in collected_gems]

        # 6. Resolve Ship-Asteroid Collisions
        terminated = False
        collided = False
        for ast in self.asteroids:
            dist = math.sqrt((self.ship_px - ast["x"]) ** 2 + (self.ship_py - ast["y"]) ** 2)
            if dist <= (SHIP_RADIUS + ast["radius"]):
                collided = True
                break
                
        if collided:
            self.lives -= 1
            reward -= 50.0
            self._spawn_explosion_particles(self.ship_px, self.ship_py, num_particles=20)
            if self.lives <= 0:
                terminated = True
            else:
                # Reset ship to center
                self.ship_px = 200.0
                self.ship_py = 150.0
                self.ship_vx = 0.0
                self.ship_vy = 0.0
                self.ship_angle = -math.pi / 2.0
                self._push_asteroids_away()

        # 7. Check Level Complete (if all asteroids destroyed, spawn more!)
        if not self.asteroids:
            self._spawn_initial_asteroids()

        obs = self._get_obs()
        info = {
            "score": self.score,
            "lives": self.lives,
            "asteroids_remaining": len(self.asteroids),
        }

        return obs, float(reward), terminated, False, info

    def render(self) -> Optional[np.ndarray]:
        """Gym render method."""
        return self._render_frame()

    def _render_frame(self) -> np.ndarray:
        """Render frame at 3x scale and downsample using LANCZOS."""
        canvas_w = WIDTH * self.SF
        canvas_h = HEIGHT * self.SF
        canvas = Image.new("RGB", (canvas_w, canvas_h), (5, 5, 12))  # Deep space dark blue
        draw = ImageDraw.Draw(canvas)
        
        sf = self.SF

        # 1. Draw distant stars
        for sx, sy, s_rad in self._stars:
            x = sx * sf
            y = sy * sf
            r = s_rad * sf / 2.0
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(180, 180, 255))

        # 2. Draw Mineral Gems (green diamonds)
        for gem in self.gems:
            gx = gem["x"] * sf
            gy = gem["y"] * sf
            r = GEM_RADIUS * sf
            points = [
                (gx, gy - r),
                (gx + r, gy),
                (gx, gy + r),
                (gx - r, gy),
            ]
            # Pulsing effect slightly
            draw.polygon(points, fill=(50, 255, 50), outline=(200, 255, 200), width=1 * sf)

        # 3. Draw Explosion Particles
        for p in self.particles:
            px = p["x"] * sf
            py = p["y"] * sf
            r = sf
            draw.ellipse([px - r, py - r, px + r, py + r], fill=p["color"])

        # 4. Draw Lasers (red lines)
        for las in self.lasers:
            lx = las["x"] * sf
            ly = las["y"] * sf
            # Determine direction segment
            v_len = math.sqrt(las["vx"] ** 2 + las["vy"] ** 2)
            dx = (las["vx"] / v_len) * 5 * sf if v_len > 0 else 0
            dy = (las["vy"] / v_len) * 5 * sf if v_len > 0 else 0
            draw.line([(lx - dx, ly - dy), (lx, ly)], fill=(255, 60, 60), width=2 * sf)

        # 5. Draw Asteroids (rotating rock jagged shapes)
        ast_color = (139, 137, 137)
        for ast in self.asteroids:
            ax = ast["x"] * sf
            ay = ast["y"] * sf
            ar = ast["radius"] * sf
            
            # Generate jagged vertices based on rotation
            vertices = []
            num_pts = 8
            for i in range(num_pts):
                theta = ast["rot"] + i * (2 * math.pi / num_pts)
                dist = ar * ast["multipliers"][i]
                vx = ax + dist * math.cos(theta)
                vy = ay + dist * math.sin(theta)
                vertices.append((vx, vy))
                
            # Draw asteroid outline and fill with dark gray
            draw.polygon(vertices, fill=(35, 33, 33), outline=ast_color, width=2 * sf)

        # 6. Draw Player Ship (white triangular ship with thrust flames)
        spx = self.ship_px * sf
        spy = self.ship_py * sf
        sr = SHIP_RADIUS * sf
        sa = self.ship_angle

        # Vertices of the triangle facing direction `sa`
        tip = (spx + sr * math.cos(sa), spy + sr * math.sin(sa))
        back_l = (spx + sr * math.cos(sa + 2.4), spy + sr * math.sin(sa + 2.4))
        back_r = (spx + sr * math.cos(sa - 2.4), spy + sr * math.sin(sa - 2.4))

        # Check if thrust action was just used to draw a thrust flame
        # (For flame animation, check if self.steps is odd/even or if action was THRUST)
        # For simplicity, we can draw a flame if thrusting
        # We don't have direct access to the action inside render, but we can look at velocity or keep a flag.
        # Let's check if the ship's current speed is above 0.2 to animate engine flame
        speed = math.sqrt(self.ship_vx ** 2 + self.ship_vy ** 2)
        if speed > 0.2:
            flame_len = sr * 0.8
            flame_tip = (spx - (sr + flame_len) * math.cos(sa) + self.np_random.uniform(-2, 2) * sf,
                         spy - (sr + flame_len) * math.sin(sa) + self.np_random.uniform(-2, 2) * sf)
            # Flame triangle
            draw.polygon([back_l, back_r, flame_tip], fill=(255, 140, 0))

        # Draw ship body outline
        draw.polygon([tip, back_l, back_r], fill=(10, 10, 20), outline=(255, 255, 255), width=2 * sf)

        # 7. Draw HUD (Score, Lives)
        hud_y = 15 * sf
        draw.text((15 * sf, hud_y), f"SCORE: {self.score}", fill=(255, 255, 255), font=self._hud_font)
        draw.text((385 * sf, hud_y), f"LIVES: {self.lives}", fill=(255, 255, 255), font=self._hud_font, anchor="rt")

        # Downsample using high-quality LANCZOS anti-aliasing
        canvas_resized = canvas.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
        return np.array(canvas_resized, dtype=np.uint8)

    def close(self) -> None:
        """Close environment."""
        pass

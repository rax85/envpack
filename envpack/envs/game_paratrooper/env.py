"""A Gymnasium environment for a classic DOS-style Paratrooper game."""

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from absl import logging
from gymnasium import spaces
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Physical and Game Constants
WIDTH = 400
HEIGHT = 300
TURRET_X = 200.0
TURRET_Y = 270.0
TURRET_RADIUS = 15.0
BARREL_LENGTH = 24.0

ROTATION_SPEED = 0.05
BULLET_SPEED = 8.0
BOMB_SPEED = 3.0
PARA_INITIAL_FALL_SPEED = 3.0
PARA_SLOW_FALL_SPEED = 1.0
PARA_FAST_FALL_SPEED = 4.0

HELI_SPEED_MIN = 2.0
HELI_SPEED_MAX = 4.0
HELI_SPAWN_CHANCE = 0.02
HELI_DROP_CHANCE = 0.015
HELI_BOMB_CHANCE = 0.15
COOLDOWN_STEPS = 5
MAX_HELICOPTERS = 3


class GymParatrooperEnv(gym.Env):
    """A Gymnasium environment for the classic Paratrooper arcade game."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode

        # Scale Factor for supersampling anti-aliasing (SSAA)
        self.SF = 3

        # Font setup (scaled by SF for high-quality downsampling)
        try:
            font_properties = font_manager.FontProperties(
                family="sans-serif", weight="bold"
            )
            font_file = font_manager.findfont(font_properties)
            self._title_font = ImageFont.truetype(font_file, 12 * self.SF)
        except Exception:
            logging.warning("Could not load system sans-serif font. Using default font.")
            try:
                self._title_font = ImageFont.load_default(size=12 * self.SF)
            except Exception:
                self._title_font = ImageFont.load_default()

        # Spaces
        # Actions: 0: Turn Left, 1: Turn Right, 2: Shoot, 3: Stay
        self.action_space = spaces.Discrete(4)

        # Observations dictionary
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
                "landed_left": spaces.Box(
                    low=0, high=4, shape=(1,), dtype=np.int32
                ),
                "landed_right": spaces.Box(
                    low=0, high=4, shape=(1,), dtype=np.int32
                ),
            }
        )

        # Internal state
        self._steps = 0
        self._score = 0
        self._landed_left = 0
        self._landed_right = 0
        self._landed_positions_left: List[float] = []
        self._landed_positions_right: List[float] = []
        self._turret_angle = math.pi / 2.0  # Straight up

        self._helicopters: List[Dict[str, Any]] = []
        self._paratroopers: List[Dict[str, Any]] = []
        self._bullets: List[Dict[str, Any]] = []
        self._bombs: List[Dict[str, Any]] = []
        self._explosions: List[Dict[str, Any]] = []
        self._cooldown_timer = 0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to its initial state."""
        super().reset(seed=seed)

        self._steps = 0
        self._score = 0
        self._landed_left = 0
        self._landed_right = 0
        self._landed_positions_left = []
        self._landed_positions_right = []
        self._turret_angle = math.pi / 2.0

        self._helicopters = []
        self._paratroopers = []
        self._bullets = []
        self._bombs = []
        self._explosions = []
        self._cooldown_timer = 0

        # Support custom state injection
        if options is not None and "state" in options:
            state = options["state"]
            self._turret_angle = float(state.get("turret_angle", math.pi / 2.0))
            self._score = int(state.get("score", 0))
            self._landed_left = int(state.get("landed_left", 0))
            self._landed_right = int(state.get("landed_right", 0))

            self._landed_positions_left = list(state.get("landed_positions_left", []))
            if not self._landed_positions_left and self._landed_left > 0:
                self._landed_positions_left = [100.0 + i * 20.0 for i in range(self._landed_left)]

            self._landed_positions_right = list(state.get("landed_positions_right", []))
            if not self._landed_positions_right and self._landed_right > 0:
                self._landed_positions_right = [300.0 - i * 20.0 for i in range(self._landed_right)]

            # Helicopters: [{x, y, vx, direction, next_drop_time}]
            for h in state.get("helicopters", []):
                self._helicopters.append({
                    "x": float(h["x"]),
                    "y": float(h["y"]),
                    "vx": float(h["vx"]),
                    "direction": int(h.get("direction", 1 if h["vx"] > 0 else -1)),
                    "next_drop_time": int(h.get("next_drop_time", 50))
                })

            # Paratroopers: [{x, y, vy, fall_dist, parachute_state, open_parachute_dist}]
            for p in state.get("paratroopers", []):
                self._paratroopers.append({
                    "x": float(p["x"]),
                    "y": float(p["y"]),
                    "vy": float(p["vy"]),
                    "fall_dist": float(p.get("fall_dist", 0.0)),
                    "parachute_state": str(p.get("parachute_state", "closed")),
                    "open_parachute_dist": float(p.get("open_parachute_dist", 40.0))
                })

            # Bullets: [{x, y, vx, vy}]
            for b in state.get("bullets", []):
                self._bullets.append({
                    "x": float(b["x"]),
                    "y": float(b["y"]),
                    "vx": float(b["vx"]),
                    "vy": float(b["vy"])
                })

            # Bombs: [{x, y, vy}]
            for bm in state.get("bombs", []):
                self._bombs.append({
                    "x": float(bm["x"]),
                    "y": float(bm["y"]),
                    "vy": float(bm.get("vy", BOMB_SPEED))
                })
        else:
            # Spawn initial helicopter to start with action
            self._spawn_helicopter()

        obs = self._get_obs()
        info = {
            "score": self._score,
            "landed_left": self._landed_left,
            "landed_right": self._landed_right,
        }
        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Perform one step in the environment."""
        self._steps += 1
        step_reward = 0.0
        terminated = False

        # 1. Action execution
        if action == 0:  # Turn Left (counter-clockwise)
            self._turret_angle = min(0.9 * math.pi, self._turret_angle + ROTATION_SPEED)
        elif action == 1:  # Turn Right (clockwise)
            self._turret_angle = max(0.1 * math.pi, self._turret_angle - ROTATION_SPEED)
        elif action == 2:  # Shoot
            if self._cooldown_timer == 0:
                # Firing starting point: end of barrel
                bx = TURRET_X + BARREL_LENGTH * math.cos(self._turret_angle)
                by = TURRET_Y - BARREL_LENGTH * math.sin(self._turret_angle)
                vx = BULLET_SPEED * math.cos(self._turret_angle)
                vy = -BULLET_SPEED * math.sin(self._turret_angle)
                self._bullets.append({"x": bx, "y": by, "vx": vx, "vy": vy})
                self._cooldown_timer = COOLDOWN_STEPS

        # Decrement cooldown
        if self._cooldown_timer > 0:
            self._cooldown_timer -= 1

        # 2. Update entities & mechanics

        # Move bullets
        for b in self._bullets:
            b["x"] += b["vx"]
            b["y"] += b["vy"]

        # Move helicopters and handle drops
        for h in self._helicopters:
            h["x"] += h["vx"]
            h["next_drop_time"] -= 1
            if h["next_drop_time"] <= 0:
                # Drop only if on-screen
                if 40.0 <= h["x"] <= 360.0:
                    if self.np_random.uniform(0.0, 1.0) < HELI_BOMB_CHANCE:
                        # Drop bomb
                        self._bombs.append({
                            "x": h["x"],
                            "y": h["y"] + 10.0,
                            "vy": BOMB_SPEED
                        })
                    else:
                        # Drop paratrooper
                        self._paratroopers.append({
                            "x": h["x"],
                            "y": h["y"] + 10.0,
                            "vy": PARA_INITIAL_FALL_SPEED,
                            "fall_dist": 0.0,
                            "parachute_state": "closed",
                            "open_parachute_dist": float(self.np_random.uniform(30.0, 50.0))
                        })
                    h["next_drop_time"] = int(self.np_random.integers(60, 120))

        # Remove out-of-screen helicopters
        self._helicopters = [
            h for h in self._helicopters
            if -50.0 <= h["x"] <= 450.0
        ]

        # Spawn new helicopters
        if len(self._helicopters) < MAX_HELICOPTERS:
            if self.np_random.uniform(0.0, 1.0) < HELI_SPAWN_CHANCE:
                self._spawn_helicopter()

        # Update falling paratroopers
        paras_to_keep = []
        for p in self._paratroopers:
            # Check if we open parachute
            if p["parachute_state"] == "closed" and p["fall_dist"] >= p["open_parachute_dist"]:
                p["parachute_state"] = "open"
                p["vy"] = PARA_SLOW_FALL_SPEED

            p["y"] += p["vy"]
            p["fall_dist"] += p["vy"]

            # Ground landing collision
            if p["y"] >= 270.0:
                if p["parachute_state"] == "open":
                    # Safe landing
                    if p["x"] < 200.0:
                        self._landed_positions_left.append(p["x"])
                    else:
                        self._landed_positions_right.append(p["x"])

                    self._landed_left = len(self._landed_positions_left)
                    self._landed_right = len(self._landed_positions_right)

                    # Check limit of 4 landed paratroopers
                    if self._landed_left >= 4 or self._landed_right >= 4:
                        terminated = True
                        step_reward -= 50.0
                        self._spawn_explosion(TURRET_X, TURRET_Y, max_radius=40.0, lifetime=30)
                else:
                    # Splat/Crash
                    self._spawn_explosion(p["x"], 270.0, max_radius=8.0, lifetime=6)
            else:
                paras_to_keep.append(p)
        self._paratroopers = paras_to_keep

        # Update bombs
        bombs_to_keep = []
        for bm in self._bombs:
            bm["y"] += bm["vy"]
            if bm["y"] >= 270.0:
                # Check collision with turret (bottom-center 200, 270)
                if 185.0 <= bm["x"] <= 215.0:
                    terminated = True
                    step_reward -= 50.0
                    self._spawn_explosion(TURRET_X, TURRET_Y, max_radius=40.0, lifetime=30)
                else:
                    # Hits ground
                    self._spawn_explosion(bm["x"], 270.0, max_radius=12.0, lifetime=10)
            else:
                bombs_to_keep.append(bm)
        self._bombs = bombs_to_keep

        # Collision detection (Bullets vs Helis/Bombs/Paras)
        bullets_to_remove = set()
        helis_to_remove = set()
        paras_to_remove = set()
        bombs_to_remove = set()

        for b_idx, b in enumerate(self._bullets):
            # Check boundary removal
            if b["x"] < 0 or b["x"] > 400 or b["y"] < 0 or b["y"] > 300:
                bullets_to_remove.add(b_idx)
                continue

            hit_something = False

            # 1. Bullet vs Helicopter
            for h_idx, h in enumerate(self._helicopters):
                if h_idx in helis_to_remove:
                    continue
                # Bounding box width=36, height=14
                if abs(b["x"] - h["x"]) < 18.0 and abs(b["y"] - h["y"]) < 8.0:
                    helis_to_remove.add(h_idx)
                    bullets_to_remove.add(b_idx)
                    self._score += 10
                    step_reward += 10.0
                    self._spawn_explosion(h["x"], h["y"], max_radius=16.0, lifetime=12)
                    hit_something = True
                    break

            if hit_something:
                continue

            # 2. Bullet vs Bomb
            for bm_idx, bm in enumerate(self._bombs):
                if bm_idx in bombs_to_remove:
                    continue
                # Radius 6
                dist_sq = (b["x"] - bm["x"])**2 + (b["y"] - bm["y"])**2
                if dist_sq < 64.0:
                    bombs_to_remove.add(bm_idx)
                    bullets_to_remove.add(b_idx)
                    self._score += 15
                    step_reward += 15.0
                    self._spawn_explosion(bm["x"], bm["y"], max_radius=12.0, lifetime=10)
                    hit_something = True
                    break

            if hit_something:
                continue

            # 3. Bullet vs Paratrooper/Parachute
            for p_idx, p in enumerate(self._paratroopers):
                if p_idx in paras_to_remove:
                    continue

                # Check Parachute hit
                if p["parachute_state"] == "open":
                    # Canopy is above the paratrooper body: y - 12
                    if abs(b["x"] - p["x"]) < 10.0 and abs(b["y"] - (p["y"] - 12.0)) < 5.0:
                        p["parachute_state"] = "destroyed"
                        p["vy"] = PARA_FAST_FALL_SPEED
                        bullets_to_remove.add(b_idx)
                        self._spawn_explosion(p["x"], p["y"] - 12.0, max_radius=8.0, lifetime=6)
                        hit_something = True
                        break

                if hit_something:
                    continue

                # Check Body hit
                if abs(b["x"] - p["x"]) < 6.0 and abs(b["y"] - p["y"]) < 8.0:
                    paras_to_remove.add(p_idx)
                    bullets_to_remove.add(b_idx)
                    self._score += 5
                    step_reward += 5.0
                    self._spawn_explosion(p["x"], p["y"], max_radius=10.0, lifetime=8)
                    hit_something = True
                    break

        # Apply removals
        self._bullets = [b for idx, b in enumerate(self._bullets) if idx not in bullets_to_remove]
        self._helicopters = [h for idx, h in enumerate(self._helicopters) if idx not in helis_to_remove]
        self._paratroopers = [p for idx, p in enumerate(self._paratroopers) if idx not in paras_to_remove]
        self._bombs = [bm for idx, bm in enumerate(self._bombs) if idx not in bombs_to_remove]

        # Update explosions
        next_explosions = []
        for exp in self._explosions:
            exp["lifetime"] -= 1
            exp["radius"] += exp["grow_rate"]
            if exp["lifetime"] > 0:
                next_explosions.append(exp)
        self._explosions = next_explosions

        # 3. Check episode terminations/truncations
        truncated = self._steps >= 1000

        obs = self._get_obs()
        info = {
            "score": self._score,
            "landed_left": self._landed_left,
            "landed_right": self._landed_right,
        }

        return obs, step_reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """Render the environment at the current state."""
        return self._render_frame()

    def close(self) -> None:
        """Close the environment."""
        pass

    def _spawn_helicopter(self) -> None:
        """Spawn a new helicopter at a random height moving horizontally."""
        direction = self.np_random.choice([-1, 1])
        y = float(self.np_random.uniform(30, 90))
        speed = float(self.np_random.uniform(HELI_SPEED_MIN, HELI_SPEED_MAX))
        if direction == 1:
            x = -40.0
            vx = speed
        else:
            x = 440.0
            vx = -speed

        self._helicopters.append({
            "x": x,
            "y": y,
            "vx": vx,
            "direction": direction,
            "next_drop_time": int(self.np_random.integers(30, 80))
        })

    def _spawn_explosion(self, x: float, y: float, max_radius: float = 15.0, lifetime: int = 10) -> None:
        """Spawn a visual explosion effect."""
        self._explosions.append({
            "x": x,
            "y": y,
            "radius": 2.0,
            "max_radius": max_radius,
            "lifetime": lifetime,
            "grow_rate": (max_radius - 2.0) / lifetime,
        })

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Generate the observations dictionary."""
        return {
            "observation": self._render_frame(),
            "valid_mask": np.ones((4,), dtype=np.int8),
            "score": np.array([self._score], dtype=np.int32),
            "landed_left": np.array([self._landed_left], dtype=np.int32),
            "landed_right": np.array([self._landed_right], dtype=np.int32),
        }

    def _render_frame(self) -> np.ndarray:
        """Generate the RGB screen view using SSAA."""
        # Create base canvas at 3x scale
        canvas_w = WIDTH * self.SF
        canvas_h = HEIGHT * self.SF
        canvas = Image.new("RGB", (canvas_w, canvas_h), (135, 206, 250))  # Sky Blue
        draw = ImageDraw.Draw(canvas)

        # 1. Draw Sky (already sky blue background)

        # 2. Draw ground (green grass)
        draw.rectangle(
            [0, int(TURRET_Y * self.SF), canvas_w - 1, canvas_h - 1],
            fill=(34, 139, 34)  # Forest Green
        )

        # 3. Draw Landed Paratroopers standing on the ground
        for x in self._landed_positions_left + self._landed_positions_right:
            px = x * self.SF
            py = TURRET_Y * self.SF

            # Head (solid black circle)
            hr = 2.5 * self.SF
            draw.ellipse(
                [px - hr, py - 14 * self.SF - hr, px + hr, py - 14 * self.SF + hr],
                fill=(0, 0, 0)
            )
            # Torso
            draw.line(
                [(px, py - 14 * self.SF), (px, py - 6 * self.SF)],
                fill=(0, 0, 0),
                width=1 * self.SF
            )
            # Arms
            draw.line(
                [(px - 4 * self.SF, py - 11 * self.SF), (px + 4 * self.SF, py - 11 * self.SF)],
                fill=(0, 0, 0),
                width=1 * self.SF
            )
            # Legs
            draw.line([(px, py - 6 * self.SF), (px - 3 * self.SF, py)], fill=(0, 0, 0), width=1 * self.SF)
            draw.line([(px, py - 6 * self.SF), (px + 3 * self.SF, py)], fill=(0, 0, 0), width=1 * self.SF)

        # 4. Draw Turret Barrel
        tx = TURRET_X * self.SF
        ty = TURRET_Y * self.SF
        bx_end = (TURRET_X + BARREL_LENGTH * math.cos(self._turret_angle)) * self.SF
        by_end = (TURRET_Y - BARREL_LENGTH * math.sin(self._turret_angle)) * self.SF
        draw.line(
            [(tx, ty), (bx_end, by_end)],
            fill=(80, 80, 80),
            width=4 * self.SF
        )

        # 5. Draw Turret Base Dome
        # Drawn as chord to get a perfect dome sitting on the ground
        draw.chord(
            [
                (TURRET_X - TURRET_RADIUS) * self.SF,
                (TURRET_Y - TURRET_RADIUS) * self.SF,
                (TURRET_X + TURRET_RADIUS) * self.SF,
                (TURRET_Y + TURRET_RADIUS) * self.SF
            ],
            start=180,
            end=360,
            fill=(100, 100, 100),
            outline=(50, 50, 50),
            width=2 * self.SF
        )

        # 6. Draw Helicopters
        for h in self._helicopters:
            hx = h["x"] * self.SF
            hy = h["y"] * self.SF

            # Fuselage
            draw.ellipse(
                [hx - 12 * self.SF, hy - 6 * self.SF, hx + 12 * self.SF, hy + 6 * self.SF],
                fill=(120, 120, 120),
                outline=(60, 60, 60),
                width=1 * self.SF
            )

            # Cockpit and tail depending on direction
            if h["vx"] > 0:
                # Cockpit window (light blue)
                draw.chord(
                    [hx - 4 * self.SF, hy - 6 * self.SF, hx + 12 * self.SF, hy + 6 * self.SF],
                    start=270,
                    end=90,
                    fill=(135, 206, 250),
                    outline=(60, 60, 60),
                    width=1 * self.SF
                )
                # Tail
                draw.line(
                    [(hx - 12 * self.SF, hy), (hx - 24 * self.SF, hy)],
                    fill=(120, 120, 120),
                    width=2 * self.SF
                )
                # Tail rotor
                draw.line(
                    [(hx - 24 * self.SF, hy - 6 * self.SF), (hx - 24 * self.SF, hy + 6 * self.SF)],
                    fill=(50, 50, 50),
                    width=1 * self.SF
                )
            else:
                # Cockpit window
                draw.chord(
                    [hx - 12 * self.SF, hy - 6 * self.SF, hx + 4 * self.SF, hy + 6 * self.SF],
                    start=90,
                    end=270,
                    fill=(135, 206, 250),
                    outline=(60, 60, 60),
                    width=1 * self.SF
                )
                # Tail
                draw.line(
                    [(hx + 12 * self.SF, hy), (hx + 24 * self.SF, hy)],
                    fill=(120, 120, 120),
                    width=2 * self.SF
                )
                # Tail rotor
                draw.line(
                    [(hx + 24 * self.SF, hy - 6 * self.SF), (hx + 24 * self.SF, hy + 6 * self.SF)],
                    fill=(50, 50, 50),
                    width=1 * self.SF
                )

            # Main Rotor Shaft
            draw.line(
                [(hx, hy - 6 * self.SF), (hx, hy - 10 * self.SF)],
                fill=(60, 60, 60),
                width=1 * self.SF
            )

            # Animated Main Rotor (simulating spinning rotors)
            rotor_w = 18.0 * math.cos(self._steps * 0.8) * self.SF
            draw.line(
                [(hx - rotor_w, hy - 10 * self.SF), (hx + rotor_w, hy - 10 * self.SF)],
                fill=(50, 50, 50),
                width=1 * self.SF
            )

        # 7. Draw Falling Paratroopers
        for p in self._paratroopers:
            px = p["x"] * self.SF
            py = p["y"] * self.SF

            # Head
            hr = 2.5 * self.SF
            draw.ellipse(
                [px - hr, py - hr, px + hr, py + hr],
                fill=(0, 0, 0)
            )
            # Torso
            draw.line(
                [(px, py + 2 * self.SF), (px, py + 10 * self.SF)],
                fill=(0, 0, 0),
                width=1 * self.SF
            )
            # Arms (raised up in parachuting posture)
            draw.line([(px, py + 4 * self.SF), (px - 4 * self.SF, py)], fill=(0, 0, 0), width=1 * self.SF)
            draw.line([(px, py + 4 * self.SF), (px + 4 * self.SF, py)], fill=(0, 0, 0), width=1 * self.SF)
            # Legs
            draw.line([(px, py + 10 * self.SF), (px - 2 * self.SF, py + 15 * self.SF)], fill=(0, 0, 0), width=1 * self.SF)
            draw.line([(px, py + 10 * self.SF), (px + 2 * self.SF, py + 15 * self.SF)], fill=(0, 0, 0), width=1 * self.SF)

            # Parachute Canopy and Cords (if open)
            if p["parachute_state"] == "open":
                cy = py - 12 * self.SF
                # Red canopy dome
                draw.chord(
                    [px - 10 * self.SF, cy - 8 * self.SF, px + 10 * self.SF, cy + 2 * self.SF],
                    start=180,
                    end=360,
                    fill=(255, 60, 60),
                    outline=(0, 0, 0),
                    width=1 * self.SF
                )
                # Cords
                draw.line([(px - 10 * self.SF, cy), (px, py)], fill=(80, 80, 80), width=1 * self.SF)
                draw.line([(px + 10 * self.SF, cy), (px, py)], fill=(80, 80, 80), width=1 * self.SF)

        # 8. Draw Bullets
        for b in self._bullets:
            bx = b["x"] * self.SF
            by = b["y"] * self.SF
            br = 2 * self.SF
            draw.ellipse(
                [bx - br, by - br, bx + br, by + br],
                fill=(255, 255, 0)
            )

        # 9. Draw Bombs
        for bm in self._bombs:
            bx = bm["x"] * self.SF
            by = bm["y"] * self.SF
            br = 3 * self.SF
            # Bomb body
            draw.ellipse([bx - br, by - br, bx + br, by + br], fill=(40, 40, 40))
            # Bomb fuse/tip
            draw.ellipse([bx - 1 * self.SF, by - 3 * self.SF, bx + 1 * self.SF, by - 1 * self.SF], fill=(255, 0, 0))

        # 10. Draw Explosions
        for exp in self._explosions:
            ex = exp["x"] * self.SF
            ey = exp["y"] * self.SF
            er = exp["radius"] * self.SF
            # Expanding circle outer edge orange, inner edge yellow
            draw.ellipse(
                [ex - er, ey - er, ex + er, ey + er],
                fill=(255, 69, 0)
            )
            if er > 2.0 * self.SF:
                draw.ellipse(
                    [ex - er * 0.5, ey - er * 0.5, ex + er * 0.5, ey + er * 0.5],
                    fill=(255, 215, 0)
                )

        # 11. Draw HUD Stats
        # Score at (15, 15)
        draw.text(
            (15 * self.SF, 15 * self.SF),
            f"SCORE: {self._score:05d}",
            fill=(255, 255, 0),
            font=self._title_font
        )

        # Landed Left / Right count at (385, 15)
        hud_text = f"L: {self._landed_left}/4  R: {self._landed_right}/4"
        try:
            text_w = draw.textlength(hud_text, font=self._title_font)
        except AttributeError:
            text_w = len(hud_text) * 6 * self.SF

        draw.text(
            (385 * self.SF - text_w, 15 * self.SF),
            hud_text,
            fill=(255, 255, 255),
            font=self._title_font
        )

        # Downsample using high-quality LANCZOS anti-aliasing
        canvas_resized = canvas.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
        return np.array(canvas_resized, dtype=np.uint8)

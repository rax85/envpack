"""A Gymnasium environment for a Tower Defense game."""

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from absl import logging
from gymnasium import spaces
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Action Constants
# 0: Idle
# 1-6: Build/Upgrade Gun Tower at slots 0-5
# 7-12: Build/Upgrade Laser Tower at slots 0-5
IDLE = 0

# Grid / Path Constants
WAYPOINTS = [
    (0, 50),
    (320, 50),
    (320, 150),
    (80, 150),
    (80, 250),
    (400, 250),
]

SLOTS = [
    (100, 90),
    (200, 90),
    (280, 100),
    (120, 190),
    (200, 210),
    (280, 210),
]

# Tower properties
# Gun Tower (Type 1): builds for 50 gold, upgrades for 40 gold
GUN_TOWER_COST = 50
GUN_UPGRADE_COST = 40
GUN_PROPERTIES = {
    1: {"range": 80.0, "damage": 4.0, "cooldown": 12},
    2: {"range": 100.0, "damage": 8.0, "cooldown": 10},
    3: {"range": 120.0, "damage": 15.0, "cooldown": 8},
}

# Laser Tower (Type 2): builds for 80 gold, upgrades for 60 gold
LASER_TOWER_COST = 80
LASER_UPGRADE_COST = 60
LASER_PROPERTIES = {
    1: {"range": 70.0, "damage": 0.3},
    2: {"range": 90.0, "damage": 0.6},
    3: {"range": 110.0, "damage": 1.2},
}

WIDTH = 400
HEIGHT = 300


class GymTowerDefenseEnv(gym.Env):
    """A Gymnasium environment for a Tower Defense game."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.SF = 3  # SSAA scale factor

        # Action Space: 0: Idle, 1-6: Gun Tower (Build/Upgrade), 7-12: Laser Tower (Build/Upgrade)
        self.action_space = spaces.Discrete(13)

        # Observation Space: Dict with observation (300, 400, 3), valid_mask, health, gold
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0, high=255, shape=(300, 400, 3), dtype=np.uint8
                ),
                "valid_mask": spaces.Box(
                    low=0, high=1, shape=(13,), dtype=np.int8
                ),
                "health": spaces.Box(
                    low=0, high=20, shape=(1,), dtype=np.int32
                ),
                "gold": spaces.Box(
                    low=0, high=100000, shape=(1,), dtype=np.int32
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

        self.reset()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)

        if options is not None and "state" in options:
            state = options["state"]
            self.health = int(state["health"])
            self.gold = int(state["gold"])
            self.score = int(state["score"])
            self.wave = int(state["wave"])
            
            # Reconstruct towers
            self.towers = {}
            for t in state["towers"]:
                self.towers[t["slot"]] = {
                    "type": t["type"],
                    "level": t["level"],
                    "cooldown": t.get("cooldown", 0),
                    "target_id": t.get("target_id", None),
                }
                
            # Reconstruct enemies
            self.enemies = {}
            for e in state["enemies"]:
                self.enemies[e["id"]] = {
                    "id": e["id"],
                    "x": float(e["x"]),
                    "y": float(e["y"]),
                    "hp": float(e["hp"]),
                    "max_hp": float(e.get("max_hp", e["hp"])),
                    "speed": float(e["speed"]),
                    "target_idx": int(e["target_idx"]),
                    "gold_reward": int(e.get("gold_reward", 15)),
                }
                
            # Reconstruct bullets
            self.bullets = []
            for b in state["bullets"]:
                self.bullets.append({
                    "x": float(b["x"]),
                    "y": float(b["y"]),
                    "target_id": b["target_id"],
                    "damage": float(b["damage"]),
                })
                
            self.enemy_id_counter = state.get("enemy_id_counter", len(self.enemies) + 100)
            self.steps = state.get("steps", 0)
            self.wave_spawn_timer = state.get("wave_spawn_timer", 0)
            self.spawned_in_wave = state.get("spawned_in_wave", 0)
        else:
            self.health = 20
            self.gold = 100
            self.score = 0
            self.wave = 0
            self.towers = {}  # slot_idx -> dict
            self.enemies = {}  # id -> dict
            self.bullets = []
            self.enemy_id_counter = 0
            self.steps = 0
            self.wave_spawn_timer = 0
            self.spawned_in_wave = 0
            
            # Start spawning wave 1 immediately
            self._start_next_wave()

        obs = self._get_obs()
        return obs, {}

    def _start_next_wave(self) -> None:
        """Initialize parameters for the next enemy wave."""
        self.wave += 1
        self.spawned_in_wave = 0
        self.wave_spawn_timer = 0

    def _spawn_enemy(self) -> None:
        """Spawn a single enemy for the current wave."""
        self.enemy_id_counter += 1
        hp = 15.0 + 8.0 * self.wave
        speed = 1.0 + 0.08 * self.wave
        speed = min(speed, 2.5)
        gold_reward = 15 + 2 * self.wave
        
        self.enemies[self.enemy_id_counter] = {
            "id": self.enemy_id_counter,
            "x": float(WAYPOINTS[0][0]),
            "y": float(WAYPOINTS[0][1]),
            "hp": hp,
            "max_hp": hp,
            "speed": speed,
            "target_idx": 1,
            "gold_reward": gold_reward,
        }
        self.spawned_in_wave += 1

    def _get_valid_mask(self) -> np.ndarray:
        """Compute the valid actions mask based on current gold and tower states."""
        mask = np.zeros((13,), dtype=np.int8)
        mask[IDLE] = 1  # Idle always valid

        # Slots: 0 to 5
        for i in range(6):
            # Gun Tower Build/Upgrade (Action 1-6)
            if i not in self.towers:
                if self.gold >= GUN_TOWER_COST:
                    mask[1 + i] = 1
            else:
                t = self.towers[i]
                if t["type"] == 1 and t["level"] < 3 and self.gold >= GUN_UPGRADE_COST:
                    mask[1 + i] = 1

            # Laser Tower Build/Upgrade (Action 7-12)
            if i not in self.towers:
                if self.gold >= LASER_TOWER_COST:
                    mask[7 + i] = 1
            else:
                t = self.towers[i]
                if t["type"] == 2 and t["level"] < 3 and self.gold >= LASER_UPGRADE_COST:
                    mask[7 + i] = 1

        return mask

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Generate observation dict."""
        return {
            "observation": self._render_frame(),
            "valid_mask": self._get_valid_mask(),
            "health": np.array([self.health], dtype=np.int32),
            "gold": np.array([self.gold], dtype=np.int32),
        }

    def _get_target_for_tower(self, tx: float, ty: float, range_val: float) -> Optional[int]:
        """Find the enemy in range that is furthest along the path."""
        best_enemy_id = None
        best_progress = -1.0
        
        for eid, e in self.enemies.items():
            dist = math.sqrt((e["x"] - tx) ** 2 + (e["y"] - ty) ** 2)
            if dist <= range_val:
                # Progress is waypoint index + distance covered to next waypoint
                progress = e["target_idx"]
                if e["target_idx"] < len(WAYPOINTS):
                    prev_wp = WAYPOINTS[e["target_idx"] - 1]
                    next_wp = WAYPOINTS[e["target_idx"]]
                    total_dist = math.sqrt((next_wp[0] - prev_wp[0]) ** 2 + (next_wp[1] - prev_wp[1]) ** 2)
                    dist_to_next = math.sqrt((next_wp[0] - e["x"]) ** 2 + (next_wp[1] - e["y"]) ** 2)
                    progress += (total_dist - dist_to_next) / max(total_dist, 1.0)
                
                if progress > best_progress:
                    best_progress = progress
                    best_enemy_id = eid
                    
        return best_enemy_id

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Advance physical simulation by one step."""
        if not (0 <= action <= 12):
            raise ValueError(f"Invalid action: {action}")

        self.steps += 1
        reward = 0.0

        # 1. Resolve Actions
        valid_mask = self._get_valid_mask()
        if valid_mask[action] == 1:
            if 1 <= action <= 6:
                slot_idx = action - 1
                if slot_idx not in self.towers:
                    # Build Gun Tower
                    self.towers[slot_idx] = {"type": 1, "level": 1, "cooldown": 0, "target_id": None}
                    self.gold -= GUN_TOWER_COST
                else:
                    # Upgrade Gun Tower
                    self.towers[slot_idx]["level"] += 1
                    self.gold -= GUN_UPGRADE_COST
            elif 7 <= action <= 12:
                slot_idx = action - 7
                if slot_idx not in self.towers:
                    # Build Laser Tower
                    self.towers[slot_idx] = {"type": 2, "level": 1, "cooldown": 0, "target_id": None}
                    self.gold -= LASER_TOWER_COST
                else:
                    # Upgrade Laser Tower
                    self.towers[slot_idx]["level"] += 1
                    self.gold -= LASER_UPGRADE_COST

        # 2. Spawning enemy waves
        # 5 enemies per wave, spawning 20 steps apart
        total_enemies_to_spawn = 4 + self.wave
        if self.spawned_in_wave < total_enemies_to_spawn:
            if self.wave_spawn_timer <= 0:
                self._spawn_enemy()
                self.wave_spawn_timer = 20
            else:
                self.wave_spawn_timer -= 1
        else:
            # If all enemies of current wave are defeated, schedule next wave
            if not self.enemies:
                self._start_next_wave()

        # 3. Move Marching Enemies
        escaped_enemies = []
        for eid, e in list(self.enemies.items()):
            tx, ty = WAYPOINTS[e["target_idx"]]
            dx = tx - e["x"]
            dy = ty - e["y"]
            dist = math.sqrt(dx ** 2 + dy ** 2)
            
            if dist <= e["speed"]:
                e["x"] = float(tx)
                e["y"] = float(ty)
                e["target_idx"] += 1
                if e["target_idx"] >= len(WAYPOINTS):
                    escaped_enemies.append(eid)
            else:
                e["x"] += (dx / dist) * e["speed"]
                e["y"] += (dy / dist) * e["speed"]

        # Handle Escaped Enemies
        for eid in escaped_enemies:
            self.enemies.pop(eid)
            self.health -= 1
            reward -= 10.0

        # 4. Tower Shooting and Cooldowns
        for slot_idx, t in self.towers.items():
            sx, sy = SLOTS[slot_idx]
            
            if t["type"] == 1:
                # Gun Tower
                props = GUN_PROPERTIES[t["level"]]
                if t["cooldown"] > 0:
                    t["cooldown"] -= 1
                
                target_id = self._get_target_for_tower(sx, sy, props["range"])
                t["target_id"] = target_id
                
                if target_id is not None and t["cooldown"] <= 0:
                    # Shoot bullet projectile
                    self.bullets.append({
                        "x": float(sx),
                        "y": float(sy),
                        "target_id": target_id,
                        "damage": props["damage"],
                    })
                    t["cooldown"] = props["cooldown"]
            elif t["type"] == 2:
                # Laser Tower (continuous beam)
                props = LASER_PROPERTIES[t["level"]]
                target_id = self._get_target_for_tower(sx, sy, props["range"])
                t["target_id"] = target_id
                
                if target_id is not None:
                    # Apply continuous damage instantly
                    self.enemies[target_id]["hp"] -= props["damage"]

        # 5. Move bullets and apply projectile hits
        dead_enemies = set()
        active_bullets = []
        bullet_speed = 10.0
        
        for b in self.bullets:
            tid = b["target_id"]
            if tid not in self.enemies:
                # Target already dead, bullet vanishes
                continue
                
            e = self.enemies[tid]
            dx = e["x"] - b["x"]
            dy = e["y"] - b["y"]
            dist = math.sqrt(dx ** 2 + dy ** 2)
            
            if dist <= bullet_speed:
                # Bullet hit!
                e["hp"] -= b["damage"]
            else:
                b["x"] += (dx / dist) * bullet_speed
                b["y"] += (dy / dist) * bullet_speed
                active_bullets.append(b)
                
        self.bullets = active_bullets

        # 6. Check enemy deaths and grant rewards
        for eid, e in list(self.enemies.items()):
            if e["hp"] <= 0:
                dead_enemies.add(eid)
                self.gold += e["gold_reward"]
                self.score += 10
                reward += 15.0
                self.enemies.pop(eid)

        # Remove dead targets reference from towers and bullets
        for slot_idx, t in self.towers.items():
            if t["target_id"] in dead_enemies:
                t["target_id"] = None
        self.bullets = [b for b in self.bullets if b["target_id"] not in dead_enemies]

        # 7. Check Terminations
        terminated = False
        if self.health <= 0:
            terminated = True
            reward -= 50.0

        obs = self._get_obs()
        info = {
            "health": self.health,
            "gold": self.gold,
            "score": self.score,
            "wave": self.wave,
            "enemies_alive": len(self.enemies),
        }

        return obs, float(reward), terminated, False, info

    def render(self) -> Optional[np.ndarray]:
        """Render for Gym framework."""
        return self._render_frame()

    def _render_frame(self) -> np.ndarray:
        """Render the 1200x900 canvas and downsample to 400x300 using LANCZOS."""
        canvas_w = WIDTH * self.SF
        canvas_h = HEIGHT * self.SF
        # Deep space dark slate green background
        canvas = Image.new("RGB", (canvas_w, canvas_h), (15, 20, 25))
        draw = ImageDraw.Draw(canvas)
        
        sf = self.SF

        # 1. Draw S-shaped path (Neon gray road)
        path_color = (40, 45, 50)
        path_border = (80, 100, 120)
        road_width = 16 * sf
        
        # Draw borders first
        for i in range(len(WAYPOINTS) - 1):
            p1 = (WAYPOINTS[i][0] * sf, WAYPOINTS[i][1] * sf)
            p2 = (WAYPOINTS[i+1][0] * sf, WAYPOINTS[i+1][1] * sf)
            draw.line([p1, p2], fill=path_border, width=road_width + 4 * sf)

        # Draw inner path
        for i in range(len(WAYPOINTS) - 1):
            p1 = (WAYPOINTS[i][0] * sf, WAYPOINTS[i][1] * sf)
            p2 = (WAYPOINTS[i+1][0] * sf, WAYPOINTS[i+1][1] * sf)
            draw.line([p1, p2], fill=path_color, width=road_width)

        # 2. Draw placement slots
        for idx, (sx, sy) in enumerate(SLOTS):
            x = sx * sf
            y = sy * sf
            r = 14 * sf
            
            if idx not in self.towers:
                # Empty slot - dashed circle
                draw.ellipse([x - r, y - r, x + r, y + r], outline=(150, 150, 150), width=1 * sf)
                # Slot number
                draw.text((x, y), str(idx), fill=(150, 150, 150), font=self._hud_font, anchor="mm")
            else:
                t = self.towers[idx]
                # Base circle
                base_color = (180, 50, 50) if t["type"] == 1 else (50, 180, 180)
                draw.ellipse([x - r, y - r, x + r, y + r], fill=base_color, outline=(255, 255, 255), width=2 * sf)
                
                # Draw weapon details
                # Level indicators
                for lvl in range(t["level"]):
                    lr = (r - 3 * sf) - lvl * 3 * sf
                    draw.ellipse([x - lr, y - lr, x + lr, y + lr], outline=(255, 255, 0), width=1 * sf)

                if t["type"] == 1:
                    # Draw gun barrel pointing to target if target is alive
                    tid = t["target_id"]
                    angle = 0.0
                    if tid in self.enemies:
                        e = self.enemies[tid]
                        angle = math.atan2((e["y"] - sy), (e["x"] - sx))
                    # Draw barrel rectangle
                    bx_tip = x + 18 * sf * math.cos(angle)
                    by_tip = y + 18 * sf * math.sin(angle)
                    draw.line([(x, y), (bx_tip, by_tip)], fill=(20, 20, 20), width=4 * sf)

        # 3. Draw Laser beams
        for idx, t in self.towers.items():
            if t["type"] == 2 and t["target_id"] in self.enemies:
                e = self.enemies[t["target_id"]]
                tx, ty = e["x"] * sf, e["y"] * sf
                sx, sy = SLOTS[idx][0] * sf, SLOTS[idx][1] * sf
                # Draw laser beam line
                draw.line([(sx, sy), (tx, ty)], fill=(0, 255, 255), width=2 * sf)
                # Draw spark at target
                draw.ellipse([tx - 4 * sf, ty - 4 * sf, tx + 4 * sf, ty + 4 * sf], fill=(255, 255, 255))

        # 4. Draw Bullets
        for b in self.bullets:
            bx = b["x"] * sf
            by = b["y"] * sf
            br = 3 * sf
            draw.ellipse([bx - br, by - br, bx + br, by + br], fill=(255, 200, 0))

        # 5. Draw marching enemies with health bars
        for eid, e in self.enemies.items():
            ex = e["x"] * sf
            ey = e["y"] * sf
            er = 8 * sf
            
            # Draw Enemy body (purple circle)
            draw.ellipse([ex - er, ey - er, ex + er, ey + er], fill=(138, 43, 226), outline=(255, 0, 255), width=1 * sf)
            
            # Draw Health Bar
            bar_w = 16 * sf
            bar_h = 3 * sf
            bx = ex - bar_w / 2
            by = ey - er - 6 * sf
            
            # Red backdrop
            draw.rectangle([bx, by, bx + bar_w, by + bar_h], fill=(255, 0, 0))
            # Green foreground based on hp percent
            hp_percent = max(0.0, e["hp"] / e["max_hp"])
            if hp_percent > 0:
                draw.rectangle([bx, by, bx + bar_w * hp_percent, by + bar_h], fill=(0, 255, 0))

        # 6. Draw HUD Info
        hud_y = 15 * sf
        draw.text((15 * sf, hud_y), f"SCORE: {self.score}  GOLD: {self.gold}", fill=(255, 255, 255), font=self._hud_font)
        draw.text((385 * sf, hud_y), f"HEALTH: {self.health}/20  WAVE: {self.wave}", fill=(255, 255, 255), font=self._hud_font, anchor="rt")

        # Downsample using high-quality LANCZOS anti-aliasing
        canvas_resized = canvas.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
        return np.array(canvas_resized, dtype=np.uint8)

    def close(self) -> None:
        """Close environment."""
        pass

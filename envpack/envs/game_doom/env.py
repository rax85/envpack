"""Doom-like 3D raycasted first-person shooter Gymnasium environment."""

import copy
import math
from typing import Any, Dict, Optional, Tuple, List

import gymnasium as gym
from gymnasium import spaces
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import numba

# Map definition (16x16)
# 1: Wall, 0: Empty space
MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1],
    [1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# Action Space constants
TURN_LEFT = 0
TURN_RIGHT = 1
MOVE_FORWARD = 2
MOVE_BACKWARD = 3
SHOOT = 4

# Viewport parameters
W = 320
H = 240
VIEW_H = 200
HUD_H = 40
FOV = math.pi / 3.0  # 60 degrees

# JIT-compiled helper functions
@numba.jit(nopython=True)
def check_collision_jit(x: float, y: float, grid: np.ndarray, radius: float = 0.25) -> bool:
    """Check if a circle at (x, y) with radius intersects any wall."""
    if x < 0 or x >= grid.shape[1] or y < 0 or y >= grid.shape[0]:
        return True
    for dy in [-radius, radius]:
        for dx in [-radius, radius]:
            ny = y + dy
            nx = x + dx
            if ny < 0 or ny >= grid.shape[0] or nx < 0 or nx >= grid.shape[1]:
                return True
            if grid[int(ny), int(nx)] == 1:
                return True
    return False

@numba.jit(nopython=True)
def line_of_sight_jit(x1: float, y1: float, x2: float, y2: float, grid: np.ndarray) -> bool:
    """Trace line from (x1, y1) to (x2, y2) checking for wall collisions."""
    dx = x2 - x1
    dy = y2 - y1
    dist = np.sqrt(dx*dx + dy*dy)
    if dist < 0.01:
        return True
    steps = int(dist * 10)
    for i in range(steps + 1):
        t = i / steps
        cx = x1 + t * dx
        cy = y1 + t * dy
        if cx < 0 or cx >= grid.shape[1] or cy < 0 or cy >= grid.shape[0]:
            return False
        if grid[int(cy), int(cx)] == 1:
            return False
    return True

@numba.jit(nopython=True)
def draw_scene_jit(
    map_grid: np.ndarray,
    player_x: float,
    player_y: float,
    player_angle: float,
    fov: float,
    w: int,
    h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform raycasting for all screen columns and render walls, ceiling, and floor."""
    pixels = np.zeros((h, w, 3), dtype=np.uint8)
    z_buffer = np.zeros(w)
    
    # 1. Render Sky Gradient and Shaded Floor (Fog)
    # Ceiling is top half, Floor is bottom half
    for y in range(h):
        if y < h // 2:
            # Sky Gradient: hellish red/orange at horizon to dark purple/black at the top
            t = y / (h // 2 - 1) if h // 2 > 1 else 0.0
            # Horizon (t = 1.0): (220, 80, 20)
            # Top (t = 0.0): (15, 5, 25)
            r = int(15 + t * (220 - 15))
            g = int(5 + t * (80 - 5))
            b = int(25 + t * (20 - 25))
            for x in range(w):
                pixels[y, x, 0] = r
                pixels[y, x, 1] = g
                pixels[y, x, 2] = b
        else:
            # Floor distance-based shading (Fog)
            # Distance is proportional to 1.0 / (y - h//2)
            floor_dist = h / (2.0 * y - h + 0.0001)
            shading = max(0.0, 1.0 - floor_dist / 12.0)
            # Base floor: dark grey (45, 45, 50)
            fr = int(45 * shading)
            fg = int(45 * shading)
            fb = int(50 * shading)
            for x in range(w):
                pixels[y, x, 0] = fr
                pixels[y, x, 1] = fg
                pixels[y, x, 2] = fb
                
    # 2. Raycast Walls
    for x in range(w):
        ray_angle = player_angle + (x / (w - 1) - 0.5) * fov
        cos_a = np.cos(ray_angle)
        sin_a = np.sin(ray_angle)
        
        map_x = int(player_x)
        map_y = int(player_y)
        
        delta_dist_x = 1e30 if cos_a == 0 else np.abs(1.0 / cos_a)
        delta_dist_y = 1e30 if sin_a == 0 else np.abs(1.0 / sin_a)
        
        if cos_a < 0:
            step_x = -1
            side_dist_x = (player_x - map_x) * delta_dist_x
        else:
            step_x = 1
            side_dist_x = (map_x + 1.0 - player_x) * delta_dist_x
            
        if sin_a < 0:
            step_y = -1
            side_dist_y = (player_y - map_y) * delta_dist_y
        else:
            step_y = 1
            side_dist_y = (map_y + 1.0 - player_y) * delta_dist_y
            
        hit = 0
        side = 0
        max_steps = 50
        steps = 0
        while hit == 0 and steps < max_steps:
            if side_dist_x < side_dist_y:
                side_dist_x += delta_dist_x
                map_x += step_x
                side = 0
            else:
                side_dist_y += delta_dist_y
                map_y += step_y
                side = 1
            steps += 1
            if map_x < 0 or map_x >= map_grid.shape[1] or map_y < 0 or map_y >= map_grid.shape[0]:
                break
            if map_grid[map_y, map_x] == 1:
                hit = 1
                
        if hit == 1:
            if side == 0:
                perp_wall_dist = (map_x - player_x + (1.0 - step_x) / 2.0) / cos_a
                wall_hit_y = player_y + perp_wall_dist * sin_a
                wall_x = wall_hit_y - np.floor(wall_hit_y)
            else:
                perp_wall_dist = (map_y - player_y + (1.0 - step_y) / 2.0) / sin_a
                wall_hit_x = player_x + perp_wall_dist * cos_a
                wall_x = wall_hit_x - np.floor(wall_hit_x)
            perp_wall_dist *= np.cos(ray_angle - player_angle)
            perp_wall_dist = max(perp_wall_dist, 0.01)
        else:
            perp_wall_dist = 999.0
            wall_x = 0.0
            
        z_buffer[x] = perp_wall_dist
        
        if hit == 1 and perp_wall_dist < 999.0:
            line_height = int(h / perp_wall_dist)
            draw_start = -line_height // 2 + h // 2
            draw_end = line_height // 2 + h // 2
            draw_start_clamped = max(0, draw_start)
            draw_end_clamped = min(h - 1, draw_end)
            
            # Base Wall color
            if side == 0:
                base_r, base_g, base_b = 180, 70, 70
            else:
                base_r, base_g, base_b = 130, 50, 50
                
            # Wall grooves/texture (vertical grooves based on wall_x)
            # Make seams/vertical panel outlines darker
            is_groove_col = (wall_x < 0.05) or (wall_x > 0.95) or (np.abs(wall_x - 0.5) < 0.03)
            
            max_depth = 12.0
            shading = max(0.0, 1.0 - perp_wall_dist / max_depth)
            
            # Darken if it's a groove column
            if is_groove_col:
                shading *= 0.5
                
            for y in range(draw_start_clamped, draw_end_clamped + 1):
                # Horizontal grooves
                tex_y = (y - draw_start) / (draw_end - draw_start) if draw_end > draw_start else 0.0
                is_groove_row = (tex_y < 0.05) or (tex_y > 0.95) or (np.abs(tex_y - 0.5) < 0.03)
                
                final_shading = shading * 0.5 if is_groove_row else shading
                
                pixels[y, x, 0] = int(base_r * final_shading)
                pixels[y, x, 1] = int(base_g * final_shading)
                pixels[y, x, 2] = int(base_b * final_shading)
                
    return pixels, z_buffer

@numba.jit(nopython=True)
def draw_sprites_jit(
    pixels: np.ndarray,
    z_buffer: np.ndarray,
    sprite_types: np.ndarray,
    sprite_xs: np.ndarray,
    sprite_ys: np.ndarray,
    sprite_states: np.ndarray,
    player_x: float,
    player_y: float,
    player_angle: float,
    fov: float,
    w: int,
    h: int,
) -> None:
    """Project and render sprites (enemies, dead enemies, items) onto the viewport."""
    num_sprites = len(sprite_types)
    if num_sprites == 0:
        return
    
    # Calculate distances
    dists = np.zeros(num_sprites)
    for i in range(num_sprites):
        dx = sprite_xs[i] - player_x
        dy = sprite_ys[i] - player_y
        dists[i] = np.sqrt(dx*dx + dy*dy)
        
    # Sort sprites by distance: furthest first
    order = np.arange(num_sprites)
    for i in range(num_sprites - 1):
        max_idx = i
        for j in range(i + 1, num_sprites):
            if dists[order[j]] > dists[order[max_idx]]:
                max_idx = j
        tmp = order[i]
        order[i] = order[max_idx]
        order[max_idx] = tmp
        
    for idx in range(num_sprites):
        i = order[idx]
        stype = sprite_types[i]
        state = sprite_states[i]
        sx = sprite_xs[i]
        sy = sprite_ys[i]
        dist = dists[i]
        
        if dist < 0.2:
            continue
            
        dx = sx - player_x
        dy = sy - player_y
        sprite_angle = np.arctan2(dy, dx)
        rel_angle = sprite_angle - player_angle
        rel_angle = np.mod(rel_angle + np.pi, 2 * np.pi) - np.pi
        
        # Check FOV
        if np.abs(rel_angle) > fov / 2.0 + 0.5:
            continue
            
        perp_dist = dist * np.cos(rel_angle)
        if perp_dist <= 0.1:
            continue
            
        sprite_screen_x = int((rel_angle / fov + 0.5) * w)
        sprite_height = int(h / perp_dist)
        sprite_width = sprite_height
        
        # Dead enemy splats are drawn flat on the ground
        if state == 0 and stype == 0:
            draw_start_y = h // 2 + int(sprite_height * 0.15)
            draw_end_y = h // 2 + int(sprite_height * 0.5)
        else:
            draw_start_y = -sprite_height // 2 + h // 2
            draw_end_y = sprite_height // 2 + h // 2
            
        draw_start_y = max(0, draw_start_y)
        draw_end_y = min(h - 1, draw_end_y)
        
        draw_start_x = sprite_screen_x - sprite_width // 2
        draw_end_x = sprite_screen_x + sprite_width // 2
        draw_start_x = max(0, draw_start_x)
        draw_end_x = min(w - 1, draw_end_x)
        
        max_depth = 12.0
        shading = max(0.1, 1.0 - perp_dist / max_depth)
        
        for col in range(draw_start_x, draw_end_x + 1):
            if perp_dist >= z_buffer[col]:
                continue
                
            tex_x = (col - (sprite_screen_x - sprite_width // 2)) / sprite_width if sprite_width > 0 else 0.0
            
            for row in range(draw_start_y, draw_end_y + 1):
                tex_y = (row - (-sprite_height // 2 + h // 2)) / sprite_height if sprite_height > 0 else 0.0
                r, g, b = 0, 0, 0
                draw_pixel = False
                
                # 1. Floor shadow (drawn under active alive enemies)
                if state == 1 and stype == 0:
                    cx = (tex_x - 0.5) * 2.0
                    cy = (tex_y - 1.0) * 4.0
                    if cx*cx + cy*cy < 0.4:
                        r, g, b = 15, 15, 15
                        draw_pixel = True
                        
                # 2. Main sprite drawing
                if not draw_pixel:
                    if stype == 0:  # Enemy (Detailed Demon with horns, claws)
                        if state == 1:  # Alive
                            # Horns
                            if tex_y < 0.25 and np.abs(tex_x - 0.5) > 0.2 and np.abs(tex_x - 0.5) < 0.4:
                                r, g, b = 30, 30, 30
                                draw_pixel = True
                            # Claws
                            elif tex_y > 0.75 and np.abs(tex_x - 0.5) > 0.25 and np.abs(tex_x - 0.5) < 0.45:
                                r, g, b = 255, 255, 255
                                draw_pixel = True
                            # Main body circle
                            elif (tex_x - 0.5)**2 + (tex_y - 0.55)**2 < 0.16:
                                # Glowing eyes
                                if 0.35 < tex_y < 0.45 and (0.28 < tex_x < 0.38 or 0.62 < tex_x < 0.72):
                                    r, g, b = 50, 255, 50
                                # Fangs
                                elif 0.6 < tex_y < 0.7 and np.abs(tex_x - 0.5) < 0.15:
                                    r, g, b = 20, 20, 20
                                else:
                                    r, g, b = 200, 30, 30  # Demon red body
                                draw_pixel = True
                        else:  # Dead (lying flat pool of blood)
                            cx = (tex_x - 0.5) * 2.0
                            cy = (row - (h // 2)) / (sprite_height * 0.5) if sprite_height > 0 else 0.0
                            if cx*cx + cy*cy < 0.8:
                                r, g, b = 120, 10, 10
                                draw_pixel = True
                                
                    elif stype == 2:  # Health pack (3D-perspective shaded box)
                        if 0.25 < tex_x < 0.75 and 0.35 < tex_y < 0.85:
                            # Front-left face: bright white; Side face: shaded gray
                            if tex_x < 0.5:
                                # Red cross on the front-left face
                                if (0.45 < tex_y < 0.75 and 0.33 < tex_x < 0.42) or (0.55 < tex_y < 0.65 and 0.28 < tex_x < 0.47):
                                    r, g, b = 255, 0, 0
                                else:
                                    r, g, b = 240, 240, 240
                            else:
                                r, g, b = 170, 170, 170
                            draw_pixel = True
                            
                    elif stype == 3:  # Ammo pack (gold ammo crate with outlines)
                        if 0.25 < tex_x < 0.75 and 0.4 < tex_y < 0.85:
                            # Check outlines
                            is_outline = (tex_x < 0.28) or (tex_x > 0.72) or (tex_y < 0.43) or (tex_y > 0.82) or (np.abs(tex_x - tex_y - 0.05) < 0.03) or (np.abs(tex_x + tex_y - 1.1) < 0.03)
                            if is_outline:
                                r, g, b = 40, 30, 10
                            else:
                                r, g, b = 210, 160, 30
                            draw_pixel = True
                            
                if draw_pixel:
                    pixels[row, col, 0] = int(r * shading)
                    pixels[row, col, 1] = int(g * shading)
                    pixels[row, col, 2] = int(b * shading)


class GymDoomEnv(gym.Env):
    """Gymnasium environment for a 3D Doom-like First Person Shooter."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.map = np.array(MAP, dtype=np.int32)
        
        # Load HUD font
        try:
            font_properties = font_manager.FontProperties(
                family="sans-serif", weight="bold"
            )
            font_file = font_manager.findfont(font_properties)
            self._hud_font = ImageFont.truetype(font_file, 10)
            self._hud_val_font = ImageFont.truetype(font_file, 14)
        except Exception:
            self._hud_font = ImageFont.load_default()
            self._hud_val_font = ImageFont.load_default()

        # Spaces
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0, high=255, shape=(H, W, 3), dtype=np.uint8
                ),
                "valid_mask": spaces.Box(
                    low=0, high=1, shape=(5,), dtype=np.int8
                ),
                "health": spaces.Box(
                    low=0, high=100, shape=(1,), dtype=np.int32
                ),
                "ammo": spaces.Box(
                    low=0, high=99, shape=(1,), dtype=np.int32
                ),
                "score": spaces.Box(
                    low=0, high=100000, shape=(1,), dtype=np.int32
                ),
            }
        )

        # Initial state setup
        self.reset()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to the initial state."""
        super().reset(seed=seed)

        if options is not None and "state" in options:
            state = options["state"]
            self.player_x = float(state["player_x"])
            self.player_y = float(state["player_y"])
            self.player_angle = float(state["player_angle"])
            self.player_health = int(state["health"])
            self.player_ammo = int(state["ammo"])
            self.player_score = int(state["score"])
            self.enemies = copy.deepcopy(state["enemies"])
            self.items = copy.deepcopy(state["items"])
            self.gun_frame = int(state.get("gun_frame", 0))
            self.total_steps = int(state.get("total_steps", 0))
        else:
            self.player_x = 1.5
            self.player_y = 1.5
            self.player_angle = 0.0
            self.player_health = 100
            self.player_ammo = 20
            self.player_score = 0
            self.gun_frame = 0
            self.total_steps = 0
            
            # Spawn enemies (at least 3)
            self.enemies = [
                {"x": 3.5, "y": 3.5, "health": 100, "status": "idle"},
                {"x": 12.5, "y": 5.5, "health": 100, "status": "idle"},
                {"x": 5.5, "y": 12.5, "health": 100, "status": "idle"},
            ]
            
            # Spawn items (health and ammo packs)
            self.items = [
                {"x": 8.5, "y": 8.5, "type": "health", "active": True},
                {"x": 2.5, "y": 14.5, "type": "health", "active": True},
                {"x": 14.5, "y": 2.5, "type": "ammo", "active": True},
                {"x": 10.5, "y": 10.5, "type": "ammo", "active": True},
            ]

        return self._get_obs(), {}

    def _get_obs(self) -> Dict[str, Any]:
        """Create the current observation dictionary."""
        obs_img = self._render_viewport()
        return {
            "observation": obs_img,
            "valid_mask": np.ones(5, dtype=np.int8),
            "health": np.array([self.player_health], dtype=np.int32),
            "ammo": np.array([self.player_ammo], dtype=np.int32),
            "score": np.array([self.player_score], dtype=np.int32),
        }

    def _get_state(self) -> Dict[str, Any]:
        """Get the full serializable state dictionary."""
        return {
            "player_x": self.player_x,
            "player_y": self.player_y,
            "player_angle": self.player_angle,
            "health": self.player_health,
            "ammo": self.player_ammo,
            "score": self.player_score,
            "enemies": copy.deepcopy(self.enemies),
            "items": copy.deepcopy(self.items),
            "gun_frame": self.gun_frame,
            "total_steps": self.total_steps,
        }

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Perform a step in the environment."""
        if not (0 <= action <= 4):
            raise ValueError(f"Invalid action: {action}")

        self.total_steps += 1
        reward = -0.01  # small step penalty to encourage fast completion
        terminated = False
        truncated = False

        if self.player_health <= 0:
            terminated = True
            return self._get_obs(), -10.0, terminated, truncated, {"state": self._get_state()}

        # 1. Gun Frame updates
        if self.gun_frame > 0:
            self.gun_frame = (self.gun_frame + 1) % 5

        # 2. Resolve Agent Action
        move_speed = 0.2
        rot_speed = 0.1
        
        if action == TURN_LEFT:
            self.player_angle -= rot_speed
        elif action == TURN_RIGHT:
            self.player_angle += rot_speed
        elif action == MOVE_FORWARD:
            dx = math.cos(self.player_angle) * move_speed
            dy = math.sin(self.player_angle) * move_speed
            new_x = self.player_x + dx
            new_y = self.player_y + dy
            # Sliding collision logic
            if not check_collision_jit(new_x, self.player_y, self.map, radius=0.25):
                self.player_x = new_x
            if not check_collision_jit(self.player_x, new_y, self.map, radius=0.25):
                self.player_y = new_y
        elif action == MOVE_BACKWARD:
            dx = math.cos(self.player_angle) * move_speed
            dy = math.sin(self.player_angle) * move_speed
            new_x = self.player_x - dx
            new_y = self.player_y - dy
            if not check_collision_jit(new_x, self.player_y, self.map, radius=0.25):
                self.player_x = new_x
            if not check_collision_jit(self.player_x, new_y, self.map, radius=0.25):
                self.player_y = new_y
        elif action == SHOOT:
            if self.player_ammo > 0:
                self.player_ammo -= 1
                if self.gun_frame == 0:
                    self.gun_frame = 1
                
                # Check for hits on active enemies in crosshair
                hit_enemy_idx = -1
                closest_hit_dist = 999.0
                
                for idx, enemy in enumerate(self.enemies):
                    if enemy["health"] <= 0:
                        continue
                    
                    # Target selection math
                    edx = enemy["x"] - self.player_x
                    edy = enemy["y"] - self.player_y
                    dist = math.hypot(edx, edy)
                    
                    angle_to_enemy = math.atan2(edy, edx)
                    rel_angle = angle_to_enemy - self.player_angle
                    # Normalize
                    rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
                    
                    # Align with center of the screen (crosshair threshold of 0.15 rads)
                    if abs(rel_angle) < 0.15 and line_of_sight_jit(
                        self.player_x, self.player_y, enemy["x"], enemy["y"], self.map
                    ):
                        if dist < closest_hit_dist:
                            closest_hit_dist = dist
                            hit_enemy_idx = idx
                            
                if hit_enemy_idx != -1:
                    enemy = self.enemies[hit_enemy_idx]
                    enemy["health"] -= 50
                    enemy["status"] = "alert"  # Alert them upon hit
                    if enemy["health"] <= 0:
                        enemy["health"] = 0
                        enemy["status"] = "dead"
                        self.player_score += 100
                        reward += 10.0  # Big reward for killing an enemy
                    else:
                        reward += 2.0  # Smaller reward for damaging an enemy
                else:
                    reward -= 0.02  # minor miss penalty
            else:
                reward -= 0.1  # penalty for trying to shoot without ammo

        # Normalize angle
        self.player_angle = (self.player_angle + math.pi) % (2 * math.pi) - math.pi

        # 3. Item Pickup checks
        for item in self.items:
            if item["active"]:
                idx_dist = math.hypot(item["x"] - self.player_x, item["y"] - self.player_y)
                if idx_dist < 0.6:
                    if item["type"] == "health" and self.player_health < 100:
                        self.player_health = min(100, self.player_health + 25)
                        item["active"] = False
                        reward += 2.0
                    elif item["type"] == "ammo" and self.player_ammo < 99:
                        self.player_ammo = min(99, self.player_ammo + 10)
                        item["active"] = False
                        reward += 2.0

        # 4. Enemy AI Updates
        enemy_speed = 0.05
        active_enemies_count = 0
        for enemy in self.enemies:
            if enemy["health"] <= 0:
                continue
            active_enemies_count += 1
            
            # Check line of sight
            has_los = line_of_sight_jit(
                enemy["x"], enemy["y"], self.player_x, self.player_y, self.map
            )
            if has_los:
                enemy["status"] = "alert"
                
            if enemy["status"] == "alert":
                edx = self.player_x - enemy["x"]
                edy = self.player_y - enemy["y"]
                edist = math.hypot(edx, edy)
                
                # Move closer if not in attack range
                if edist > 0.6:
                    ex_step = (edx / edist) * enemy_speed
                    ey_step = (edy / edist) * enemy_speed
                    nex = enemy["x"] + ex_step
                    ney = enemy["y"] + ey_step
                    
                    if not check_collision_jit(nex, ney, self.map, radius=0.25):
                        enemy["x"] = nex
                        enemy["y"] = ney
                    else:
                        if not check_collision_jit(nex, enemy["y"], self.map, radius=0.25):
                            enemy["x"] = nex
                        if not check_collision_jit(enemy["x"], ney, self.map, radius=0.25):
                            enemy["y"] = ney
                            
                # Attack player if extremely close
                if edist < 0.8:
                    self.player_health = max(0, self.player_health - 5)
                    reward -= 1.0  # Penalty for taking damage

        # 5. Terminal Conditions
        if self.player_health <= 0:
            terminated = True
            reward -= 10.0
        elif active_enemies_count == 0:
            terminated = True
            reward += 20.0  # Big victory bonus!

        if self.total_steps >= 500:
            truncated = True

        return self._get_obs(), float(reward), terminated, truncated, {"state": self._get_state()}

    def _render_viewport(self) -> np.ndarray:
        """Render the 3D first-person perspective viewport."""
        # 1. Cast rays & render wall textures
        view_pixels, z_buffer = draw_scene_jit(
            self.map,
            self.player_x,
            self.player_y,
            self.player_angle,
            FOV,
            W,
            VIEW_H,
        )
        
        # 2. Extract and format active sprites for JIT
        sprite_types = []
        sprite_xs = []
        sprite_ys = []
        sprite_states = []
        
        # Enemies
        for enemy in self.enemies:
            sprite_types.append(0)
            sprite_xs.append(enemy["x"])
            sprite_ys.append(enemy["y"])
            sprite_states.append(1 if enemy["health"] > 0 else 0)
            
        # Items
        for item in self.items:
            if item["active"]:
                sprite_types.append(2 if item["type"] == "health" else 3)
                sprite_xs.append(item["x"])
                sprite_ys.append(item["y"])
                sprite_states.append(1)
                
        # Draw all sprites
        if len(sprite_types) > 0:
            draw_sprites_jit(
                view_pixels,
                z_buffer,
                np.array(sprite_types, dtype=np.int32),
                np.array(sprite_xs, dtype=np.float64),
                np.array(sprite_ys, dtype=np.float64),
                np.array(sprite_states, dtype=np.int32),
                self.player_x,
                self.player_y,
                self.player_angle,
                FOV,
                W,
                VIEW_H,
            )
            
        # 3. Handle Recoil Screen Shake (when gun_frame = 1)
        shake_x = 0
        shake_y = 0
        if self.gun_frame == 1:
            shake_x = self.np_random.integers(-4, 5)
            shake_y = self.np_random.integers(-4, 5)
            
        # Paste 3D viewport with recoil screen shake offset
        viewport_img = Image.fromarray(view_pixels)
        shaked_viewport = Image.new("RGB", (W, VIEW_H), (0, 0, 0))
        shaked_viewport.paste(viewport_img, (shake_x, shake_y))
        
        # Create final canvas
        canvas = Image.new("RGB", (W, H), (30, 30, 30))
        canvas.paste(shaked_viewport, (0, 0))
        
        draw = ImageDraw.Draw(canvas)
        
        # 4. Draw Detailed Double-Barrel Shotgun
        gun_x = W // 2
        gun_y = VIEW_H
        if self.gun_frame == 0:  # Idle Double Barrel
            # Draw double barrels
            draw.polygon([(gun_x - 14, gun_y), (gun_x - 10, gun_y - 50), (gun_x - 2, gun_y - 50), (gun_x - 2, gun_y)], fill=(75, 75, 80))
            draw.polygon([(gun_x + 2, gun_y), (gun_x + 2, gun_y - 50), (gun_x + 10, gun_y - 50), (gun_x + 14, gun_y)], fill=(75, 75, 80))
            draw.polygon([(gun_x - 2, gun_y), (gun_x - 2, gun_y - 50), (gun_x + 2, gun_y - 50), (gun_x + 2, gun_y)], fill=(50, 50, 52))
            # barrel highlights
            draw.line((gun_x - 10, gun_y - 48, gun_x - 12, gun_y), fill=(120, 120, 125), width=2)
            draw.line((gun_x + 4, gun_y - 48, gun_x + 4, gun_y), fill=(120, 120, 125), width=2)
            # openings
            draw.ellipse([gun_x - 9, gun_y - 52, gun_x - 3, gun_y - 48], fill=(10, 10, 10))
            draw.ellipse([gun_x + 3, gun_y - 52, gun_x + 9, gun_y - 48], fill=(10, 10, 10))
        elif self.gun_frame == 1:  # Firing (Flash)
            # Draw barrels
            draw.polygon([(gun_x - 14, gun_y), (gun_x - 10, gun_y - 50), (gun_x - 2, gun_y - 50), (gun_x - 2, gun_y)], fill=(75, 75, 80))
            draw.polygon([(gun_x + 2, gun_y), (gun_x + 2, gun_y - 50), (gun_x + 10, gun_y - 50), (gun_x + 14, gun_y)], fill=(75, 75, 80))
            # Double muzzle flashes
            for fx in [gun_x - 6, gun_x + 6]:
                draw.ellipse([fx - 12, gun_y - 62, fx + 12, gun_y - 38], fill=(255, 200, 0))
                draw.ellipse([fx - 6, gun_y - 56, fx + 6, gun_y - 44], fill=(255, 255, 255))
                draw.polygon([(fx, gun_y - 75), (fx - 8, gun_y - 58), (fx - 18, gun_y - 62), (fx - 10, gun_y - 48), (fx, gun_y - 35), (fx + 10, gun_y - 48), (fx + 18, gun_y - 62), (fx + 8, gun_y - 58)], fill=(255, 120, 0))
        elif self.gun_frame == 2:  # Recoil
            # Shifted down
            draw.polygon([(gun_x - 14, gun_y), (gun_x - 10, gun_y - 38), (gun_x - 2, gun_y - 38), (gun_x - 2, gun_y)], fill=(70, 70, 75))
            draw.polygon([(gun_x + 2, gun_y), (gun_x + 2, gun_y - 38), (gun_x + 10, gun_y - 38), (gun_x + 14, gun_y)], fill=(70, 70, 75))
            draw.ellipse([gun_x - 9, gun_y - 40, gun_x - 3, gun_y - 36], fill=(10, 10, 10))
            draw.ellipse([gun_x + 3, gun_y - 40, gun_x + 9, gun_y - 36], fill=(10, 10, 10))
        elif self.gun_frame == 3:  # Reloading / Tilted
            # Tilted barrels
            draw.polygon([(gun_x - 20, gun_y), (gun_x - 24, gun_y - 30), (gun_x - 16, gun_y - 30), (gun_x - 12, gun_y)], fill=(65, 65, 70))
            draw.polygon([(gun_x - 8, gun_y), (gun_x - 12, gun_y - 30), (gun_x - 4, gun_y - 30), (gun_x, gun_y)], fill=(65, 65, 70))
        elif self.gun_frame == 4:  # Returning
            draw.polygon([(gun_x - 14, gun_y), (gun_x - 10, gun_y - 44), (gun_x - 2, gun_y - 44), (gun_x - 2, gun_y)], fill=(72, 72, 77))
            draw.polygon([(gun_x + 2, gun_y), (gun_x + 2, gun_y - 44), (gun_x + 10, gun_y - 44), (gun_x + 14, gun_y)], fill=(72, 72, 77))
            draw.ellipse([gun_x - 9, gun_y - 46, gun_x - 3, gun_y - 42], fill=(10, 10, 10))
            draw.ellipse([gun_x + 3, gun_y - 46, gun_x + 9, gun_y - 42], fill=(10, 10, 10))
            
        # 5. Draw Minimap (top-left corner, 80x80 pixels)
        map_size = 80
        mx_offset = 10
        my_offset = 10
        cell_size = 5
        
        # Base background
        draw.rectangle([mx_offset - 1, my_offset - 1, mx_offset + map_size, my_offset + map_size], outline=(150, 150, 150), fill=(20, 20, 20))
        
        # Walls
        for my in range(16):
            for mx in range(16):
                if self.map[my, mx] == 1:
                    draw.rectangle([mx_offset + mx*cell_size, my_offset + my*cell_size, mx_offset + mx*cell_size + cell_size - 1, my_offset + my*cell_size + cell_size - 1], fill=(80, 80, 80))
                    
        # Items
        for item in self.items:
            if item["active"]:
                ix = mx_offset + int(item["x"] * cell_size)
                iy = my_offset + int(item["y"] * cell_size)
                icolor = (100, 100, 255) if item["type"] == "health" else (220, 220, 0)
                draw.ellipse([ix - 1, iy - 1, ix + 1, iy + 1], fill=icolor)
                
        # Enemies
        for enemy in self.enemies:
            ex = mx_offset + int(enemy["x"] * cell_size)
            ey = my_offset + int(enemy["y"] * cell_size)
            ecolor = (255, 0, 0) if enemy["health"] > 0 else (120, 0, 0)
            draw.ellipse([ex - 1, ey - 1, ex + 1, ey + 1], fill=ecolor)
            
        # Player & FOV lines
        px = mx_offset + int(self.player_x * cell_size)
        py = my_offset + int(self.player_y * cell_size)
        
        left_ang = self.player_angle - FOV / 2.0
        right_ang = self.player_angle + FOV / 2.0
        cone_len = 15
        
        lx = px + math.cos(left_ang) * cone_len
        ly = py + math.sin(left_ang) * cone_len
        rx = px + math.cos(right_ang) * cone_len
        ry = py + math.sin(right_ang) * cone_len
        
        draw.line((px, py, lx, ly), fill=(0, 255, 0), width=1)
        draw.line((px, py, rx, ry), fill=(0, 255, 0), width=1)
        draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(0, 255, 0))

        # 6. Draw HUD (bottom 40 pixels)
        # Background: tech-frame outline and dividers
        draw.rectangle([0, VIEW_H, W, H], fill=(55, 55, 60))
        # Sleek steel-gray tech-frame outline
        draw.rectangle([0, VIEW_H, W - 1, H - 1], outline=(120, 120, 125), width=3)
        # Vertical dividers
        draw.line((75, VIEW_H, 75, H), fill=(120, 120, 125), width=2)
        draw.line((135, VIEW_H, 135, H), fill=(120, 120, 125), width=2)
        draw.line((185, VIEW_H, 185, H), fill=(120, 120, 125), width=2)
        draw.line((240, VIEW_H, 240, H), fill=(120, 120, 125), width=2)
        
        # Draw stats headers
        draw.text((10, VIEW_H + 5), "HEALTH", fill=(255, 80, 80), font=self._hud_font)
        draw.text((85, VIEW_H + 5), "AMMO", fill=(255, 220, 80), font=self._hud_font)
        draw.text((250, VIEW_H + 5), "SCORE", fill=(80, 255, 80), font=self._hud_font)
        
        # Draw values
        draw.text((10, VIEW_H + 18), f"{self.player_health}%", fill=(255, 255, 255), font=self._hud_val_font)
        draw.text((85, VIEW_H + 18), f"{self.player_ammo}", fill=(255, 255, 255), font=self._hud_val_font)
        draw.text((250, VIEW_H + 18), f"{self.player_score}", fill=(255, 255, 255), font=self._hud_val_font)
        
        # 7. Draw Detailed Marine Helmet with Animated Visor
        face_x = W // 2
        face_y = VIEW_H + 20
        face_rad = 12
        
        # Outer Helmet
        draw.ellipse([face_x - face_rad, face_y - face_rad, face_x + face_rad, face_y + face_rad], fill=(100, 100, 105), outline=(30, 30, 30))
        
        # Helmet details
        # Mouth guard grille
        draw.polygon([(face_x - 4, face_y + 4), (face_x + 4, face_y + 4), (face_x + 2, face_y + 10), (face_x - 2, face_y + 10)], fill=(30, 30, 30))
        
        # Determine Visor Color / cracks
        if self.player_health == 0:
            # Dead visor: dark gray & major black cracks
            visor_color = (50, 50, 50)
            draw.polygon([(face_x - 9, face_y - 7), (face_x + 9, face_y - 7), (face_x + 6, face_y - 1), (face_x - 6, face_y - 1)], fill=visor_color)
            draw.line((face_x - 8, face_y - 5, face_x, face_y - 2), fill=(10, 10, 10), width=1)
            draw.line((face_x, face_y - 2, face_x + 8, face_y - 6), fill=(10, 10, 10), width=1)
            draw.line((face_x - 2, face_y - 7, face_x - 4, face_y - 1), fill=(10, 10, 10), width=1)
        elif self.player_health < 30:
            # Low Health: Visor flashes red
            visor_color = (255, 0, 0) if (self.total_steps // 5) % 2 == 0 else (120, 0, 0)
            draw.polygon([(face_x - 9, face_y - 7), (face_x + 9, face_y - 7), (face_x + 6, face_y - 1), (face_x - 6, face_y - 1)], fill=visor_color)
            # Red warning cracks
            draw.line((face_x - 7, face_y - 5, face_x - 2, face_y - 2), fill=(255, 100, 100), width=1)
            draw.line((face_x + 7, face_y - 5, face_x + 2, face_y - 2), fill=(255, 100, 100), width=1)
        else:
            # Healthy: Cyan visor with animated reflection
            visor_color = (0, 220, 255)
            draw.polygon([(face_x - 9, face_y - 7), (face_x + 9, face_y - 7), (face_x + 6, face_y - 1), (face_x - 6, face_y - 1)], fill=visor_color)
            # Animated reflection highlight moving on visor
            ref_x = face_x - 6 + (self.total_steps % 8)
            draw.line((ref_x, face_y - 6, ref_x + 2, face_y - 2), fill=(255, 255, 255), width=1)

        return np.array(canvas, dtype=np.uint8)

    def render(self) -> Optional[np.ndarray]:
        """Return the current screen representation as a 240x320 RGB array."""
        return self._render_viewport()

    def close(self) -> None:
        """Clean up the environment."""
        pass

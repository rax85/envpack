"""A Gymnasium environment for the classic arcade game Space Invaders."""

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from absl import logging
from gymnasium import spaces
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Canvas Constants
WIDTH = 400
HEIGHT = 300
HEADER_PX = 40
FOOTER_PX = 20
PLAY_MIN_Y = HEADER_PX
PLAY_MAX_Y = HEIGHT - FOOTER_PX

# Colors
COLOR_BG = (10, 10, 20)           # Deep space blue-black
COLOR_HEADER = (44, 62, 80)       # Dark slate blue
COLOR_FOOTER = (20, 20, 30)       # Deep slate grey
COLOR_PLAYER = (46, 204, 113)     # Green
COLOR_LASER = (52, 152, 219)      # Blue laser
COLOR_ENEMY_BULLET = (231, 76, 60) # Red enemy bullet
COLOR_BUNKER = (241, 196, 15)     # Yellow
COLOR_TEXT = (236, 240, 241)

# Enemy Types & Scores/Colors
ENEMY_TYPES = [
    {"points": 30, "color": (230, 126, 34)}, # Orange (Top row)
    {"points": 20, "color": (155, 89, 182)}, # Purple (Mid rows)
    {"points": 20, "color": (155, 89, 182)},
    {"points": 10, "color": (52, 152, 219)},  # Blue (Bottom rows)
    {"points": 10, "color": (52, 152, 219)},
]


class GymSpaceInvadersEnv(gym.Env):
    """A Gymnasium environment for Space Invaders."""

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
        # Actions: 0: Stay, 1: Move Left, 2: Move Right, 3: Shoot
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0, high=255, shape=(HEIGHT, WIDTH, 3), dtype=np.uint8
                ),
                "valid_mask": spaces.Box(
                    low=0, high=1, shape=(4,), dtype=np.int8
                ),
                "score": spaces.Box(
                    low=0, high=100000, shape=(1,), dtype=np.int32
                ),
                "lives": spaces.Box(
                    low=0, high=3, shape=(1,), dtype=np.int32
                ),
            }
        )

        # Base background canvas
        self._background = np.full((HEIGHT, WIDTH, 3), COLOR_BG, dtype=np.uint8)
        self._background[0:HEADER_PX, :] = COLOR_HEADER
        self._background[HEIGHT - FOOTER_PX :, :] = COLOR_FOOTER

        self.reset()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to the initial state."""
        super().reset(seed=seed)

        if options is not None and "state" in options:
            state = options["state"]
            self._player_x = state["player_x"]
            self._score = state["score"]
            self._lives = state["lives"]
            self._invaders = copy.deepcopy(state["invaders"])
            self._lasers = copy.deepcopy(state["lasers"])
            self._enemy_bullets = copy.deepcopy(state["enemy_bullets"])
            self._bunkers = copy.deepcopy(state["bunkers"])
            self._invader_dir = state["invader_dir"]
            self._invader_step_timer = state["invader_step_timer"]
            self._cooldown = state["cooldown"]
            self._step_count = state["step_count"]
            self._invulnerable_timer = state["invulnerable_timer"]
            return self._create_observation(), {}

        # Default initialization
        self._player_x = float(WIDTH // 2)
        self._score = 0
        self._lives = 3
        self._invader_dir = 1  # 1 for right, -1 for left
        self._invader_step_timer = 0
        self._cooldown = 0
        self._step_count = 0
        self._invulnerable_timer = 0

        self._lasers: List[Dict[str, Any]] = []        # Player lasers: {"x", "y"}
        self._enemy_bullets: List[Dict[str, Any]] = [] # Enemy bullets: {"x", "y"}

        # Initialize Bunkers: 3 bunkers, each consists of 4 parts (left, center-left, center-right, right)
        # Each part has 3 HP.
        self._bunkers = []
        bunker_x_centers = [80, 200, 320]
        for bx in bunker_x_centers:
            # 4 parts per bunker
            parts = [
                {"x": bx - 15, "y": 230, "w": 8, "h": 12, "hp": 3},
                {"x": bx - 7, "y": 230, "w": 8, "h": 12, "hp": 3},
                {"x": bx + 1, "y": 230, "w": 8, "h": 12, "hp": 3},
                {"x": bx + 9, "y": 230, "w": 8, "h": 12, "hp": 3},
            ]
            self._bunkers.extend(parts)

        # Initialize Invaders Grid: 5 rows, 8 columns
        self._spawn_invaders()

        return self._create_observation(), {}

    def _spawn_invaders(self) -> None:
        """Spawn the grid of space invaders."""
        self._invaders = []
        start_x = 50.0
        start_y = 60.0
        x_spacing = 30.0
        y_spacing = 20.0

        for row in range(5):
            points = ENEMY_TYPES[row]["points"]
            color = ENEMY_TYPES[row]["color"]
            for col in range(8):
                self._invaders.append({
                    "x": start_x + col * x_spacing,
                    "y": start_y + row * y_spacing,
                    "w": 16,
                    "h": 12,
                    "points": points,
                    "color": color,
                    "alive": True,
                })

    def _create_observation(self) -> Dict[str, Any]:
        """Produce the observation dictionary."""
        # Setup valid action mask
        valid_mask = np.ones((4,), dtype=np.int8)
        # If player laser cooldown active, can't shoot
        if self._cooldown > 0:
            valid_mask[3] = 0

        # Create screen render for Box observation
        image = Image.fromarray(self._background.copy())
        draw = ImageDraw.Draw(image)

        # Draw header text
        draw.text((10, 10), "SPACE INVADERS", fill=COLOR_TEXT, font=self._title_font)
        draw.text(
            (220, 12),
            f"SCORE: {self._score:05d}   LIVES: {self._lives}",
            fill=COLOR_TEXT,
            font=self._stats_font,
        )

        # Draw player ship (if not invulnerable or blinking)
        if self._invulnerable_timer == 0 or (self._invulnerable_timer // 3) % 2 == 0:
            px = self._player_x
            py = 265.0
            draw.rectangle(
                [px - 12, py, px + 12, py + 8],
                fill=COLOR_PLAYER,
                outline=(39, 174, 96),
            )
            # Draw turret barrel
            draw.rectangle([px - 2, py - 4, px + 2, py], fill=COLOR_PLAYER)

        # Draw bunkers
        for b in self._bunkers:
            if b["hp"] > 0:
                # Color intensity depends on HP
                color_factor = b["hp"] / 3.0
                color = (
                    int(COLOR_BUNKER[0] * color_factor),
                    int(COLOR_BUNKER[1] * color_factor),
                    int(COLOR_BUNKER[2] * color_factor),
                )
                draw.rectangle(
                    [b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]],
                    fill=color,
                    outline=(100, 100, 10),
                )

        # Draw invaders
        for inv in self._invaders:
            if inv["alive"]:
                ix, iy = inv["x"], inv["y"]
                iw, ih = inv["w"], inv["h"]
                draw.rectangle(
                    [ix - iw // 2, iy - ih // 2, ix + iw // 2, iy + ih // 2],
                    fill=inv["color"],
                    outline=(50, 50, 50),
                )
                # Draw small eyes/pixels for alien detail
                draw.rectangle([ix - 4, iy - 2, ix - 2, iy], fill=(0, 0, 0))
                draw.rectangle([ix + 2, iy - 2, ix + 4, iy], fill=(0, 0, 0))

        # Draw player lasers
        for las in self._lasers:
            lx, ly = las["x"], las["y"]
            draw.rectangle([lx - 1, ly - 5, lx + 1, ly], fill=COLOR_LASER)

        # Draw enemy bullets
        for bul in self._enemy_bullets:
            bx, by = bul["x"], bul["y"]
            draw.rectangle([bx - 1, by, bx + 1, by + 5], fill=COLOR_ENEMY_BULLET)

        # Draw footer info
        draw.text((10, HEIGHT - 15), "STAY [0] | LEFT [1] | RIGHT [2] | SHOOT [3]", fill=COLOR_TEXT, font=self._stats_font)

        rgb_observation = np.array(image, dtype=np.uint8)

        return {
            "observation": rgb_observation,
            "valid_mask": valid_mask,
            "score": np.array([self._score], dtype=np.int32),
            "lives": np.array([self._lives], dtype=np.int32),
        }

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Perform one step of game simulation."""
        self._step_count += 1
        reward = 0.01  # Small step survival reward
        terminated = False
        truncated = False

        # Cooldown & Invulnerability updates
        if self._cooldown > 0:
            self._cooldown -= 1
        if self._invulnerable_timer > 0:
            self._invulnerable_timer -= 1

        # 1. Handle Player Action
        if action == 1:   # LEFT
            self._player_x = max(15.0, self._player_x - 3.0)
        elif action == 2: # RIGHT
            self._player_x = min(WIDTH - 15.0, self._player_x + 3.0)
        elif action == 3 and self._cooldown == 0: # SHOOT
            self._lasers.append({"x": self._player_x, "y": 260.0})
            self._cooldown = 12  # Laser fire cooldown

        # 2. Update Invaders Movement
        # Determine the speed factor based on how many invaders are left
        alive_invaders = [inv for inv in self._invaders if inv["alive"]]
        if len(alive_invaders) == 0:
            # Clear wave bonus
            reward += 15.0
            self._score += 1000
            self._spawn_invaders()
            alive_invaders = [inv for inv in self._invaders if inv["alive"]]

        total_invaders = 40
        fraction_alive = len(alive_invaders) / total_invaders
        # Invaders move faster as they die
        invader_speed_steps = max(2, int(15 * fraction_alive))

        self._invader_step_timer += 1
        if self._invader_step_timer >= invader_speed_steps:
            self._invader_step_timer = 0
            
            # Check if any invader hits the wall boundaries
            hit_wall = False
            for inv in alive_invaders:
                if self._invader_dir == 1 and inv["x"] >= WIDTH - 20:
                    hit_wall = True
                    break
                elif self._invader_dir == -1 and inv["x"] <= 20:
                    hit_wall = True
                    break

            if hit_wall:
                # Shift down and change direction
                self._invader_dir *= -1
                for inv in self._invaders:
                    inv["y"] += 8.0
            else:
                # Move sideways
                for inv in self._invaders:
                    inv["x"] += self._invader_dir * 4.0

        # Check if invaders reached player level (y >= 250)
        for inv in alive_invaders:
            if inv["y"] >= 250.0:
                reward -= 10.0
                terminated = True

        # 3. Invader Shooting
        if len(alive_invaders) > 0 and self.np_random.uniform() < 0.05:
            # Pick a random alive invader to shoot
            shooting_invader = self.np_random.choice(alive_invaders)
            self._enemy_bullets.append({
                "x": shooting_invader["x"],
                "y": shooting_invader["y"] + 6.0
            })

        # 4. Update Player Lasers
        for las in self._lasers[:]:
            las["y"] -= 6.0
            if las["y"] < PLAY_MIN_Y:
                self._lasers.remove(las)
                continue

            # Check collision with invaders
            hit_invader = False
            for inv in self._invaders:
                if inv["alive"]:
                    dist_x = abs(las["x"] - inv["x"])
                    dist_y = abs(las["y"] - inv["y"])
                    if dist_x < (inv["w"] // 2 + 2) and dist_y < (inv["h"] // 2 + 2):
                        inv["alive"] = False
                        hit_invader = True
                        self._score += inv["points"]
                        reward += inv["points"] / 10.0
                        break
            
            if hit_invader:
                self._lasers.remove(las)
                continue

            # Check collision with bunkers
            hit_bunker = False
            for b in self._bunkers:
                if b["hp"] > 0:
                    if b["x"] <= las["x"] <= b["x"] + b["w"] and b["y"] <= las["y"] <= b["y"] + b["h"]:
                        b["hp"] -= 1
                        hit_bunker = True
                        break
            if hit_bunker:
                self._lasers.remove(las)

        # 5. Update Enemy Bullets
        for bul in self._enemy_bullets[:]:
            bul["y"] += 4.0
            if bul["y"] > PLAY_MAX_Y:
                self._enemy_bullets.remove(bul)
                continue

            # Check collision with bunkers
            hit_bunker = False
            for b in self._bunkers:
                if b["hp"] > 0:
                    if b["x"] <= bul["x"] <= b["x"] + b["w"] and b["y"] <= bul["y"] <= b["y"] + b["h"]:
                        b["hp"] -= 1
                        hit_bunker = True
                        break
            if hit_bunker:
                self._enemy_bullets.remove(bul)
                continue

            # Check collision with player ship
            if self._invulnerable_timer == 0:
                px = self._player_x
                py = 265.0
                dist_x = abs(bul["x"] - px)
                dist_y = abs(bul["y"] - (py + 4.0))
                if dist_x < 14 and dist_y < 8:
                    # Player hit!
                    self._lives -= 1
                    self._invulnerable_timer = 45  # Flashing invulnerability
                    reward -= 5.0
                    self._enemy_bullets.remove(bul)
                    if self._lives <= 0:
                        reward -= 10.0
                        terminated = True
                    continue

        return self._create_observation(), float(reward), terminated, truncated, {"state": self._get_state()}

    def _get_state(self) -> Dict[str, Any]:
        """Return the environment state dictionary."""
        return {
            "player_x": self._player_x,
            "score": self._score,
            "lives": self._lives,
            "invaders": copy.deepcopy(self._invaders),
            "lasers": copy.deepcopy(self._lasers),
            "enemy_bullets": copy.deepcopy(self._enemy_bullets),
            "bunkers": copy.deepcopy(self._bunkers),
            "invader_dir": self._invader_dir,
            "invader_step_timer": self._invader_step_timer,
            "cooldown": self._cooldown,
            "step_count": self._step_count,
            "invulnerable_timer": self._invulnerable_timer,
        }

    def render(self) -> np.ndarray:
        """Render the environment to an RGB array."""
        obs = self._create_observation()
        return obs["observation"]

    def close(self) -> None:
        """Clean up resources."""
        pass

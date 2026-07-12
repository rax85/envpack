"""A Gymnasium environment for a simultaneous 2-player Street Fighter-style fighting game."""

import math
import copy
from typing import Any, Tuple, Dict, Optional, List

import gymnasium as gym
from absl import logging
from gymnasium import spaces
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import numpy.typing as npt

# Action Constants
IDLE = 0
WALK_LEFT = 1
WALK_RIGHT = 2
JUMP = 3
CROUCH = 4
PUNCH = 5
KICK = 6
SPECIAL_FIREBALL = 7

# Get appropriate Lanczos constant from PIL
if hasattr(Image, "Resampling"):
    LANCZOS = Image.Resampling.LANCZOS
else:
    LANCZOS = Image.LANCZOS


class GymStreetFighterEnv(gym.Env):
    """Gymnasium environment for 2-player Street Fighter-style game."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = "rgb_array") -> None:
        super().__init__()
        self.render_mode = render_mode

        # Font setup (using matplotlib's font manager to locate a bold sans-serif font)
        try:
            font_properties = font_manager.FontProperties(
                family="sans-serif", weight="bold"
            )
            font_file = font_manager.findfont(font_properties)
            self._font_large = ImageFont.truetype(font_file, 36)  # 12pt * 3
            self._font_huge = ImageFont.truetype(font_file, 54)   # 18pt * 3
            self._font_small = ImageFont.truetype(font_file, 24)  # 8pt * 3
        except Exception:
            logging.warning("Could not load system sans-serif font. Using default font.")
            self._font_large = ImageFont.load_default()
            self._font_huge = ImageFont.load_default()
            self._font_small = ImageFont.load_default()

        # Action space: MultiDiscrete([8, 8]) representing P1 and P2 actions
        self.action_space = spaces.MultiDiscrete([8, 8])

        # Observation space
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0, high=255, shape=(300, 400, 3), dtype=np.uint8
                ),
                "valid_mask": spaces.Box(
                    low=0, high=1, shape=(2, 8), dtype=np.int8
                ),
                "health": spaces.Box(
                    low=0, high=100, shape=(2,), dtype=np.int32
                ),
                "total_score": spaces.Box(
                    low=0, high=1000, shape=(2,), dtype=np.int32
                ),
            }
        )

        # Stage parameters
        self.stage_width = 400
        self.floor_y = 240

        # Physics parameters
        self.walk_speed = 3.0
        self.jump_vy = 12.0
        self.gravity = 0.8

        # Create static base background at 3x scale (1200x900)
        self._base_bg = self._create_base_bg()

        self.reset()

    def _create_base_bg(self) -> Image.Image:
        """Create static Dojo background at 3x scale."""
        bg = Image.new("RGB", (1200, 900))
        draw = ImageDraw.Draw(bg)

        # 1. Sunset sky gradient (y from 0 to 720)
        for y in range(720):
            if y < 360:
                # purple (80, 20, 120) to red (200, 40, 40)
                t = y / 360.0
                r = int(80 + (200 - 80) * t)
                g = int(20 + (40 - 20) * t)
                b = int(120 + (40 - 120) * t)
            else:
                # red (200, 40, 40) to orange (240, 120, 20)
                t = (y - 360) / 360.0
                r = int(200 + (240 - 200) * t)
                g = int(40 + (120 - 40) * t)
                b = int(40 + (20 - 40) * t)
            draw.line([(0, y), (1200, y)], fill=(r, g, b))

        # 2. Shoji Windows in Dojo background (from y=150 to y=540, x from 120 to 1080)
        draw.rectangle([120, 150, 1080, 540], fill=(245, 240, 225))
        # Shoji grid lines
        for y in range(150, 541, 60):
            draw.line([(120, y), (1080, y)], fill=(80, 50, 30), width=6)
        for x in range(120, 1081, 80):
            draw.line([(x, 150), (x, 540)], fill=(80, 50, 30), width=6)
        # Outer frame of Shoji
        draw.rectangle([120, 150, 1080, 540], outline=(80, 50, 30), width=12)

        # 3. Wooden Floor (y from 720 to 900)
        draw.rectangle([0, 720, 1200, 900], fill=(160, 110, 60))
        # Horizontal floorboards
        for y in range(720, 901, 30):
            draw.line([(0, y), (1200, y)], fill=(100, 70, 30), width=3)
        # Vertical board edges
        for x in range(0, 1201, 150):
            draw.line([(x, 720), (x, 900)], fill=(100, 70, 30), width=3)

        # 4. Red Pillars on left and right
        draw.rectangle([0, 0, 120, 720], fill=(180, 30, 30))
        draw.rectangle([0, 0, 120, 720], outline=(100, 20, 20), width=6)
        draw.rectangle([1080, 0, 1200, 720], fill=(180, 30, 30))
        draw.rectangle([1080, 0, 1200, 720], outline=(100, 20, 20), width=6)

        return bg

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment state."""
        super().reset(seed=seed)

        # Initialize internal variables
        self.x = [100.0, 300.0]
        self.y_offset = [0.0, 0.0]
        self.vy = [0.0, 0.0]
        self.vx = [0.0, 0.0]
        self.facing = [1, -1]  # 1: face right, -1: face left
        self.health = [100, 100]
        self.wins = [0, 0]  # P1 wins, P2 wins
        self.timer = 99
        self.round_steps = 0
        self.total_steps = 0
        self.state = ["idle", "idle"]
        self.hitstun = [0, 0]
        self.knockdown = [0, 0]
        self.combo_count = [0, 0]
        self.combo_timer = [0, 0]
        self.last_horizontal_dir = [0, 0]
        self.fireballs = []
        self.sparks = []

        # State injection support
        if options is not None and "state" in options:
            state = options["state"]
            self.x = list(map(float, state.get("x", [100.0, 300.0])))
            self.y_offset = list(map(float, state.get("y_offset", [0.0, 0.0])))
            self.vy = list(map(float, state.get("vy", [0.0, 0.0])))
            self.vx = list(map(float, state.get("vx", [0.0, 0.0])))
            self.facing = list(map(int, state.get("facing", [1, -1])))
            self.health = list(map(int, state.get("health", [100, 100])))
            self.wins = list(map(int, state.get("wins", [0, 0])))
            self.timer = int(state.get("timer", 99))
            self.round_steps = int(state.get("round_steps", 0))
            self.total_steps = int(state.get("total_steps", 0))
            self.state = list(state.get("state", ["idle", "idle"]))
            self.hitstun = list(map(int, state.get("hitstun", [0, 0])))
            self.knockdown = list(map(int, state.get("knockdown", [0, 0])))
            self.combo_count = list(map(int, state.get("combo_count", [0, 0])))
            self.combo_timer = list(map(int, state.get("combo_timer", [0, 0])))
            self.last_horizontal_dir = list(
                map(int, state.get("last_horizontal_dir", [0, 0]))
            )
            self.fireballs = copy.deepcopy(state.get("fireballs", []))
            self.sparks = copy.deepcopy(state.get("sparks", []))

        # Dynamic facing update
        self._update_facing()

        observation = self._create_observation()
        return observation, {}

    def _update_facing(self) -> None:
        """Update facing direction so players face each other."""
        if self.x[0] < self.x[1]:
            self.facing[0] = 1
            self.facing[1] = -1
        elif self.x[0] > self.x[1]:
            self.facing[0] = -1
            self.facing[1] = 1

    def _get_valid_mask(self) -> npt.NDArray[np.int8]:
        """Compute the mask of valid moves (always all ones)."""
        return np.ones((2, 8), dtype=np.int8)

    def _is_blocking(
        self, attacker_idx: int, attack_type: str, defender_idx: int, defender_action: int
    ) -> bool:
        """Determine if defender successfully blocks the attack."""
        if self.y_offset[defender_idx] > 0:
            # Cannot block in mid-air
            return False

        # Standing block condition: holding backward (away from opponent)
        # If opponent is to the right of defender, defender must walk left (1) to block.
        # If opponent is to the left of defender, defender must walk right (2) to block.
        is_holding_backward = False
        if self.x[attacker_idx] > self.x[defender_idx]:
            if defender_action == WALK_LEFT:
                is_holding_backward = True
        else:
            if defender_action == WALK_RIGHT:
                is_holding_backward = True

        # Crouching block condition: crouch action (4)
        is_crouching = (defender_action == CROUCH)

        if attack_type in ["standing_punch", "standing_kick"]:
            return is_holding_backward
        elif attack_type in ["crouching_punch", "crouching_kick"]:
            return is_crouching
        elif attack_type == "fireball":
            # Fireball is blocked by either standing block or crouching block
            return is_holding_backward or is_crouching

        return False

    def step(
        self, action: npt.NDArray[np.int32]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one step."""
        action = np.asarray(action, dtype=np.int32)
        if action.shape != (2,):
            raise ValueError(f"Action must be of shape (2,), got {action.shape}")

        self.total_steps += 1
        self.round_steps += 1

        # Decrement round timer
        if self.round_steps % 10 == 0:
            self.timer = max(0, self.timer - 1)

        # Parse actions for both players
        act1, act2 = action[0], action[1]

        # Check hitstun and knockdown states before applying actions
        # Hitstun / knockdown overrides actions to IDLE (0)
        resolved_actions = [act1, act2]
        in_knockdown = [False, False]
        in_hitstun = [False, False]
        for p in (0, 1):
            if self.knockdown[p] > 0:
                in_knockdown[p] = True
                self.knockdown[p] -= 1
                resolved_actions[p] = IDLE
                self.state[p] = "knockdown"
            elif self.hitstun[p] > 0:
                in_hitstun[p] = True
                self.hitstun[p] -= 1
                resolved_actions[p] = IDLE
                self.state[p] = "hitstun"
            else:
                if self.state[p] in ["hitstun", "knockdown"]:
                    self.state[p] = "idle"

        # Record if player was crouching in the previous step
        was_crouching = [
            (self.state[p] in ["crouch", "crouch_punch", "crouch_kick"])
            for p in (0, 1)
        ]

        # Apply movements and transition states
        for p in (0, 1):
            if in_knockdown[p] or in_hitstun[p]:
                continue
            act = resolved_actions[p]

            if self.y_offset[p] > 0:
                # In the air (jump physics)
                self.y_offset[p] += self.vy[p]
                self.vy[p] -= self.gravity
                self.x[p] += self.vx[p]
                self.state[p] = "jump"

                # Check landing
                if self.y_offset[p] <= 0:
                    self.y_offset[p] = 0.0
                    self.vy[p] = 0.0
                    self.vx[p] = 0.0
                    self.state[p] = "idle"

                # Standard jumping punch/kick animations if triggered mid-air
                if act == PUNCH:
                    self.state[p] = "punch"
                elif act == KICK:
                    self.state[p] = "kick"

            else:
                # Grounded state
                if act == WALK_LEFT:
                    self.x[p] -= self.walk_speed
                    self.last_horizontal_dir[p] = -1
                    self.state[p] = "walk"
                elif act == WALK_RIGHT:
                    self.x[p] += self.walk_speed
                    self.last_horizontal_dir[p] = 1
                    self.state[p] = "walk"
                elif act == JUMP:
                    # Initiate jump
                    self.y_offset[p] = 1.0
                    self.vy[p] = self.jump_vy
                    # Jump direction based on last walk direction
                    self.vx[p] = self.last_horizontal_dir[p] * self.walk_speed
                    self.state[p] = "jump"
                elif act == CROUCH:
                    self.state[p] = "crouch"
                    self.last_horizontal_dir[p] = 0
                elif act == PUNCH:
                    self.state[p] = "crouch_punch" if was_crouching[p] else "punch"
                    self.last_horizontal_dir[p] = 0
                elif act == KICK:
                    self.state[p] = "crouch_kick" if was_crouching[p] else "kick"
                    self.last_horizontal_dir[p] = 0
                elif act == SPECIAL_FIREBALL:
                    self.state[p] = "fireball"
                    self.last_horizontal_dir[p] = 0
                else:
                    self.state[p] = "idle"
                    self.last_horizontal_dir[p] = 0

            # Clamp coordinates to screen boundaries
            self.x[p] = float(np.clip(self.x[p], 15.0, 385.0))

        # Dynamic facing update based on new positions
        self._update_facing()

        # Keep track of rewards
        reward_p1_deltas = 0.0
        reward_p2_deltas = 0.0

        # Resolve close-range strikes (PUNCH / KICK)
        # Note: both strikes happen simultaneously, trade hits can occur
        strike_hits = [False, False]
        strike_details = []  # list of (attacker, defender, damage, hitstun, reach, knockback, type)

        for p in (0, 1):
            opp = 1 - p
            # Check if player is performing a close-range strike
            if self.state[p] in ["punch", "crouch_punch", "kick", "crouch_kick"]:
                # Determine reach and details
                if "punch" in self.state[p]:
                    reach = 25.0
                    damage = 5
                    hitstun_dur = 8
                    knockback_dist = 15.0
                    attack_type = "crouching_punch" if "crouch" in self.state[p] else "standing_punch"
                else:
                    reach = 35.0
                    damage = 8
                    hitstun_dur = 12
                    knockback_dist = 20.0
                    attack_type = "crouching_kick" if "crouch" in self.state[p] else "standing_kick"

                # Check horizontal and vertical range
                dist = abs(self.x[p] - self.x[opp])
                vert_dist = abs(self.y_offset[p] - self.y_offset[opp])

                if dist <= reach and vert_dist <= 40.0:
                    strike_details.append(
                        (p, opp, damage, hitstun_dur, reach, knockback_dist, attack_type)
                    )

        # Apply close range strike results
        for p, opp, damage, hitstun_dur, reach, knockback_dist, attack_type in strike_details:
            # Check blocking
            blocked = self._is_blocking(p, attack_type, opp, resolved_actions[opp])
            if blocked:
                # Spawn block effect, 0 damage, 0 hitstun
                pass
            else:
                # Apply hit
                self.health[opp] = max(0, self.health[opp] - damage)
                self.hitstun[opp] = hitstun_dur
                self.state[opp] = "hitstun"

                # Knockback
                self.x[opp] += self.facing[p] * knockback_dist
                self.x[opp] = float(np.clip(self.x[opp], 15.0, 385.0))

                # Spark at midpoint
                spark_x = (self.x[p] + self.x[opp]) / 2.0
                spark_y = self.floor_y - 40.0 - self.y_offset[opp]
                self.sparks.append({"x": spark_x, "y": spark_y, "lifetime": 4})

                # Combo counters
                if self.combo_timer[p] > 0:
                    self.combo_count[p] += 1
                else:
                    self.combo_count[p] = 1
                self.combo_timer[p] = 15

                # Reward signals
                if p == 0:
                    reward_p1_deltas += damage * 0.1
                    reward_p2_deltas -= damage * 0.1
                else:
                    reward_p2_deltas += damage * 0.1
                    reward_p1_deltas -= damage * 0.1

        # Resolve SPECIAL_FIREBALL spawning
        for p in (0, 1):
            if resolved_actions[p] == SPECIAL_FIREBALL and self.y_offset[p] == 0:
                # Standard limit: maximum 1 active fireball per player
                has_active = any(f["owner"] == p for f in self.fireballs)
                if not has_active:
                    self.fireballs.append(
                        {
                            "x": self.x[p] + self.facing[p] * 20.0,
                            "y": self.floor_y - 40.0,
                            "dir": self.facing[p],
                            "owner": p,
                            "speed": 5.0,
                            "active": True,
                        }
                    )

        # Update fireball positions
        for f in self.fireballs:
            f["x"] += f["dir"] * f["speed"]

        # Check fireball-fireball collisions (distance < 12 cancels out)
        for i in range(len(self.fireballs)):
            for j in range(i + 1, len(self.fireballs)):
                f1 = self.fireballs[i]
                f2 = self.fireballs[j]
                if f1["active"] and f2["active"] and f1["owner"] != f2["owner"]:
                    if abs(f1["x"] - f2["x"]) < 12.0:
                        f1["active"] = False
                        f2["active"] = False
                        # Spawn neutral clash spark
                        self.sparks.append(
                            {
                                "x": (f1["x"] + f2["x"]) / 2.0,
                                "y": self.floor_y - 40.0,
                                "lifetime": 4,
                            }
                        )

        # Check fireball-player collisions
        for f in self.fireballs:
            if not f["active"]:
                continue
            opp = 1 - f["owner"]
            # Opponent vertical reach checks
            if abs(f["x"] - self.x[opp]) < 20.0 and self.y_offset[opp] < 45.0:
                f["active"] = False

                # Check blocking
                blocked = self._is_blocking(f["owner"], "fireball", opp, resolved_actions[opp])
                if blocked:
                    # Spawn hit spark at contact, but no damage/hitstun
                    self.sparks.append({"x": f["x"], "y": f["y"], "lifetime": 4})
                else:
                    # Damage 10, hitstun 10, knockback 20
                    self.health[opp] = max(0, self.health[opp] - 10)
                    self.hitstun[opp] = 10
                    self.state[opp] = "hitstun"

                    # Knockback
                    self.x[opp] += f["dir"] * 20.0
                    self.x[opp] = float(np.clip(self.x[opp], 15.0, 385.0))

                    # Spark
                    self.sparks.append({"x": f["x"], "y": f["y"], "lifetime": 4})

                    # Combo counters
                    owner = f["owner"]
                    if self.combo_timer[owner] > 0:
                        self.combo_count[owner] += 1
                    else:
                        self.combo_count[owner] = 1
                    self.combo_timer[owner] = 15

                    # Reward signals
                    if owner == 0:
                        reward_p1_deltas += 1.0
                        reward_p2_deltas -= 1.0
                    else:
                        reward_p2_deltas += 1.0
                        reward_p1_deltas -= 1.0

        # Filter active fireballs inside screen bounds
        self.fireballs = [
            f for f in self.fireballs if f["active"] and 0.0 <= f["x"] <= 400.0
        ]

        # Update hit sparks lifetime
        for s in self.sparks:
            s["lifetime"] -= 1
        self.sparks = [s for s in self.sparks if s["lifetime"] > 0]

        # Update combo timers
        for p in (0, 1):
            if self.combo_timer[p] > 0:
                self.combo_timer[p] -= 1
                if self.combo_timer[p] == 0:
                    self.combo_count[p] = 0

        # Check for round end condition due to health reach 0
        for p in (0, 1):
            if self.health[p] <= 0 and self.knockdown[p] == 0 and self.state[p] != "knockdown":
                self.knockdown[p] = 20  # Duration for knockdown pose
                self.state[p] = "knockdown"
                self.vy[p] = 0.0
                self.y_offset[p] = 0.0

        # Check round status
        round_done = False
        round_winner = -1

        # Check if knockdown timer just hit 0 for a defeated player, indicating round reset
        defeated_p = [p for p in (0, 1) if self.health[p] <= 0 and self.knockdown[p] == 0]
        if defeated_p or self.timer <= 0:
            round_done = True
            # Determine round winner
            if self.health[0] <= 0 and self.health[1] <= 0:
                # Double KO (draw, no wins awarded or both? Let's say no win)
                round_winner = -1
            elif self.health[0] <= 0:
                round_winner = 1
            elif self.health[1] <= 0:
                round_winner = 0
            else:
                # Timeout winner
                if self.health[0] > self.health[1]:
                    round_winner = 0
                elif self.health[1] > self.health[0]:
                    round_winner = 1
                else:
                    round_winner = -1

        reward = reward_p1_deltas

        terminated = False
        if round_done:
            if round_winner == 0:
                self.wins[0] += 1
                reward += 10.0
            elif round_winner == 1:
                self.wins[1] += 1
                reward -= 10.0

            # Check if match is over (first to 2 wins)
            if self.wins[0] >= 2:
                reward += 50.0
                terminated = True
            elif self.wins[1] >= 2:
                reward -= 50.0
                terminated = True
            else:
                # Reset for next round
                self.health = [100, 100]
                self.x = [100.0, 300.0]
                self.y_offset = [0.0, 0.0]
                self.vy = [0.0, 0.0]
                self.vx = [0.0, 0.0]
                self.state = ["idle", "idle"]
                self.hitstun = [0, 0]
                self.knockdown = [0, 0]
                self.timer = 99
                self.round_steps = 0
                self.facing = [1, -1]
                self.fireballs = []
                self.sparks = []
                self.combo_count = [0, 0]
                self.combo_timer = [0, 0]

        truncated = (self.total_steps >= 1000)

        observation = self._create_observation()
        info = {"state": self._get_state()}
        return observation, float(reward), terminated, truncated, info

    def _get_state(self) -> Dict[str, Any]:
        """Return a copy of the current internal state."""
        return {
            "x": list(self.x),
            "y_offset": list(self.y_offset),
            "vy": list(self.vy),
            "vx": list(self.vx),
            "facing": list(self.facing),
            "health": list(self.health),
            "wins": list(self.wins),
            "timer": self.timer,
            "round_steps": self.round_steps,
            "total_steps": self.total_steps,
            "state": list(self.state),
            "hitstun": list(self.hitstun),
            "knockdown": list(self.knockdown),
            "combo_count": list(self.combo_count),
            "combo_timer": list(self.combo_timer),
            "last_horizontal_dir": list(self.last_horizontal_dir),
            "fireballs": copy.deepcopy(self.fireballs),
            "sparks": copy.deepcopy(self.sparks),
        }

    def _create_observation(self) -> Dict[str, Any]:
        """Create the Gym observation dictionary."""
        obs_img = self._render_rgb_array()
        return {
            "observation": obs_img,
            "valid_mask": self._get_valid_mask(),
            "health": np.array(self.health, dtype=np.int32),
            "total_score": np.array(self.wins, dtype=np.int32),
        }

    def _render_rgb_array(self) -> npt.NDArray[np.uint8]:
        """Draw visuals representing the current frame with 3x SSAA and Lanczos downscaling."""
        # 1. Start with static base background
        canvas = self._base_bg.copy()
        draw = ImageDraw.Draw(canvas)

        # 2. Draw fireballs
        for f in self.fireballs:
            cx = f["x"] * 3.0
            cy = f["y"] * 3.0
            fd = f["dir"]
            if f["owner"] == 0:
                # Cyan Hadouken for Ryu
                # Outer glow
                draw.ellipse([cx - 24, cy - 24, cx + 24, cy + 24], fill=(100, 240, 255))
                # Inner core
                draw.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=(255, 255, 255))
                # Trail
                draw.ellipse([cx - fd * 36 - 18, cy - 18, cx - fd * 36 + 18, cy + 18], fill=(50, 180, 220))
                draw.ellipse([cx - fd * 72 - 12, cy - 12, cx - fd * 72 + 12, cy + 12], fill=(20, 120, 180))
            else:
                # Orange Hadouken for Ken
                # Outer glow
                draw.ellipse([cx - 24, cy - 24, cx + 24, cy + 24], fill=(255, 140, 40))
                # Inner core
                draw.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=(255, 230, 150))
                # Trail
                draw.ellipse([cx - fd * 36 - 18, cy - 18, cx - fd * 36 + 18, cy + 18], fill=(220, 80, 20))
                draw.ellipse([cx - fd * 72 - 12, cy - 12, cx - fd * 72 + 12, cy + 12], fill=(180, 40, 10))

        # 3. Draw characters
        # P1 is Ryu: gi=white, belt=black, headband=red, hair=dark
        self._draw_character(
            draw,
            p_idx=0,
            gi_color=(240, 240, 240),
            belt_color=(30, 30, 30),
            hair_color=(45, 35, 30),
            headband_color=(220, 30, 30),
        )

        # P2 is Ken: gi=red, belt=black, headband=None, hair=blonde
        self._draw_character(
            draw,
            p_idx=1,
            gi_color=(200, 30, 30),
            belt_color=(30, 30, 30),
            hair_color=(245, 220, 70),
            headband_color=None,
        )

        # 4. Draw hit sparks
        for s in self.sparks:
            cx = s["x"] * 3.0
            cy = s["y"] * 3.0
            # Draw bright yellow star / flash
            draw.ellipse([cx - 24, cy - 8, cx + 24, cy + 8], fill=(255, 230, 100))
            draw.ellipse([cx - 8, cy - 24, cx + 8, cy + 24], fill=(255, 230, 100))
            draw.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=(255, 255, 255))

        # 5. Draw HUD elements (Health bars, wins, timer, combos)
        # Ryu Health Bar (P1)
        draw.rectangle([60, 60, 510, 105], outline=(50, 50, 50), width=6)
        draw.rectangle([63, 63, 507, 102], fill=(180, 40, 40))
        p1_bar_w = int(444 * (self.health[0] / 100.0))
        if p1_bar_w > 0:
            draw.rectangle([63, 63, 63 + p1_bar_w, 102], fill=(40, 200, 40))
        draw.text((60, 20), "RYU", fill=(255, 255, 255), font=self._font_large)

        # Ken Health Bar (P2)
        draw.rectangle([690, 60, 1140, 105], outline=(50, 50, 50), width=6)
        draw.rectangle([693, 63, 1137, 102], fill=(180, 40, 40))
        p2_bar_w = int(444 * (self.health[1] / 100.0))
        if p2_bar_w > 0:
            draw.rectangle([1137 - p2_bar_w, 63, 1137, 102], fill=(40, 200, 40))
        draw.text((1140, 20), "KEN", fill=(255, 255, 255), font=self._font_large, anchor="rt")

        # Round Indicators
        # P1 Wins
        for i in range(2):
            x = 60 + i * 35
            if self.wins[0] > i:
                draw.ellipse([x - 10, 125, x + 10, 145], fill=(220, 40, 40))
            else:
                draw.ellipse([x - 10, 125, x + 10, 145], outline=(100, 100, 100), width=3)
        # P2 Wins
        for i in range(2):
            x = 1140 - i * 35
            if self.wins[1] > i:
                draw.ellipse([x - 10, 125, x + 10, 145], fill=(220, 40, 40))
            else:
                draw.ellipse([x - 10, 125, x + 10, 145], outline=(100, 100, 100), width=3)

        # Timer
        draw.text((600, 82), f"{self.timer:02d}", fill=(255, 255, 255), font=self._font_huge, anchor="mm")

        # Combo Counters
        if self.combo_count[0] >= 2 and self.combo_timer[0] > 0:
            draw.text((60, 180), f"{self.combo_count[0]} HIT COMBO!", fill=(255, 220, 0), font=self._font_large)
        if self.combo_count[1] >= 2 and self.combo_timer[1] > 0:
            draw.text((1140, 180), f"{self.combo_count[1]} HIT COMBO!", fill=(255, 220, 0), font=self._font_large, anchor="rt")

        # 6. Downscale to 400x300 using Lanczos filter
        resized = canvas.resize((400, 300), LANCZOS)
        return np.array(resized, dtype=np.uint8)

    def _draw_character(
        self,
        draw: ImageDraw.ImageDraw,
        p_idx: int,
        gi_color: Tuple[int, int, int],
        belt_color: Tuple[int, int, int],
        hair_color: Tuple[int, int, int],
        headband_color: Optional[Tuple[int, int, int]],
    ) -> None:
        """Draw character body parts based on state on 3x canvas."""
        x = self.x[p_idx]
        y_offset = self.y_offset[p_idx]
        facing = self.facing[p_idx]
        state = self.state[p_idx]

        skin_color = (255, 219, 172)
        cx = x * 3.0
        cy = (self.floor_y - y_offset) * 3.0

        # Helper functions to draw shapes without ordering issues
        def draw_rect(x0, y0, x1, y1, fill=None, outline=None, width=1):
            draw.rectangle(
                [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                fill=fill,
                outline=outline,
                width=width,
            )

        def draw_ellipse(x0, y0, x1, y1, fill=None, outline=None, width=1):
            draw.ellipse(
                [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                fill=fill,
                outline=outline,
                width=width,
            )

        # Set vertical offsets based on bounce/crouch/jump/knockdown state
        bounce = 0
        if state == "idle":
            # Idle bouncing motion
            bounce = int(math.sin(self.total_steps * 0.4) * 6)

        # Handle Pose-specific coordinates
        if state in ["knockdown", "ko"]:
            # Knocked down flat pose
            # Head
            head_cx = cx - facing * 80
            head_cy = cy - 20
            draw_ellipse(head_cx - 20, head_cy - 20, head_cx + 20, head_cy + 20, fill=skin_color)
            # Hair
            draw.chord([head_cx - 22, head_cy - 22, head_cx + 10, head_cy + 10], start=180, end=360, fill=hair_color)
            if headband_color:
                draw_rect(head_cx - 21, head_cy - 6, head_cx - 5, head_cy - 2, fill=headband_color)
            # Gi Body
            draw_rect(cx - facing * 60, cy - 20, cx + facing * 20, cy, fill=gi_color)
            # Belt
            draw_rect(cx + facing * 10, cy - 22, cx + facing * 18, cy + 2, fill=belt_color)
            # Legs
            draw_rect(cx + facing * 20, cy - 20, cx + facing * 80, cy, fill=gi_color)
            return

        # Regular poses: standing, walking, crouching, jumping, attacking, hitstun
        is_crouching = "crouch" in state
        is_hitstun = state == "hitstun"

        # Apply tilt/offsets for hitstun
        tilt_x = 0.0
        if is_hitstun:
            tilt_x = -facing * 18.0

        if is_crouching:
            head_cy = cy - 140
            body_top = cy - 120
            body_bottom = cy - 60
            shoulder_y = cy - 100
            hip_y = cy - 60
        else:
            head_cy = cy - 210 + bounce
            body_top = cy - 180 + bounce
            body_bottom = cy - 96 + bounce
            shoulder_y = cy - 160 + bounce
            hip_y = cy - 96 + bounce

        belt_top = body_bottom - 6
        belt_bottom = body_bottom + 6

        # Draw Back Leg
        if state == "walk":
            walk_cycle = math.sin(self.total_steps * 0.6)
            draw.polygon([cx + tilt_x, hip_y, cx - facing * 24 - walk_cycle * 24, cy, cx - facing * 8 - walk_cycle * 24, cy, cx + tilt_x, hip_y], fill=gi_color)
        elif is_crouching:
            draw.rounded_rectangle([cx - 36, hip_y, cx + 36, cy], radius=12, fill=gi_color)
        else:
            draw_rect(cx - facing * 20 + tilt_x, hip_y, cx - facing * 4 + tilt_x, cy, fill=gi_color)

        # Draw Body
        draw.rounded_rectangle([cx - 24 + tilt_x, body_top, cx + 24 + tilt_x, body_bottom], radius=6, fill=gi_color)

        # Draw Belt
        draw_rect(cx - 26 + tilt_x, belt_top, cx + 26 + tilt_x, belt_bottom, fill=belt_color)
        # Belt Tails
        draw.polygon([cx + tilt_x, belt_bottom, cx + facing * 12 + tilt_x, belt_bottom + 30, cx + facing * 6 + tilt_x, belt_bottom + 32, cx - facing * 6 + tilt_x, belt_bottom], fill=belt_color)

        # Draw Front Leg
        if state == "walk":
            walk_cycle = math.sin(self.total_steps * 0.6)
            draw.polygon([cx + tilt_x, hip_y, cx + facing * 8 + walk_cycle * 24, cy, cx + facing * 24 + walk_cycle * 24, cy, cx + tilt_x, hip_y], fill=gi_color)
        elif state == "kick":
            # Extended kick leg
            draw_rect(cx + facing * 12, hip_y - 10, cx + facing * 117, hip_y + 10, fill=gi_color)
            draw_ellipse(cx + facing * 112, hip_y - 12, cx + facing * 124, hip_y + 12, fill=skin_color)
        elif state == "crouch_kick":
            # Low kick leg
            draw_rect(cx + facing * 12, cy - 24, cx + facing * 117, cy - 4, fill=gi_color)
            draw_ellipse(cx + facing * 112, cy - 26, cx + facing * 124, cy - 2, fill=skin_color)
        elif is_crouching:
            pass  # Back leg handles the crouching lump
        else:
            # Stand front leg
            draw_rect(cx + facing * 4 + tilt_x, hip_y, cx + facing * 20 + tilt_x, cy, fill=gi_color)

        # Draw Back Arm
        if state not in ["punch", "crouch_punch", "fireball"]:
            draw_rect(cx - facing * 26 + tilt_x, shoulder_y, cx - facing * 14 + tilt_x, shoulder_y + 48, fill=gi_color)
            draw_ellipse(cx - facing * 26 + tilt_x, shoulder_y + 44, cx - facing * 14 + tilt_x, shoulder_y + 56, fill=skin_color)

        # Draw Front Arm / Punching
        if state in ["punch", "crouch_punch"]:
            # Punch extension
            draw_rect(cx + facing * 12 + tilt_x, shoulder_y + 10, cx + facing * 87 + tilt_x, shoulder_y + 30, fill=gi_color)
            draw_ellipse(cx + facing * 82 + tilt_x, shoulder_y + 8, cx + facing * 94 + tilt_x, shoulder_y + 32, fill=skin_color)
        elif state == "fireball":
            # Fireball casting: double arm extension
            draw_rect(cx + facing * 12 + tilt_x, shoulder_y + 5, cx + facing * 72 + tilt_x, shoulder_y + 20, fill=gi_color)
            draw_rect(cx + facing * 12 + tilt_x, shoulder_y + 25, cx + facing * 72 + tilt_x, shoulder_y + 40, fill=gi_color)
            draw_ellipse(cx + facing * 68 + tilt_x, shoulder_y + 5, cx + facing * 78 + tilt_x, shoulder_y + 40, fill=skin_color)
        else:
            # Stand front arm
            draw_rect(cx + facing * 18 + tilt_x, shoulder_y, cx + facing * 30 + tilt_x, shoulder_y + 48, fill=gi_color)
            draw_ellipse(cx + facing * 18 + tilt_x, shoulder_y + 44, cx + facing * 30 + tilt_x, shoulder_y + 56, fill=skin_color)

        # Draw Head
        draw_ellipse(cx - 20 + tilt_x, head_cy - 20, cx + 20 + tilt_x, head_cy + 20, fill=skin_color)

        # Draw Hair
        draw.chord([cx - 22 + tilt_x, head_cy - 22, cx + 22 + tilt_x, head_cy + 4], start=180, end=360, fill=hair_color)

        # Draw Headband (Ryu only)
        if headband_color is not None:
            draw_rect(cx - 21 + tilt_x, head_cy - 6, cx + 21 + tilt_x, head_cy - 2, fill=headband_color)
            # Headband tails blowing in wind
            draw.polygon(
                [
                    cx - facing * 20 + tilt_x,
                    head_cy - 4,
                    cx - facing * 40 + tilt_x,
                    head_cy + 10,
                    cx - facing * 35 + tilt_x,
                    head_cy + 15,
                    cx - facing * 20 + tilt_x,
                    head_cy + 2,
                ],
                fill=headband_color,
            )

    def render(self) -> Optional[npt.NDArray[np.uint8]]:
        """Render the environment visual output."""
        return self._render_rgb_array()

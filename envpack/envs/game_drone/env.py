"""A Gymnasium environment for a physically accurate fixed-wing drone flight simulator with 3D wireframe rendering."""

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from absl import logging
from gymnasium import spaces
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import numpy.typing as npt

# Simulation Constants
DT = 0.05                         # 20 Hz simulation rate
MAP_SIZE = 2000.0                 # 2000m x 2000m area
G = 9.81                          # Gravity (m/s^2)
RHO = 1.225                       # Air density at sea level (kg/m^3)

# Drone Physical Parameters
MASS = 5.0                        # kg
WING_AREA = 0.8                   # m^2
C_L0 = 0.2                        # Lift coefficient at zero AoA
C_L_ALPHA = 4.0                   # Lift slope (per radian)
C_D0 = 0.04                       # Parasitic drag coefficient
K_INDUCED = 0.05                  # Induced drag factor (K * C_L^2)
MAX_THRUST = 45.0                 # N (thrust at 100% throttle)
FUEL_CAPACITY = 200.0             # fuel units

# Control Limits (Fly-by-wire rates)
MAX_PITCH_RATE = 0.8              # rad/s
MAX_ROLL_RATE = 1.2               # rad/s
MAX_YAW_RATE = 0.4                # rad/s

# Airstrips/Runways Configuration
RUNWAY_1 = {"x": 200.0, "z": 200.0, "heading": math.pi / 4, "length": 120.0, "width": 15.0} # Start Runway
RUNWAY_2 = {"x": 1600.0, "z": 1600.0, "heading": math.pi / 4, "length": 120.0, "width": 15.0} # Target Runway

# Canvas Dimensions
WIDTH = 400
HEIGHT = 300
HEADER_PX = 30
FOOTER_PX = 20

# Colors
COLOR_BG = (15, 23, 42)           # Dark slate blue
COLOR_HEADER = (30, 41, 59)
COLOR_TEXT = (248, 250, 252)
COLOR_SKY = (30, 41, 59)
COLOR_GROUND = (15, 118, 110)      # Teal/Cyan ground lines
COLOR_RUNWAY = (234, 179, 8)       # Yellow runway markings
COLOR_DRONE = (244, 63, 94)        # Rose/Red drone color
COLOR_HUD = (34, 197, 94)          # Green HUD markings


class GymDroneEnv(gym.Env):
    """A Gymnasium environment for a physically accurate fixed-wing drone simulator."""

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
            self._title_font = ImageFont.truetype(font_file, 12)
            self._stats_font = ImageFont.truetype(font_file, 10)
        except Exception:
            logging.warning("Could not load system sans-serif font. Using default font.")
            self._title_font = ImageFont.load_default()
            self._stats_font = ImageFont.load_default()

        # Action space: [throttle, elevator, aileron, rudder]
        # throttle: [0.0, 1.0]
        # elevator/aileron/rudder: [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )

        # Observation space
        # [x, y, z, vx, vy, vz, pitch, roll, yaw, fuel, dist_to_r2, heading_err]
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
                ),
                "valid_mask": spaces.Box(
                    low=1, high=1, shape=(4,), dtype=np.int8
                ),
            }
        )

        # Background base setup
        self._background = np.full((HEIGHT, WIDTH, 3), COLOR_BG, dtype=np.uint8)
        self._background[0:HEADER_PX, :] = COLOR_HEADER
        self._background[HEIGHT - FOOTER_PX :, :] = COLOR_HEADER

        self.reset()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset flight state variables."""
        super().reset(seed=seed)

        if options is not None and "state" in options:
            state = options["state"]
            self._pos = np.array(state["pos"], dtype=np.float64)
            self._vel = np.array(state["vel"], dtype=np.float64)
            self._pitch = state["pitch"]
            self._roll = state["roll"]
            self._yaw = state["yaw"]
            self._fuel = state["fuel"]
            self._step_count = state["step_count"]
            self._crashed = state["crashed"]
            self._landed = state["landed"]
            return self._create_observation(), {}

        # Reset drone at Runway 1 starting position
        self._pos = np.array([RUNWAY_1["x"], 0.0, RUNWAY_1["z"]], dtype=np.float64)
        # Pointing along the runway heading
        self._pitch = 0.0
        self._roll = 0.0
        self._yaw = RUNWAY_1["heading"]

        # Initially stationary
        self._vel = np.zeros(3, dtype=np.float64)
        self._fuel = FUEL_CAPACITY
        self._step_count = 0
        self._crashed = False
        self._landed = False

        return self._create_observation(), {}

    def _get_terrain_height(self, x: float, z: float) -> float:
        """Procedurally generate landscape height using superposition of harmonic waves."""
        # Ensure flat runway pads
        # Start Runway weight
        d1 = math.hypot(x - RUNWAY_1["x"], z - RUNWAY_1["z"])
        w1 = max(0.0, min(1.0, (150.0 - d1) / 50.0)) if d1 < 150.0 else 0.0

        # Target Runway weight
        d2 = math.hypot(x - RUNWAY_2["x"], z - RUNWAY_2["z"])
        w2 = max(0.0, min(1.0, (150.0 - d2) / 50.0)) if d2 < 150.0 else 0.0

        # Procedural mountain/hill formulas
        h = 80.0 * (
            0.45 * math.sin(0.0035 * x) * math.cos(0.003 * z)
            + 0.25 * math.sin(0.009 * x + 0.004 * z) * math.cos(0.008 * z)
            + 0.15 * math.sin(0.018 * x) * math.sin(0.015 * z)
        )
        # Scale height down to zero near runways
        h_final = h * (1.0 - w1) * (1.0 - w2)
        return max(0.0, h_final)

    def _create_observation(self) -> Dict[str, Any]:
        """Generate observation vector and action valid mask."""
        # Calculate target metrics
        dx = RUNWAY_2["x"] - self._pos[0]
        dz = RUNWAY_2["z"] - self._pos[2]
        dist = math.hypot(dx, dz)
        
        target_yaw = math.atan2(dx, dz)
        heading_err = (target_yaw - self._yaw + math.pi) % (2.0 * math.pi) - math.pi

        obs = np.array(
            [
                self._pos[0],            # X coordinate
                self._pos[1],            # Altitude (Y)
                self._pos[2],            # Z coordinate
                self._vel[0],            # Vx
                self._vel[1],            # Vy
                self._vel[2],            # Vz
                self._pitch,             # Pitch
                self._roll,              # Roll
                self._yaw,               # Yaw
                self._fuel,              # Fuel
                dist,                    # Distance to R2
                heading_err,             # Heading error to R2
            ],
            dtype=np.float32,
        )

        valid_mask = np.ones((4,), dtype=np.int8)

        return {
            "observation": obs,
            "valid_mask": valid_mask,
        }

    def _on_runway(self, x: float, z: float) -> bool:
        """Check if coordinates are within Runway 1 or Runway 2 dimensions."""
        for rwy in [RUNWAY_1, RUNWAY_2]:
            dx = x - rwy["x"]
            dz = z - rwy["z"]
            angle = rwy["heading"]
            # Rotate offset back to runway local frame
            local_x = dx * math.cos(angle) + dz * math.sin(angle)
            local_z = -dx * math.sin(angle) + dz * math.cos(angle)
            if abs(local_x) < rwy["length"] / 2.0 and abs(local_z) < rwy["width"] / 2.0:
                return True
        return False

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Run one physics update step."""
        self._step_count += 1
        throttle, elevator, aileron, rudder = action

        # Fuel consumption
        fuel_consumption = (0.05 + 0.45 * throttle) * DT
        self._fuel = max(0.0, self._fuel - fuel_consumption)

        if self._crashed or self._landed:
            return self._create_observation(), 0.0, True, False, {"state": self._get_state()}

        # 1. Update Attitude kinematics (Fly-by-wire gyroscope rates)
        # Clamped pitch and roll if on the ground rolling
        is_on_rwy = self._on_runway(self._pos[0], self._pos[2])
        ground_y = self._get_terrain_height(self._pos[0], self._pos[2])
        is_grounded = (self._pos[1] - ground_y) <= 0.05

        if is_grounded and is_on_rwy:
            # Allow pitching up to lift off, but keep roll flat and pitch controlled
            self._roll = 0.0
            self._pitch = max(0.0, self._pitch + elevator * MAX_PITCH_RATE * DT)
            self._yaw += rudder * MAX_YAW_RATE * DT
        else:
            self._pitch += elevator * MAX_PITCH_RATE * DT
            self._roll += aileron * MAX_ROLL_RATE * DT
            self._yaw += rudder * MAX_YAW_RATE * DT

        # Normalize angles
        self._pitch = max(-math.pi / 3, min(math.pi / 3, self._pitch))
        self._roll = max(-math.pi / 2, min(math.pi / 2, self._roll))
        self._yaw = (self._yaw + math.pi) % (2.0 * math.pi) - math.pi

        # 2. Flight Dynamics Physics
        # Compute airspeed vector and magnitude
        airspeed = np.linalg.norm(self._vel)

        # Get orientation unit vectors
        # Heading vector (forward nose)
        hx = math.cos(self._pitch) * math.sin(self._yaw)
        hy = math.sin(self._pitch)
        hz = math.cos(self._pitch) * math.cos(self._yaw)
        heading_vec = np.array([hx, hy, hz])

        # Lift vector (perpendicular to wings, tilting with roll)
        fwd = heading_vec
        right = np.array([math.cos(self._yaw), 0.0, -math.sin(self._yaw)])
        up = np.cross(right, fwd)
        
        # Apply roll rotation to the lift direction
        lift_dir = up * math.cos(self._roll) + right * math.sin(self._roll)
        lift_dir /= np.linalg.norm(lift_dir)

        # Thrust Force
        thrust_force = throttle * MAX_THRUST * heading_vec if self._fuel > 0 else np.zeros(3)

        # Compute Angle of Attack (AoA)
        if airspeed > 1.0:
            vel_dir = self._vel / airspeed
            aoa = math.acos(max(-1.0, min(1.0, np.dot(heading_vec, vel_dir))))
            if np.dot(heading_vec, vel_dir) < 0:
                aoa = math.pi - aoa
        else:
            aoa = 0.0
            vel_dir = heading_vec

        # Stall condition: lift breaks down at high AoA
        is_stalled = abs(aoa) > 0.28  # ~16 degrees

        # Lift Force
        c_l = (C_L0 + C_L_ALPHA * aoa) if not is_stalled else 0.05
        lift_mag = 0.5 * RHO * (airspeed**2) * WING_AREA * c_l
        lift_force = lift_mag * lift_dir

        # Drag Force
        c_d = C_D0 + K_INDUCED * (c_l**2)
        drag_mag = 0.5 * RHO * (airspeed**2) * WING_AREA * c_d
        drag_force = -drag_mag * vel_dir

        # Gravity Force
        gravity_force = np.array([0.0, -MASS * G, 0.0])

        # Combine Forces
        total_force = thrust_force + drag_force + lift_force + gravity_force
        
        # Ground reaction and friction when grounded on runway
        if is_grounded and is_on_rwy:
            # Cancel gravity and downward velocity
            if total_force[1] < 0.0:
                total_force[1] = 0.0
            # Apply rolling friction
            friction = -0.15 * self._vel
            total_force += friction

        accel = total_force / MASS

        # 3. Update position and velocity
        self._vel += accel * DT
        
        # Prevent sinking into the runway ground when rolling
        if is_grounded and is_on_rwy:
            self._vel[1] = max(0.0, self._vel[1])
            
        self._pos += self._vel * DT

        # Keep within map bounds
        self._pos[0] = max(0.0, min(MAP_SIZE, self._pos[0]))
        self._pos[2] = max(0.0, min(MAP_SIZE, self._pos[2]))

        # Calculate heights
        ground_y = self._get_terrain_height(self._pos[0], self._pos[2])
        agl = self._pos[1] - ground_y

        # 4. Check Ground Collision & Landing
        reward = 0.1  # Survival reward
        terminated = False

        # Reward lower fuel usage
        reward -= 0.05 * fuel_consumption

        # Terrain proximity penalty (Minimal AGL requirement = 15m)
        # Ignore this penalty when within takeoff/landing range of the runways
        d_runway1 = math.hypot(self._pos[0] - RUNWAY_1["x"], self._pos[2] - RUNWAY_1["z"])
        d_runway2 = math.hypot(self._pos[0] - RUNWAY_2["x"], self._pos[2] - RUNWAY_2["z"])
        
        near_runway = (d_runway1 < 150.0) or (d_runway2 < 150.0)

        if agl < 15.0 and not near_runway:
            # Penalty proportional to terrain closeness
            reward -= 0.2 * (15.0 - agl)

        # Ground collision check
        if agl <= 0.0:
            is_currently_on_rwy = self._on_runway(self._pos[0], self._pos[2])
            if is_currently_on_rwy:
                # Safe touchdown check
                sink_ok = self._vel[1] >= -3.0  # soft landing sink rate
                pitch_ok = abs(self._pitch) < 0.15
                roll_ok = abs(self._roll) < 0.15
                
                if sink_ok and pitch_ok and roll_ok:
                    # Grounded on runway: clamp altitude and vertical velocity
                    self._pos[1] = ground_y
                    self._vel[1] = 0.0
                    self._roll = 0.0
                    
                    # If we touched down on target Runway 2 at safe speed, we won!
                    if d_runway2 < 60.0 and airspeed > 2.0:
                        self._landed = True
                        reward += 100.0  # Success!
                        terminated = True
                else:
                    self._crashed = True
                    reward -= 50.0
                    terminated = True
            else:
                # Crashed elsewhere in the landscape
                self._crashed = True
                reward -= 50.0
                terminated = True


        # Timeout / Max steps check handled by Gymnasium wrapper or max steps limit
        return self._create_observation(), float(reward), terminated, False, {"state": self._get_state()}

    def _get_state(self) -> Dict[str, Any]:
        return {
            "pos": self._pos.tolist(),
            "vel": self._vel.tolist(),
            "pitch": self._pitch,
            "roll": self._roll,
            "yaw": self._yaw,
            "fuel": self._fuel,
            "step_count": self._step_count,
            "crashed": self._crashed,
            "landed": self._landed,
        }

    def render(self) -> np.ndarray:
        """Produce 3D wireframe render and 2D overview map onto canvas."""
        image = Image.fromarray(self._background.copy())
        draw = ImageDraw.Draw(image)

        # Header Text info
        draw.text((10, 8), "FIXED WING AUTOPILOT SIMULATOR", fill=COLOR_TEXT, font=self._title_font)

        # 3D Viewport Drawing (Clip limits: y in [HEADER_PX, HEIGHT-FOOTER_PX])
        self._render_3d_viewport(draw)

        # 2D Map Overlap Drawing (Top Right)
        self._render_2d_map(draw)

        # Bottom stats
        airspeed = np.linalg.norm(self._vel)
        agl = self._pos[1] - self._get_terrain_height(self._pos[0], self._pos[2])
        draw.text(
            (10, HEIGHT - 15),
            f"SPD: {airspeed:4.1f}m/s  |  ALT: {self._pos[1]:4.0f}m (AGL:{agl:3.0f}m)  |  FUEL: {self._fuel:3.0f}u",
            fill=COLOR_TEXT,
            font=self._stats_font,
        )

        return np.array(image, dtype=np.uint8)

    def _render_3d_viewport(self, draw: ImageDraw.Draw) -> None:
        """Render a 3D wireframe representation of the landscape, runway, and drone."""
        # Define Chase Camera Position (Behind the drone)
        hx = math.cos(self._pitch) * math.sin(self._yaw)
        hy = math.sin(self._pitch)
        hz = math.cos(self._pitch) * math.cos(self._yaw)
        heading_vec = np.array([hx, hy, hz])

        # Put camera 35m behind and 8m above drone
        cam_pos = self._pos - 35.0 * heading_vec + np.array([0.0, 8.0, 0.0])
        look_at = self._pos + 15.0 * heading_vec

        # Build View Matrix vectors
        fwd_cam = look_at - cam_pos
        fwd_cam /= np.linalg.norm(fwd_cam)

        right_cam = np.cross(np.array([0.0, 1.0, 0.0]), fwd_cam)
        right_cam /= np.linalg.norm(right_cam)

        up_cam = np.cross(fwd_cam, right_cam)

        # Projection Focal Length
        focal = 220.0
        cx, cy = WIDTH // 2, (HEIGHT - HEADER_PX - FOOTER_PX) // 2 + HEADER_PX

        def project(pt_world: npt.NDArray[np.float64]) -> Optional[Tuple[float, float]]:
            """Project 3D world coordinate to 2D screen coordinate."""
            rel = pt_world - cam_pos
            z_cam = np.dot(rel, fwd_cam)
            if z_cam < 0.5:
                return None
            x_cam = np.dot(rel, right_cam)
            y_cam = np.dot(rel, up_cam)

            sx = cx + (x_cam / z_cam) * focal
            sy = cy - (y_cam / z_cam) * focal  # invert y for screen coordinates
            return sx, sy

        # Clip line boundary inside the viewport area
        def draw_viewport_line(p1, p2, fill_color, width=1):
            if p1 is None or p2 is None:
                return
            # Crop to viewport Y boundary
            y1 = max(HEADER_PX, min(HEIGHT - FOOTER_PX, p1[1]))
            y2 = max(HEADER_PX, min(HEIGHT - FOOTER_PX, p2[1]))
            if HEADER_PX <= p1[1] <= HEIGHT - FOOTER_PX or HEADER_PX <= p2[1] <= HEIGHT - FOOTER_PX:
                draw.line([p1[0], y1, p2[0], y2], fill=fill_color, width=width)

        # 1. Draw Terrain Wireframe
        # Render a 12x12 grid around the drone position
        grid_res = 12
        grid_spacing = 50.0
        center_x = round(self._pos[0] / grid_spacing) * grid_spacing
        center_z = round(self._pos[2] / grid_spacing) * grid_spacing

        projected_grid = {}
        for i in range(-6, 7):
            for j in range(-6, 7):
                gx = center_x + i * grid_spacing
                gz = center_z + j * grid_spacing
                if 0.0 <= gx <= MAP_SIZE and 0.0 <= gz <= MAP_SIZE:
                    gy = self._get_terrain_height(gx, gz)
                    pt_scr = project(np.array([gx, gy, gz]))
                    projected_grid[(i, j)] = pt_scr

        # Draw grid lines
        for i in range(-6, 7):
            for j in range(-6, 7):
                p_curr = projected_grid.get((i, j))
                p_right = projected_grid.get((i + 1, j))
                p_down = projected_grid.get((i, j + 1))
                if p_curr:
                    draw_viewport_line(p_curr, p_right, COLOR_GROUND)
                    draw_viewport_line(p_curr, p_down, COLOR_GROUND)

        # 2. Draw Target Runway 2
        rx, rz = RUNWAY_2["x"], RUNWAY_2["z"]
        r_head = RUNWAY_2["heading"]
        r_len = RUNWAY_2["length"]
        r_wid = RUNWAY_2["width"]

        # Runway corners
        r_fwd = np.array([math.sin(r_head), 0.0, math.cos(r_head)])
        r_right = np.cross(np.array([0.0, 1.0, 0.0]), r_fwd)

        c1 = np.array([rx, self._get_terrain_height(rx, rz), rz]) + r_fwd * (r_len / 2) - r_right * (r_wid / 2)
        c2 = np.array([rx, self._get_terrain_height(rx, rz), rz]) + r_fwd * (r_len / 2) + r_right * (r_wid / 2)
        c3 = np.array([rx, self._get_terrain_height(rx, rz), rz]) - r_fwd * (r_len / 2) + r_right * (r_wid / 2)
        c4 = np.array([rx, self._get_terrain_height(rx, rz), rz]) - r_fwd * (r_len / 2) - r_right * (r_wid / 2)

        s1, s2, s3, s4 = project(c1), project(c2), project(c3), project(c4)
        draw_viewport_line(s1, s2, COLOR_RUNWAY, width=2)
        draw_viewport_line(s2, s3, COLOR_RUNWAY, width=2)
        draw_viewport_line(s3, s4, COLOR_RUNWAY, width=2)
        draw_viewport_line(s4, s1, COLOR_RUNWAY, width=2)

        # 3. Draw a HUD Flight Ladder
        # Center indicator
        draw.line([cx - 8, cy, cx - 2, cy], fill=COLOR_HUD)
        draw.line([cx + 2, cy, cx + 8, cy], fill=COLOR_HUD)
        draw.line([cx, cy - 2, cx, cy + 2], fill=COLOR_HUD)

        # Pitch indicators (moving with roll and pitch)
        pitch_deg = math.degrees(self._pitch)
        roll_deg = math.degrees(self._roll)
        
        # Simple pitch bar draw
        bar_y_offset = (pitch_deg / 20.0) * 80.0
        bx = cx
        by = cy + bar_y_offset

        # Draw a line rotated by the roll angle
        cos_r = math.cos(-self._roll)
        sin_r = math.sin(-self._roll)
        
        rx1, ry1 = -20 * cos_r, -20 * sin_r
        rx2, ry2 = 20 * cos_r, 20 * sin_r
        
        draw.line([bx + rx1, by + ry1, bx + rx2, by + ry2], fill=COLOR_HUD)

    def _render_2d_map(self, draw: ImageDraw.Draw) -> None:
        """Render a 2D overview map showing the drone position and targets."""
        map_size_px = 70
        offset_x = WIDTH - map_size_px - 8
        offset_y = HEADER_PX + 8

        # Draw map outline
        draw.rectangle(
            [offset_x - 1, offset_y - 1, offset_x + map_size_px + 1, offset_y + map_size_px + 1],
            outline=COLOR_TEXT,
            fill=(10, 15, 30),
        )

        def map_coords(wx: float, wz: float) -> Tuple[float, float]:
            sx = offset_x + (wx / MAP_SIZE) * map_size_px
            sy = offset_y + (wz / MAP_SIZE) * map_size_px
            return sx, sy

        # Draw Runway 1 (Green)
        r1x, r1y = map_coords(RUNWAY_1["x"], RUNWAY_1["z"])
        draw.rectangle([r1x - 2, r1y - 2, r1x + 2, r1y + 2], fill=(74, 222, 128))

        # Draw Runway 2 (Red)
        r2x, r2y = map_coords(RUNWAY_2["x"], RUNWAY_2["z"])
        draw.rectangle([r2x - 2, r2y - 2, r2x + 2, r2y + 2], fill=(248, 113, 113))

        # Draw Drone indicator (arrow based on yaw)
        dx, dy = map_coords(self._pos[0], self._pos[2])
        # Simple triangle arrow pointing along yaw heading
        yaw_vec = np.array([math.sin(self._yaw), -math.cos(self._yaw)])  # top-down 2D vector
        yaw_perp = np.array([-yaw_vec[1], yaw_vec[0]])
        
        p1 = np.array([dx, dy]) + 4.0 * yaw_vec
        p2 = np.array([dx, dy]) - 3.0 * yaw_vec + 2.0 * yaw_perp
        p3 = np.array([dx, dy]) - 3.0 * yaw_vec - 2.0 * yaw_perp

        draw.polygon([tuple(p1), tuple(p2), tuple(p3)], fill=COLOR_DRONE)

    def close(self) -> None:
        pass

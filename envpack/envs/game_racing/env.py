"""A Gymnasium environment for two-player manual transmission car racing with procedural tracks."""

import copy
from typing import Any, Tuple, Dict, Optional, List

import gymnasium as gym
from absl import logging
from gymnasium import spaces
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import numpy.typing as npt

# Physical Constants (BMW M4 Inspired)
MASS = 1600.0            # kg
WHEEL_RADIUS = 0.34      # meters
FINAL_DRIVE = 3.46
GEAR_RATIOS = [0.0, 4.11, 2.32, 1.54, 1.18, 1.00, 0.85]  # Neutral, 1st to 6th gear
MAX_STEER = 0.4          # radians
MAX_BRAKE = 12000.0      # N
DRAG_COEFF = 0.32
ROLLING_RES = 220.0
GRAVITY = 9.81
IDLE_RPM = 800.0
REDLINE_RPM = 7200.0
TRACK_WIDTH = 40.0       # pixels (width of road)

# Canvas constants
WIDTH = 500
HEIGHT = 500
HEADER_PX = 70
FOOTER_PX = 30
PADDING_PX = 8
CANVAS_SIZE = (WIDTH, HEIGHT + HEADER_PX + FOOTER_PX)

# Visual colors
COLOR_GRASS = (20, 45, 20)
COLOR_ASPHALT = (45, 45, 50)
COLOR_CURB_RED = (231, 76, 60)
COLOR_CURB_WHITE = (236, 240, 241)
COLOR_P1 = (52, 172, 224)           # Cyan
COLOR_P2 = (255, 107, 129)          # Coral/Pink
COLOR_TEXT = (248, 250, 252)
COLOR_REDLINE = (235, 77, 75)


def get_engine_torque(rpm: float) -> float:
    """Returns engine torque based on M4 torque curve."""
    if rpm < 1000.0:
        return 150.0 + (rpm / 1000.0) * 150.0
    elif rpm < 1850.0:
        return 300.0 + ((rpm - 1000.0) / 850.0) * 250.0
    elif rpm < 5800.0:
        return 550.0
    elif rpm < 7200.0:
        return 550.0 - ((rpm - 5800.0) / 1400.0) * 100.0
    else:
        return 0.0  # Rev-limiter cutoff


class GymRacingEnv(gym.Env):
    """Gymnasium environment for 2-player manual racing with RWD drift dynamics."""

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
            self._title_font = ImageFont.truetype(font_file, 13)
            self._stats_font = ImageFont.truetype(font_file, 10)
        except Exception:
            logging.warning("Could not load system sans-serif font. Using default font.")
            self._title_font = ImageFont.load_default()
            self._stats_font = ImageFont.load_default()

        # Spaces
        # Action space: Dict of [steering, throttle/brake] in Box(-1.0, 1.0) and gear action in Discrete(3)
        # Gear action: 0: hold, 1: shift down, 2: shift up
        self.action_space = spaces.Dict(
            {
                "p1_steer_throttle": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                "p1_gear": spaces.Discrete(3),
                "p2_steer_throttle": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                "p2_gear": spaces.Discrete(3),
            }
        )

        # Observation Space
        # p1: [x, y, vx, vy, theta, gear, rpm, progress]
        # p2: [x, y, vx, vy, theta, gear, rpm, progress]
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-1000.0, high=1000.0, shape=(16,), dtype=np.float32
                ),
                "total_score": spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
            }
        )

        self.reset()

    def _generate_track(self) -> None:
        """Procedurally generate a smooth closed-loop asphalt track."""
        num_points = 8
        radius_base = 150.0
        offset_range = 35.0
        center_x = WIDTH / 2
        center_y = HEIGHT / 2

        control_points = []
        for i in range(num_points):
            angle = i * (2.0 * np.pi / num_points)
            r = radius_base + self.np_random.uniform(-offset_range, offset_range)
            px = center_x + r * np.cos(angle)
            py = center_y + r * np.sin(angle)
            control_points.append([px, py])

        # Interpolate closed loop using cubic splines
        control_points = np.array(control_points)
        closed_pts = np.vstack([control_points, control_points[0:3]])

        track_points = []
        for i in range(num_points):
            p0, p1, p2, p3 = closed_pts[i], closed_pts[i+1], closed_pts[i+2], closed_pts[i+3]
            for t in np.linspace(0.0, 1.0, 40, endpoint=False):
                # Catmull-Rom basis spline
                pt = 0.5 * (
                    (2.0 * p1) +
                    (-p0 + p2) * t +
                    (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * (t**2) +
                    (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * (t**3)
                )
                track_points.append(pt)

        self._track_spline = np.array(track_points)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset track and vehicle positions."""
        super().reset(seed=seed)

        if options is not None and "state" in options:
            state = options["state"]
            self._track_spline = np.array(state["track_spline"]).copy()
            
            self._p1_x, self._p1_y = state["p1_pos"]
            self._p1_vx, self._p1_vy = state["p1_vel"]
            self._p1_theta = state["p1_theta"]
            self._p1_gear = state["p1_gear"]
            self._p1_rpm = state["p1_rpm"]
            self._p1_progress = state["p1_progress"]

            self._p2_x, self._p2_y = state["p2_pos"]
            self._p2_vx, self._p2_vy = state["p2_vel"]
            self._p2_theta = state["p2_theta"]
            self._p2_gear = state["p2_gear"]
            self._p2_rpm = state["p2_rpm"]
            self._p2_progress = state["p2_progress"]

            self._steps = state["steps"]
            self._scores = np.array(state["scores"], dtype=np.float32)
            self._p1_skids = copy.deepcopy(state["p1_skids"])
            self._p2_skids = copy.deepcopy(state["p2_skids"])

            return self._create_observation(), {}

        self._generate_track()

        # Place cars at starting line (track_spline[0]) facing next point (track_spline[1])
        start_pt = self._track_spline[0]
        next_pt = self._track_spline[1]
        
        diff = next_pt - start_pt
        heading = np.arctan2(diff[1], diff[0])
        normal = np.array([-diff[1], diff[0]])
        normal = normal / np.linalg.norm(normal)

        # Place cars offset laterally from spline centerline
        p1_start = start_pt + normal * 10.0
        p2_start = start_pt - normal * 10.0

        # P1 states
        self._p1_x, self._p1_y = p1_start
        self._p1_vx, self._p1_vy = 0.0, 0.0
        self._p1_theta = heading
        self._p1_gear = 1
        self._p1_rpm = IDLE_RPM
        self._p1_progress = 0.0

        # P2 states
        self._p2_x, self._p2_y = p2_start
        self._p2_vx, self._p2_vy = 0.0, 0.0
        self._p2_theta = heading
        self._p2_gear = 1
        self._p2_rpm = IDLE_RPM
        self._p2_progress = 0.0

        self._steps = 0
        self._scores = np.zeros(2, dtype=np.float32)
        
        # Keep track of drift skids for visual rendering
        self._p1_skids: List[Tuple[float, float]] = []
        self._p2_skids: List[Tuple[float, float]] = []

        return self._create_observation(), {}

    def _get_distance_to_track(self, x: float, y: float) -> Tuple[float, int]:
        """Returns distance to closest track point and its index."""
        dists = np.linalg.norm(self._track_spline - np.array([x, y]), axis=1)
        min_idx = int(np.argmin(dists))
        return dists[min_idx], min_idx

    def _create_observation(self) -> Dict[str, Any]:
        """Create normalized observation vector."""
        obs = np.array(
            [
                self._p1_x / WIDTH, self._p1_y / HEIGHT,
                self._p1_vx / 50.0, self._p1_vy / 50.0,
                self._p1_theta / np.pi, float(self._p1_gear) / 6.0,
                self._p1_rpm / REDLINE_RPM, self._p1_progress,

                self._p2_x / WIDTH, self._p2_y / HEIGHT,
                self._p2_vx / 50.0, self._p2_vy / 50.0,
                self._p2_theta / np.pi, float(self._p2_gear) / 6.0,
                self._p2_rpm / REDLINE_RPM, self._p2_progress,
            ],
            dtype=np.float32,
        )
        return {
            "observation": obs,
            "total_score": self._scores.copy(),
        }

    def _update_vehicle(
        self,
        x: float, y: float, vx: float, vy: float, theta: float,
        gear: int, rpm: float, action_steer_throttle: npt.NDArray[np.float32],
        action_gear: int, skids_list: List[Tuple[float, float]]
    ) -> Tuple[float, float, float, float, float, int, float, bool]:
        """Updates vehicle dynamics for one step (RWD oversteer, manual gearbox, RPM torque)."""
        dt = 0.05
        steer_input, throttle_brake = action_steer_throttle

        # 1. Gearbox Shifting
        if action_gear == 1:
            gear = max(1, gear - 1)
        elif action_gear == 2:
            gear = min(6, gear + 1)

        # 2. Local velocities
        # Rotate world velocity to local frame
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        vx_local = vx * cos_t + vy * sin_t
        vy_local = -vx * sin_t + vy * cos_t

        # 3. RPM & Torque Calculations
        # Engine RPM is proportional to wheel speed in gear
        wheel_speed = max(0.0, vx_local) / WHEEL_RADIUS
        engine_speed_rad = wheel_speed * FINAL_DRIVE * GEAR_RATIOS[gear]
        rpm_from_wheel = engine_speed_rad * (60.0 / (2.0 * np.pi))
        
        # Clutch slip logic when starting from standstill
        clutch_slip = 1.0
        if rpm_from_wheel < IDLE_RPM:
            rpm = IDLE_RPM
            # Slip the clutch at low speed to transfer partial torque without stalling
            clutch_slip = max(0.1, rpm_from_wheel / IDLE_RPM)
        else:
            rpm = min(rpm_from_wheel, 7500.0)

        # Overrev damage / engine drag if redlining
        overrev = False
        if rpm > REDLINE_RPM:
            overrev = True

        torque = get_engine_torque(rpm)

        # 4. Forces
        # Drive force (RWD drive torque)
        throttle = max(0.0, throttle_brake)
        drive_torque = torque * GEAR_RATIOS[gear] * FINAL_DRIVE * throttle * clutch_slip
        F_drive = drive_torque / WHEEL_RADIUS

        # Braking force
        brake = max(0.0, -throttle_brake)
        F_brake = -brake * MAX_BRAKE * np.sign(vx_local) if abs(vx_local) > 0.1 else 0.0

        # Aerodynamic drag & rolling resistance
        F_drag = -DRAG_COEFF * vx_local * abs(vx_local) - ROLLING_RES * np.sign(vx_local)
        F_long = F_drive + F_brake + F_drag

        # 5. Lateral Dynamics & Slip (RWD Drift)
        # Determine grip coefficient depending on whether vehicle is on/off track
        dist_to_track, _ = self._get_distance_to_track(x, y)
        on_track = dist_to_track <= TRACK_WIDTH / 2.0
        grip_coeff = 1.0 if on_track else 0.4

        # Max grip circle for rear wheels
        F_grip_max = (MASS * 0.5) * GRAVITY * grip_coeff
        # Drive force consumes longitudinal traction. Remaining lateral grip available:
        F_lateral_max = np.sqrt(max(0.0, F_grip_max**2 - (F_drive * 0.5)**2))

        # Lateral force needed to steer
        steering_angle = steer_input * MAX_STEER
        
        # Calculate yaw rate matching steering angle
        target_yaw_rate = steering_angle * vx_local * 0.15
        
        # If overrevving, reduce engine torque & engine slows down car
        if overrev:
            F_long -= 3000.0

        # Calculate centripetal force required to hold path
        F_lat_required = MASS * vx_local * target_yaw_rate
        sliding = False

        if abs(F_lat_required) > F_lateral_max:
            # RWD loss of traction -> sliding sideways (drift!)
            sliding = True
            slide_accel = (F_lat_required - np.sign(F_lat_required) * F_lateral_max) / MASS
            vy_local += slide_accel * dt
            # Spin/Oversteer effect rotates the vehicle heading
            yaw_rate = target_yaw_rate + np.sign(steering_angle) * abs(vy_local) * 0.15
            
            # Record skid mark at rear axle
            skids_list.append((x - cos_t * 12.0, y - sin_t * 12.0))
            if len(skids_list) > 100:
                skids_list.pop(0)
        else:
            vy_local = 0.0
            yaw_rate = target_yaw_rate

        # 6. Apply updates
        theta += yaw_rate * dt
        # Clamp theta to [-pi, pi]
        theta = (theta + np.pi) % (2.0 * np.pi) - np.pi

        vx_local += (F_long / MASS) * dt
        if vx_local < -2.0:  # Cap reverse speed
            vx_local = -2.0

        # Transform local coordinates back to world coordinates
        vx_new = vx_local * cos_t - vy_local * sin_t
        vy_new = vx_local * sin_t + vy_local * cos_t

        x_new = x + vx_new * dt
        y_new = y + vy_new * dt

        # Wrap around grid boundaries
        x_new = np.clip(x_new, 5.0, WIDTH - 5.0)
        y_new = np.clip(y_new, 5.0, HEIGHT - 5.0)

        return x_new, y_new, vx_new, vy_new, theta, gear, rpm, sliding

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Advance both racing vehicles by one time step."""
        p1_st = action["p1_steer_throttle"]
        p1_g = action["p1_gear"]
        p2_st = action["p2_steer_throttle"]
        p2_g = action["p2_gear"]

        self._steps += 1

        # Update P1 Vehicle
        self._p1_x, self._p1_y, self._p1_vx, self._p1_vy, self._p1_theta, self._p1_gear, self._p1_rpm, p1_slide = (
            self._update_vehicle(
                self._p1_x, self._p1_y, self._p1_vx, self._p1_vy, self._p1_theta,
                self._p1_gear, self._p1_rpm, p1_st, p1_g, self._p1_skids
            )
        )

        # Update P2 Vehicle
        self._p2_x, self._p2_y, self._p2_vx, self._p2_vy, self._p2_theta, self._p2_gear, self._p2_rpm, p2_slide = (
            self._update_vehicle(
                self._p2_x, self._p2_y, self._p2_vx, self._p2_vy, self._p2_theta,
                self._p2_gear, self._p2_rpm, p2_st, p2_g, self._p2_skids
            )
        )

        # Calculate race progress index along track points
        # Track spline has 320 points (8 control points * 40 interpolation steps)
        _, p1_idx = self._get_distance_to_track(self._p1_x, self._p1_y)
        _, p2_idx = self._get_distance_to_track(self._p2_x, self._p2_y)

        # Normalize progress as fraction of lap: [0.0, 1.0]
        self._p1_progress = float(p1_idx) / len(self._track_spline)
        self._p2_progress = float(p2_idx) / len(self._track_spline)

        # Win conditions
        # Completion of 1 lap: index reaches length of track
        p1_wins = p1_idx >= len(self._track_spline) - 2 and self._p1_progress > 0.9
        p2_wins = p2_idx >= len(self._track_spline) - 2 and self._p2_progress > 0.9

        reward = 0.0
        terminated = False

        if p1_wins or p2_wins:
            terminated = True
            if p1_wins and p2_wins:
                # Joint tie
                reward = 0.0
            elif p1_wins:
                reward = 10.0
                self._scores[0] += 1.0
            else:
                reward = -10.0
                self._scores[1] += 1.0
        else:
            # Small step reward encouraging forward lap progress from P1 perspective
            # Difference in relative progress
            reward = (self._p1_progress - self._p2_progress) * 0.1
            
            # Penalize going off-track
            p1_dist, _ = self._get_distance_to_track(self._p1_x, self._p1_y)
            p2_dist, _ = self._get_distance_to_track(self._p2_x, self._p2_y)
            if p1_dist > TRACK_WIDTH / 2.0:
                reward -= 0.01
            if p2_dist > TRACK_WIDTH / 2.0:
                reward += 0.01  # Positive because P2 goes off-track (P1 perspective)

        truncated = self._steps >= 1000
        return self._create_observation(), float(reward), terminated, truncated, {"state": self._get_state()}

    def _get_state(self) -> Dict[str, Any]:
        """Return environment internal state dictionary."""
        return {
            "track_spline": self._track_spline.tolist(),
            "p1_pos": [self._p1_x, self._p1_y],
            "p1_vel": [self._p1_vx, self._p1_vy],
            "p1_theta": self._p1_theta,
            "p1_gear": self._p1_gear,
            "p1_rpm": self._p1_rpm,
            "p1_progress": self._p1_progress,
            "p2_pos": [self._p2_x, self._p2_y],
            "p2_vel": [self._p2_vx, self._p2_vy],
            "p2_theta": self._p2_theta,
            "p2_gear": self._p2_gear,
            "p2_rpm": self._p2_rpm,
            "p2_progress": self._p2_progress,
            "steps": self._steps,
            "scores": self._scores.tolist(),
            "p1_skids": copy.deepcopy(self._p1_skids),
            "p2_skids": copy.deepcopy(self._p2_skids),
        }

    def _draw_hud_bar(self, draw: ImageDraw.ImageDraw, x: int, y: int, rpm: float, gear: int, color: Tuple[int, int, int], label: str) -> None:
        """Draw an RPM tachometer bar and gear indicator inside the header HUD."""
        # RPM bar: 150px width
        bar_w = 140
        bar_h = 10
        rpm_ratio = max(0.0, min(rpm / REDLINE_RPM, 1.0))
        fill_w = int(bar_w * rpm_ratio)

        # Draw background bar outline
        draw.rectangle([x, y + 14, x + bar_w, y + 14 + bar_h], outline=(100, 100, 100), width=1)
        
        # Color changes to red if redlining
        bar_color = COLOR_REDLINE if rpm > REDLINE_RPM else color
        draw.rectangle([x + 1, y + 15, x + fill_w - 1, y + 13 + bar_h], fill=bar_color)

        # Draw redline mark at 7000 RPM (around 97% of width)
        redline_x = x + int(bar_w * (7000.0 / REDLINE_RPM))
        draw.line([(redline_x, y + 14), (redline_x, y + 14 + bar_h)], fill=COLOR_REDLINE, width=1)

        # Text labels
        speed_kmh = int(np.linalg.norm([self._p1_vx if label=="P1" else self._p2_vx, self._p1_vy if label=="P2" else self._p2_vy]) * 3.6)
        draw.text(
            (x, y),
            f"{label} - G:{gear} - {int(rpm)} RPM - {speed_kmh} km/h",
            fill=COLOR_TEXT,
            font=self._stats_font,
        )

    def render(self) -> Optional[npt.NDArray[np.uint8]]:
        """Render the environment to an RGB canvas array."""
        # Make a copy of base static background template
        canvas = Image.new("RGB", CANVAS_SIZE, COLOR_GRASS)
        draw = ImageDraw.Draw(canvas)

        # Draw track asphalt base
        # To draw a smooth thick spline, we draw overlapping circles
        for pt in self._track_spline:
            tx, ty = int(pt[0]), int(pt[1] + HEADER_PX)
            draw.ellipse(
                [tx - int(TRACK_WIDTH/2), ty - int(TRACK_WIDTH/2), tx + int(TRACK_WIDTH/2), ty + int(TRACK_WIDTH/2)],
                fill=COLOR_ASPHALT,
            )

        # Draw white dashed centerline on track
        for i, pt in enumerate(self._track_spline):
            if i % 3 == 0:
                tx, ty = int(pt[0]), int(pt[1] + HEADER_PX)
                draw.ellipse([tx - 1, ty - 1, tx + 1, ty + 1], fill=(200, 200, 200))

        # Draw red/white curbs on outside bends
        # If distance from center is large or coordinates change quickly, we place curbs
        for i in range(len(self._track_spline)):
            if i % 4 in (0, 1):
                pt = self._track_spline[i]
                next_pt = self._track_spline[(i + 1) % len(self._track_spline)]
                diff = next_pt - pt
                normal = np.array([-diff[1], diff[0]])
                norm_len = np.linalg.norm(normal)
                if norm_len > 0:
                    normal = normal / norm_len
                    # Draw on outside border
                    cx, cy = pt + normal * (TRACK_WIDTH / 2.0)
                    cx, cy = int(cx), int(cy + HEADER_PX)
                    curb_color = COLOR_CURB_RED if i % 8 < 4 else COLOR_CURB_WHITE
                    draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=curb_color)

        # Draw skidmarks
        for sx, sy in self._p1_skids:
            draw.ellipse([int(sx), int(sy + HEADER_PX) - 1, int(sx) + 1, int(sy + HEADER_PX) + 1], fill=(10, 10, 10))
        for sx, sy in self._p2_skids:
            draw.ellipse([int(sx), int(sy + HEADER_PX) - 1, int(sx) + 1, int(sy + HEADER_PX) + 1], fill=(10, 10, 10))

        # Draw static HUD header area
        draw.rectangle([0, 0, CANVAS_SIZE[0] - 1, HEADER_PX - 1], fill=(25, 25, 30))
        draw.rectangle(
            [0, CANVAS_SIZE[1] - FOOTER_PX, CANVAS_SIZE[0] - 1, CANVAS_SIZE[1] - 1],
            fill=(20, 20, 25),
        )

        # Header Title
        draw.text(
            (PADDING_PX + 5, 12),
            "RACING DUEL (RWD Manual)",
            fill=COLOR_TEXT,
            font=self._title_font,
        )

        # Draw HUD bars
        # P1 HUD (left)
        self._draw_hud_bar(draw, 20, 36, self._p1_rpm, self._p1_gear, COLOR_P1, "P1")
        # P2 HUD (right)
        self._draw_hud_bar(draw, CANVAS_SIZE[0] - 160, 36, self._p2_rpm, self._p2_gear, COLOR_P2, "P2")

        # Draw vehicles as oriented rectangles with tail lights
        def draw_vehicle(x: float, y: float, theta: float, color: Tuple[int, int, int]) -> None:
            # Coordinates
            vx, vy = x, y + HEADER_PX
            l, w = 14.0, 7.0  # Car length & width

            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            # 4 corners of car body
            f_right = (vx + cos_t * (l/2) - sin_t * (w/2), vy + sin_t * (l/2) + cos_t * (w/2))
            f_left  = (vx + cos_t * (l/2) + sin_t * (w/2), vy + sin_t * (l/2) - cos_t * (w/2))
            b_right = (vx - cos_t * (l/2) - sin_t * (w/2), vy - sin_t * (l/2) + cos_t * (w/2))
            b_left  = (vx - cos_t * (l/2) + sin_t * (w/2), vy - sin_t * (l/2) - cos_t * (w/2))

            draw.polygon([f_right, f_left, b_left, b_right], fill=color, outline=(255, 255, 255))
            
            # Red tail lights on rear corners
            draw.ellipse([int(b_right[0]) - 1, int(b_right[1]) - 1, int(b_right[0]) + 1, int(b_right[1]) + 1], fill=(255, 0, 0))
            draw.ellipse([int(b_left[0]) - 1, int(b_left[1]) - 1, int(b_left[0]) + 1, int(b_left[1]) + 1], fill=(255, 0, 0))

        # P1 (Cyan)
        draw_vehicle(self._p1_x, self._p1_y, self._p1_theta, COLOR_P1)
        # P2 (Coral/Pink)
        draw_vehicle(self._p2_x, self._p2_y, self._p2_theta, COLOR_P2)

        # Footer statistics
        draw.text(
            (PADDING_PX + 5, CANVAS_SIZE[1] - FOOTER_PX // 2),
            f"Steps: {self._steps} | P1 Lap: {int(self._p1_progress * 100)}% | P2 Lap: {int(self._p2_progress * 100)}%",
            fill=(180, 180, 180),
            font=self._stats_font,
            anchor="lm",
        )

        return np.array(canvas, dtype=np.uint8)

    def close(self) -> None:
        """Close the environment."""
        pass

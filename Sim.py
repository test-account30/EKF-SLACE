import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from utils import normalize_angle
from Config import *

"""
SLACE.py
Description:
    Robot Sim Engine
Author:
    Matthew Allen
Date Created:
    May 27, 2026
"""

def get_covariance_ellipse(mean: np.ndarray, cov: np.ndarray, scale: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """Generates X, Y points for a 2D covariance ellipse."""
    vals, vecs = np.linalg.eigh(cov)
    angle = np.arctan2(vecs[1, 0], vecs[0, 0])
    width, height = 2 * scale * np.sqrt(np.maximum(vals, 0))
    
    t = np.linspace(0, 2*np.pi, 30)
    ellipse_x = (width / 2) * np.cos(t)
    ellipse_y = (height / 2) * np.sin(t)
    
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = np.dot(np.vstack([ellipse_x, ellipse_y]).T, R.T)
    return mean[0] + rotated[:, 0], mean[1] + rotated[:, 1]

def compute_frenet_frame(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Computes line fit, normal, b_meas, and theta_meas from local points via SVD."""
    mean_pt = np.mean(pts, axis=0)
    _, _, Vh = np.linalg.svd(pts - mean_pt)
    t = Vh[0] if Vh[0, 0] >= 0 else -Vh[0]
    n = np.array([-t[1], t[0]])
    b = -np.dot(mean_pt, n)
    theta = -np.arctan2(t[1], t[0])
    return t, n, b, theta


class Sim:
    def __init__(self, config: SimConfig, track_data: dict):
        self.cfg = config
        self.dt = 1 / self.cfg.imu_update_rate

        self.center = track_data["center"]
        self.left = track_data["left"]
        self.right = track_data["right"]
        self.width = track_data["width"]
        self.s = track_data["s"]

        # initialise pose on centreline
        self.true_pose = np.array([
            self.center[0, 0],
            self.center[0, 1],
            np.pi / 2
        ])

        self.closest_idx = 0
        self.steps = 0
        self.prev_v = np.array([0.0, 0.0])

        self.odom_skip = int(1 / (self.dt * self.cfg.odom_update_rate))
        self.cam_skip = int(1 / (self.dt * self.cfg.map_update_rate))

    def step(self, cmd_vx, cmd_vy, cmd_w):
        """Steps physics using provided commands, returns noisy odometry."""

        dists = np.linalg.norm(self.center - self.true_pose[:2], axis=1)
        self.closest_idx = np.argmin(dists)
        vx_act = cmd_vx + np.random.normal(0, self.cfg.true_kine_noise[0])
        vy_act = cmd_vy + np.random.normal(0, self.cfg.true_kine_noise[0])
        w_act  = cmd_w  + np.random.normal(0, self.cfg.true_kine_noise[1])

        theta = self.true_pose[2]

        self.true_pose[0] += (vx_act * np.cos(theta) - vy_act * np.sin(theta)) * self.dt
        self.true_pose[1] += (vx_act * np.sin(theta) + vy_act * np.cos(theta)) * self.dt
        self.true_pose[2] = normalize_angle(theta + w_act * self.dt)

        v_now = np.array([vx_act, vy_act])
        accel_body = (v_now - self.prev_v) / self.dt
        self.prev_v = v_now

        accel_imu = accel_body + np.random.normal(0, self.cfg.imu_accel_noise, 2)

        imu_meas = IMUMeasurement(
            acc_x=accel_imu[0],
            acc_y=accel_imu[1],
            omega=w_act + np.random.normal(0, self.cfg.imu_gyro_noise),
            dt=self.dt
        )

        odom_meas = None
        if self.steps % self.odom_skip == 0:
            odom_meas = OdomMeasurement(
                vx_enc=vx_act + np.random.normal(0, self.cfg.odom_noise[0]),
                vy_enc=vy_act + np.random.normal(0, self.cfg.odom_noise[0]),
                w_enc=w_act + np.random.normal(0, self.cfg.odom_noise[1])
            )

        self.steps += 1

        return imu_meas, odom_meas

    @property
    def is_camera_ready(self) -> bool:
        return self.steps % self.cam_skip == 0

class SimCam:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.temp_count = 0

    def get_observations(self) -> Observation:
        """Generates synthetic local observations with width attached."""

        idxs = (self.sim.closest_idx + np.array(self.sim.cfg.vision_lookahead)) % len(self.sim.center)
        pts = self.sim.center[idxs]
        widths = self.sim.width[idxs]

        self.temp_count += 1

        dx = pts[:, 0] - self.sim.true_pose[0]
        dy = pts[:, 1] - self.sim.true_pose[1]

        c, s = np.cos(self.sim.true_pose[2]), np.sin(self.sim.true_pose[2])

        x_local = dx * c + dy * s
        y_local = -dx * s + dy * c

        x_local += np.random.normal(0, self.sim.cfg.vision_noise_std, len(dx))
        y_local += np.random.normal(0, self.sim.cfg.vision_noise_std, len(dy))
        widths += np.random.normal(0, self.sim.cfg.vision_noise_std, len(widths))

        local_pts = np.column_stack([x_local, y_local, widths])


        dist_to_start = np.linalg.norm(self.sim.true_pose[:2] - self.sim.center[0])

        trigger_lc = (
            dist_to_start < 0.5
            and self.sim.steps > 5000
            and (self.sim.true_pose[1] > 0)
        )

        return Observation(
            local_points=local_pts,
            loop_closure_triggered=trigger_lc
        )


class LiveVisualizer:
    def __init__(self, gt_track, map_skip_val: int = 10, record_video: bool = False, video_path: str = "slace.mp4"):
        self.map_skip_val = map_skip_val

        if isinstance(gt_track, np.lib.npyio.NpzFile):
            self.center = gt_track["center"]
            self.width = gt_track["width"]
        else:
            self.center = gt_track
            self.width = np.ones(len(gt_track)) * 0.5

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 7))

        self.ax.plot(self.center[:, 0], self.center[:, 1],
                     'k--', alpha=0.3, label='Track centreline')

        self.track_patch = None
        self.est_track_patch = None
        
        # Draw ground truth ribbon once at startup
        left, right = self._compute_ribbon(self.center, self.width)
        poly = np.vstack([left, right[::-1]])
        self.track_patch = self.ax.fill(
            poly[:, 0], poly[:, 1],
            color='grey', alpha=0.15, label='GT Track width'
        )[0]

        self.line_true_path, = self.ax.plot([], [], 'g-', linewidth=2, label='True Path')
        self.line_est_path, = self.ax.plot([], [], 'r--', linewidth=1.5, label='Est. Path')
        self.line_spline, = self.ax.plot([], [], 'b-o', markersize=4, label='EKF Map Spline')
        
        # --- New MPC Planned Path Artist ---
        self.line_mpc_path, = self.ax.plot([], [], color='orange', linestyle='-', linewidth=2.5, label='MPC Plan')
        
        self.scatter_obs = self.ax.scatter([], [], c='magenta', s=15, alpha=0.7)

        self.robot_true_dot, = self.ax.plot([], [], 'go', markersize=8)
        self.robot_est_dot, = self.ax.plot([], [], 'ro', markersize=8)

        self.ellipse_artists = []

        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle=':', alpha=0.6)
        self.ax.set_title('EKF-SLACE (PoC)')
        self.ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))

        self.true_path_x, self.true_path_y = [], []
        self.est_path_x, self.est_path_y = [], []

    def _compute_ribbon(self, P: np.ndarray, w: np.ndarray):
        """Computes the left and right boundary points of a track ribbon."""
        n = len(P)
        left = np.zeros_like(P)
        right = np.zeros_like(P)

        for i in range(n):
            p_prev = P[i - 1]
            p_next = P[(i + 1) % n]

            t = p_next - p_prev
            t /= np.linalg.norm(t) + 1e-9

            nvec = np.array([-t[1], t[0]])

            left[i] = P[i] - w[i] * nvec
            right[i] = P[i] + w[i] * nvec

        return left, right

    def update(self, step: int, sim, ekf, obs, planned_path: np.ndarray = None):
        # Always append trajectory history at high rate
        self.true_path_x.append(sim.true_pose[0])
        self.true_path_y.append(sim.true_pose[1])
        self.est_path_x.append(ekf.pose[0])
        self.est_path_y.append(ekf.pose[1])

        # Drop out early if it's not a visualization frame
        if step % self.map_skip_val != 0:
            return

        for artist in self.ellipse_artists:
            artist.remove()
        self.ellipse_artists.clear()

        # Update historical paths
        self.line_true_path.set_data(self.true_path_x, self.true_path_y)
        self.line_est_path.set_data(self.est_path_x, self.est_path_y)

        # --- Safe MPC Rollout Rendering ---
        if planned_path is not None and isinstance(planned_path, np.ndarray) and planned_path.ndim == 2 and len(planned_path) > 0:
            self.line_mpc_path.set_data(planned_path[:, 0], planned_path[:, 1])
        else:
            # Clear the old trajectory trace if no plan is supplied or if solver failed
            self.line_mpc_path.set_data([], [])

        # Update Estimated Track Map Spline 
        M_pts = ekf.M.reshape(-1, 2)
        self.line_spline.set_data(M_pts[:, 0], M_pts[:, 1])

        if len(M_pts) > 2: 
            left_est, right_est = self._compute_ribbon(M_pts, ekf.W_est)
            poly_est = np.vstack([left_est, right_est[::-1]])

            if self.est_track_patch is not None:
                self.est_track_patch.remove()

            self.est_track_patch = self.ax.fill(
                poly_est[:, 0], poly_est[:, 1],
                color='blue', alpha=0.15
            )[0]

        # Update Local Point Cloud Measurements
        if len(obs.local_points) > 0:
            c, s = np.cos(ekf.pose[2]), np.sin(ekf.pose[2])
            R = np.array([[c, -s], [s, c]])
            global_obs = ekf.pose[:2] + obs.local_points[:, :2] @ R.T
            self.scatter_obs.set_offsets(global_obs)

        # Update current robot markers
        self.robot_true_dot.set_data([sim.true_pose[0]], [sim.true_pose[1]])
        self.robot_est_dot.set_data([ekf.pose[0]], [ekf.pose[1]])

        # Update Covariance Ellipses
        for i in range(len(M_pts)):
            cov = ekf.Sigma_M[2*i:2*i+2, 2*i:2*i+2]
            ex, ey = get_covariance_ellipse(M_pts[i], cov)
            self.ellipse_artists.append(self.ax.plot(ex, ey, 'b-', alpha=0.2, linewidth=1)[0])

        # Redraw viewport canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
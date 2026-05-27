import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple
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

def normalize_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi

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
    def __init__(self, config: SimConfig, gt_track: np.ndarray):
        self.cfg = config
        self.gt_track = gt_track
        self.true_pose = np.array([gt_track[0,0], gt_track[0,1], -np.pi / 2])
        self.closest_idx = 0
        self.steps = 0

    def step(self) -> Tuple[float, float]:
        """Steps pure pursuit kinematics, returns simulated odometry (v_enc, w_enc)."""
        N = len(self.gt_track)
        idx_window = (np.arange(self.closest_idx - 5, self.closest_idx + 6)) % N
        self.closest_idx = idx_window[np.argmin(np.linalg.norm(self.gt_track[idx_window] - self.true_pose[:2], axis=1))]
        
        target = self.gt_track[(self.closest_idx + self.cfg.lookahead_idx) % N]
        heading_err = normalize_angle(np.arctan2(target[1] - self.true_pose[1], target[0] - self.true_pose[0]) - self.true_pose[2])
        omega = self.cfg.steering_gain * heading_err
        
        v_act = self.cfg.target_v + np.random.normal(0, self.cfg.true_kine_noise[0])
        w_act = omega + np.random.normal(0, self.cfg.true_kine_noise[1])
        
        self.true_pose[0] += v_act * self.cfg.dt * np.cos(self.true_pose[2])
        self.true_pose[1] += v_act * self.cfg.dt * np.sin(self.true_pose[2])
        self.true_pose[2] = normalize_angle(self.true_pose[2] + w_act * self.cfg.dt)
        self.steps += 1
        
        return v_act + np.random.normal(0, self.cfg.odom_noise[0]), w_act + np.random.normal(0, self.cfg.odom_noise[1])

class SimCam:
    def __init__(self, sim: Sim):
        self.sim = sim

    def get_observations(self) -> Observation:
        """Generates synthetic local observations based on true pose."""
        pts = self.sim.gt_track[(self.sim.closest_idx + np.array(self.sim.cfg.vision_lookahead)) % len(self.sim.gt_track)]
        
        dx = pts[:, 0] - self.sim.true_pose[0]
        dy = pts[:, 1] - self.sim.true_pose[1]
        c, s = np.cos(self.sim.true_pose[2]), np.sin(self.sim.true_pose[2])
        
        local_pts = np.vstack((
            (dx * c + dy * s) + np.random.normal(0, self.sim.cfg.vision_noise_std, len(dx)),
            (-dx * s + dy * c) + np.random.normal(0, self.sim.cfg.vision_noise_std, len(dy))
        )).T
        
        dist_to_start = np.linalg.norm(self.sim.true_pose[:2] - self.sim.gt_track[0])
        trigger_lc = dist_to_start < 0.5 and self.sim.steps > 100
        
        return Observation(local_points=local_pts, loop_closure_triggered=trigger_lc)


class LiveVisualizer:
    """Isolated rendering engine for live EKF SLACE monitoring."""
    def __init__(self, gt_track: np.ndarray, map_skip_val: int = 10):
        self.map_skip_val = map_skip_val
        
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        
        # Static background
        self.ax.plot(gt_track[:, 0], gt_track[:, 1], 'k--', alpha=0.3, label='Ground truth track')
        
        # Dynamic lines and scatters
        self.line_true_path, = self.ax.plot([], [], 'g-', linewidth=2, label='Robot true trajectory')
        self.line_est_path, = self.ax.plot([], [], 'r--', linewidth=1.5, label='EKF estimate trajectory')
        self.line_spline, = self.ax.plot([], [], 'b-o', markersize=4, label='Estimated map')
        self.scatter_obs = self.ax.scatter([], [], c='magenta', s=15, alpha=0.7, label='Observations')
        
        # Robot markers
        self.robot_true_dot, = self.ax.plot([], [], 'go', markersize=8)
        self.robot_est_dot, = self.ax.plot([], [], 'ro', markersize=8)
        
        self.ellipse_artists = []
        
        # Setup plot limits and aesthetics
        self.ax.set_xlim(-7, 1.5)
        self.ax.set_ylim(-7, 3)
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle=':', alpha=0.6)
        self.ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1.0), borderaxespad=0.0, fontsize=9)
        self.ax.set_title('EKF SLACE')
        
        # Storage for trajectory trails
        self.true_path_x, self.true_path_y = [], []
        self.est_path_x, self.est_path_y = [], []

    def update(self, step: int, sim: Sim, ekf: EKFSLACE, obs: Observation):
        """Updates the plot data buffers and redraws the canvas."""
        # Update trajectory trails
        self.true_path_x.append(sim.true_pose[0])
        self.true_path_y.append(sim.true_pose[1])
        self.est_path_x.append(ekf.pose[0])
        self.est_path_y.append(ekf.pose[1])
        
        if step % self.map_skip_val != 0:
            return

        # Clear old covariance ellipses
        for artist in self.ellipse_artists:
            artist.remove()
        self.ellipse_artists.clear()
        
        # Update lines
        self.line_true_path.set_data(self.true_path_x, self.true_path_y)
        self.line_est_path.set_data(self.est_path_x, self.est_path_y)
        
        # Update map points
        M_pts = ekf.M.reshape(-1, 2)
        self.line_spline.set_data(M_pts[:, 0], M_pts[:, 1])
        
        # Update observations (convert local to global for plotting)
        if len(obs.local_points) > 0:
            cos_e, sin_e = np.cos(ekf.pose[2]), np.sin(ekf.pose[2])
            R_pose = np.array([[cos_e, -sin_e], [sin_e, cos_e]])
            global_obs = ekf.pose[:2] + obs.local_points @ R_pose.T
            self.scatter_obs.set_offsets(global_obs)
        
        # Update robot positions
        self.robot_true_dot.set_data([sim.true_pose[0]], [sim.true_pose[1]])
        self.robot_est_dot.set_data([ekf.pose[0]], [ekf.pose[1]])
        
        # Draw new covariance ellipses for map nodes
        for i in range(len(M_pts)):
            cov_block = ekf.Sigma_M[2*i:2*i+2, 2*i:2*i+2]
            ex, ey = get_covariance_ellipse(M_pts[i], cov_block)
            ellipse_line, = self.ax.plot(ex, ey, 'b-', alpha=0.3, linewidth=1)
            self.ellipse_artists.append(ellipse_line)
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
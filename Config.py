import numpy as np
from dataclasses import dataclass, field
from typing import List

"""
Config.py
Description:
    Robot config
Author:
    Matthew Allen
Date Created:
    May 27, 2026
"""

# Using fancy python dataclasses 
@dataclass(frozen=True)
class SimConfig:
    dt: float = 0.1
    sim_steps: int = 10000
    # True process/kinematic propagation noise: [vx, w]
    true_kine_noise: np.ndarray = field(default_factory=lambda: np.array([0.00, 0.00]))
    # 3-DOF Holonomic wheel encoder measurement noise: [vx, vy, w]
    odom_noise: np.ndarray = field(default_factory=lambda: np.array([0.003, 0.003, (0.5 * np.pi / 180)]))
    vision_noise_std: float = 0.015
    vision_lookahead: List[int] = field(default_factory=lambda: list(range(-40, 41)))


@dataclass(frozen=True)
class EKFConfig:
    # Initial state covariance diagonal for 6-DOF state vector: [x, y, theta, vx, vy, omega]
    init_p_diag: np.ndarray = field(default_factory=lambda: np.array([
        0.1**2, 0.1**2, (1 * np.pi / 180)**2,  # Pose variances
        0.15**2, 0.15**2, 0.8**2 # Velocity variances
    ]))
    
    # Continuous process noise diagonal for 6-DOF tracking dynamics: [x, y, theta, vx, vy, omega]
    q_diag: np.ndarray = field(default_factory=lambda: np.array([
        0.1**2, 0.1**2, (0.5 * np.pi / 180)**2, # Pose process noise
        0.01**2, 0.01**2, 0.005**2 # IMU acceleration / velocity process noise
    ]))
    
    # Holonomic wheel encoder measurement noise parameters utilized by the EKF update: [vx, vy, w]
    odom_noise: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.01, (5 * np.pi / 180)]))
    
    # Polyline measurement observation noise covariance: [b_intercept, theta_heading]
    r_meas_diag: np.ndarray = field(default_factory=lambda: np.array([0.15**2, (20 * np.pi / 180)**2]))
    
    # Local map landmark point update tracking variances: [Longitudinal, Lateral, Heading]
    point_cov: np.ndarray = field(default_factory=lambda: np.array([0.4**2, 0.25**2, (25 * np.pi / 180)**2]))
    
    # Dynamic Map Augmentation Control Parameters
    augment_dist: float = 0.25
    augment_sigma: float = 0.15
    trigger_dist_edge: float = 0.02
    lane_width_limit: float = 0.5 # m


@dataclass(frozen=True)
class PurePursuitConfig:
    target_v: float = 0.3
    steering_gain: float = 4.0
    lookahead_idx: int = 10


@dataclass
class Observation:
    local_points: np.ndarray
    loop_closure_triggered: bool
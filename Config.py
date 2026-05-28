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
    true_kine_noise: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.02]))
    odom_noise: np.ndarray = field(default_factory=lambda: np.array([0.003, 0.003, (0.5 * np.pi/180)])) # Vx, Vy, W
    vision_noise_std: float = 0.015
    vision_lookahead: List[int] = field(default_factory=lambda: list(range(-40, 41)))
    imu_accel_noise: float = 0.05  # m/s^2
    imu_gyro_noise: float = 0.01   # rad/s

@dataclass(frozen=True)
class EKFConfig:
    init_p_diag: np.ndarray = field(default_factory=lambda: np.array([0.1**2, 0.1**2, (1 * np.pi/180)**2]))
    init_v_diag: np.ndarray = field(default_factory=lambda: np.array([0.1**2, 0.1**2, 0.05**2]))
    q_diag: np.ndarray = field(default_factory=lambda: np.array([0.01**2, 0.01**2, (0.5 * np.pi/180)**2]))
    r_meas_diag: np.ndarray = field(default_factory=lambda: np.array([0.15**2, (20 * np.pi/180)**2]))
    point_cov: np.ndarray = field(default_factory=lambda: np.array([0.4**2, 0.25**2, (25 * np.pi/180)**2]))  # [Longitudinal, Lateral, Heading]
    map_process_noise: float = 0.0
    augment_dist: float = 0.25
    augment_sigma: float = 0.12
    trigger_dist_edge: float = 0.02
    lane_width_limit: float = 0.5
    loop_closure_sigma: float = 0.05
    augment_sigma: float = 0.15
    post_closure_map_gain: float = 0.5

@dataclass(frozen=True)
class PurePursuitConfig:
    lookahead_idx: int = 15
    target_v: float = 0.3
    steering_gain: float = 12.0

    

@dataclass
class Observation:
    local_points: np.ndarray
    loop_closure_triggered: bool
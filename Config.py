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
    target_v: float = 0.3
    steering_gain: float = 12.0
    lookahead_idx: int = 10 # Need to wrap these using field to stop python being dumb and reusing pointer
    true_kine_noise: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.02]))
    odom_noise: np.ndarray = field(default_factory=lambda: np.array([0.003, 0.001]))
    vision_noise_std: float = 0.015
    vision_lookahead: List[int] = field(default_factory=lambda: list(range(-20, 21, 3)))

@dataclass(frozen=True)
class EKFConfig:
    init_p_diag: np.ndarray = field(default_factory=lambda: np.array([0.1**2, 0.1**2, 0.05**2]))
    q_diag: np.ndarray = field(default_factory=lambda: np.array([0.01**2, 0.01**2, 0.005**2]))
    r_meas_diag: np.ndarray = field(default_factory=lambda: np.array([0.2**2, 0.2**2]))
    point_cov: np.ndarray = field(default_factory=lambda: np.array([20.0**2, 0.7**2]))  # [Longitudinal, Lateral]
    map_process_noise: float = 0.0
    augment_dist: float = 0.28
    augment_sigma: float = 0.12
    trigger_dist_edge: float = 0.02
    lane_width_limit: float = 0.5

@dataclass
class Observation:
    local_points: np.ndarray
    loop_closure_triggered: bool
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, NamedTuple, Optional

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
    imu_update_rate: float = 250 # 250 hz (this sets the sim rate!!)
    odom_update_rate: float = 50 # hz
    map_update_rate: float = 10 # hz - needs to be at least 5x slower than odom!! plz or else ill cry :(
    sim_steps: int = 100000
    true_kine_noise: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.02]))
    odom_noise: np.ndarray = field(default_factory=lambda: np.array([0.05, (1 * np.pi/180)])) # Vx, Vy, W
    vision_noise_std: float = 0.015
    vision_lookahead: List[int] = field(default_factory=lambda: list(range(-40, 41)))
    imu_accel_noise: float = 0.05  # m/s^2
    imu_gyro_noise: float = 0.01   # rad/s

@dataclass(frozen=True)
class EKFConfig:
    # Covariences:
    init_p_diag: np.ndarray = field(default_factory=lambda: np.array([0.1**2, 0.1**2, (1 * np.pi/180)**2]))
    init_v_diag: np.ndarray = field(default_factory=lambda: np.array([0.1**2, 0.1**2, 0.05**2]))
    q_diag: np.ndarray = field(default_factory=lambda: np.array([
        0.1**2, 0.1**2, (0.5 * np.pi / 180)**2, # Pose process noise
        0.05**2, 0.05**2, 0.01**2 # IMU acceleration / velocity process noise
    ]))
    odom_noise: np.ndarray = field(default_factory=lambda: np.array([0.05**2, 0.05**2, (1 * np.pi/180)**2])) # Vx, Vy, w
    r_meas_diag: np.ndarray = field(default_factory=lambda: np.array([0.15**2, (20 * np.pi/180)**2])) # [Lat, theta] measurment of robot pose
    point_cov: np.ndarray = field(default_factory=lambda: np.array([0.4**2, 0.25**2, (25 * np.pi/180)**2]))  # [Longitudinal, Lateral, Heading] for map measurement

    # Mapping Params
    augment_dist: float = 0.25 # Spacing of new map points (m)
    augment_sigma: float = 0.12 # Covarience of new map point (m**2)
    trigger_dist_edge: float = 0.02 # Distance from edge new map point is added (m)
    lane_width_limit: float = 0.5 # Width of the track (m)
    loop_closure_sigma: float = 0.05 # Covarience after loop closure (m**2)
    map_init_sigma: float = 0.05 # Init covarience of map point (m**2)

    #Point Cloud Handling
    map_point_decimation_factor: int = 8 # sample every nth point recieved from camera. Adjust to taste
    frenet_sample_radius: float = 0.3 # radius (m) of camera point around robot used to estimate its line offset (beta) and heading error (theta) must keep smol 2 appox linear

    #Map Width Params
    width_meas_variance: float = 0.02**2 # m**2
    default_width: float = 0.5 # m
    width_var_init: float = 0.1 # m**2
    width_process_noise: float = 1e-4


@dataclass(frozen=True)
class PurePursuitConfig:
    lookahead_idx: int = 15
    target_v: float = 0.3
    steering_gain: float = 12.0

@dataclass
class IMUMeasurement:
    acc_x: float
    acc_y: float
    omega: float
    dt: float

@dataclass
class OdomMeasurement:
    vx_enc: float
    vy_enc: float
    w_enc: float

@dataclass(frozen=False)
class Observation:
    local_points: np.ndarray
    loop_closure_triggered: bool
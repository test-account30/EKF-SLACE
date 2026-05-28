import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from Sim import *
from Config import *
from Controller import PurePursuitController

"""
SLACE.py
Description:
    EKF SLACE (Simultaneous Localization And Curve Estimation)
    Multi-rate augmented state EKF fusing high-rate IMU, 3-DOF holonomic odometry, and polylines.
Author:
    Matthew Allen
"""

def compute_frenet_frame(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Computes line fit, normal, b_meas, and theta_meas from local points via SVD."""
    mean_pt = np.mean(pts, axis=0)
    _, _, Vh = np.linalg.svd(pts - mean_pt)
    t = Vh[0] if Vh[0, 0] >= 0 else -Vh[0]
    n = np.array([-t[1], t[0]])
    b = -np.dot(mean_pt, n)
    theta = -np.arctan2(t[1], t[0])
    return t, n, b, theta


import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from Sim import *
from Config import *
from Controller import PurePursuitController

def compute_frenet_frame(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Computes line fit, normal, b_meas, and theta_meas from local points via SVD."""
    mean_pt = np.mean(pts, axis=0)
    _, _, Vh = np.linalg.svd(pts - mean_pt)
    t = Vh[0] if Vh[0, 0] >= 0 else -Vh[0]
    n = np.array([-t[1], t[0]])
    b = -np.dot(mean_pt, n)
    theta = -np.arctan2(t[1], t[0])
    return t, n, b, theta


class EKFSLACE:
    """Core EKF SLACE (Simultaneous Localization And Curve Estimation)"""
    def __init__(self, config: EKFConfig, initial_pose: np.ndarray, initial_map_pts: np.ndarray):
        self.cfg = config
        
        # Build 6x6 continuous process noise matrix to match 6-DOF tracking model
        if len(config.q_diag) == 3:
            q_padded = list(config.q_diag) + [0.15**2, 0.15**2, 0.05**2]
            self.Q = np.diag(q_padded)
        else:
            self.Q = np.diag(config.q_diag)
            
        self.R = np.diag(config.r_meas_diag)
        self.loop_closed = False
        self.current_s = 0.0

        initial_map_flat = initial_map_pts.flatten()
        total_dim = 6 + len(initial_map_flat)  # 6-DOF Vehicle substate + N map elements
        
        # 1. State Allocation Layout
        self.X = np.zeros(total_dim)
        self.X[0:3] = initial_pose
        self.X[3:6] = np.array([0.0, 0.0, 0.0])  # Local body velocities: vx, vy, omega
        self.X[6:] = initial_map_flat

        # 2. Joint Covariance Allocation Layout
        P_init = np.diag(config.init_p_diag) if len(config.init_p_diag) == 3 else np.diag(config.init_p_diag[:3])
        V_init = np.diag([0.1**2, 0.1**2, 0.05**2])  # Tiny velocity variance baseline
        Sigma_M_init = np.eye(len(initial_map_flat)) * (0.05**2) 
        
        self.Sigma = np.zeros((total_dim, total_dim))
        self.Sigma[0:3, 0:3] = P_init
        self.Sigma[3:6, 3:6] = V_init
        self.Sigma[6:, 6:] = Sigma_M_init

        self.map_s = self._calc_arc_lengths(initial_map_pts)

    # State Accessor Properties
    @property
    def pose(self) -> np.ndarray:
        return self.X[0:3]

    @property
    def velocities(self) -> np.ndarray:
        return self.X[3:6]  # [vx, vy, omega] -> Read directly for low level holonomic controllers

    @property
    def M(self) -> np.ndarray:
        return self.X[6:]   

    @property
    def Sigma_M(self) -> np.ndarray:
        return self.Sigma[6:, 6:]

    def _sync_loop_closure_seam(self):
        """Enforces physical identity constraints on the wrap-around seam elements."""
        if not self.loop_closed:
            return
        closure_idx = len(self.M) // 2 - 1
        start_slot = 6
        end_slot = 6 + 2 * closure_idx
        
        # Force states to match flawlessly
        self.X[end_slot : end_slot + 2] = self.X[start_slot : start_slot + 2]
        # Force symmetric cross-covariance matrices to map cleanly
        self.Sigma[end_slot : end_slot + 2, :] = self.Sigma[start_slot : start_slot + 2, :]
        self.Sigma[:, end_slot : end_slot + 2] = self.Sigma[:, start_slot : start_slot + 2]

    def predict_via_imu(self, imu: IMUMeasurement):
        """Propagates 6-DOF kinematics using high-rate IMU strapdown calculations."""
        dt = imu.dt
        theta = self.X[2]
        vx, vy, omega = self.X[3], self.X[4], self.X[5]

        # 1. Kinematics State Integration 
        self.X[0] += (vx * np.cos(theta) - vy * np.sin(theta)) * dt
        self.X[1] += (vx * np.sin(theta) + vy * np.cos(theta)) * dt
        self.X[2] = normalize_angle(theta + omega * dt)
        self.X[3] += imu.acc_x * dt
        self.X[4] += imu.acc_y * dt
        # FIX: Let the EKF integrate omega smoothly instead of hard overwriting it,
        # or handle its derivative process noise properly.
        self.X[5] += 0.0  # Constant velocity model assumption between sensor updates

        # 2. Complete 6x6 Robot Analytical Sub-state Jacobian
        F_x = np.eye(6)
        F_x[0, 2] = (-vx * np.sin(theta) - vy * np.cos(theta)) * dt
        F_x[0, 3] = np.cos(theta) * dt
        F_x[0, 4] = -np.sin(theta) * dt
        F_x[1, 2] = (vx * np.cos(theta) - vy * np.sin(theta)) * dt
        F_x[1, 3] = np.sin(theta) * dt
        F_x[1, 4] = np.cos(theta) * dt
        F_x[2, 5] = dt
        F_x[5, 5] = 0.0

        # 3. Sparse O(N) Joint Matrix Multiplications
        self.Sigma[0:6, :] = F_x @ self.Sigma[0:6, :]
        self.Sigma[:, 0:6] = self.Sigma[:, 0:6] @ F_x.T
        self.Sigma[0:6, 0:6] += self.Q

    def update_odometry(self, odom: OdomMeasurement):
        """Correction tracking using 3-DOF Holonomic wheel encoder data (vx, vy, omega)."""
        z_meas = np.array([odom.vx_enc, odom.vy_enc, odom.w_enc])
        z_pred = np.array([self.X[3], self.X[4], self.X[2]])  # Pulls [vx, vy, omega] from state
        
        r = z_meas - z_pred
        r[2] = normalize_angle(r[2])
        
        # 3-DOF Measurement Mapping Matrix
        H = np.zeros((3, len(self.X)))
        H[0, 3] = 1.0  # vx map
        H[1, 4] = 1.0  # vy map
        H[2, 2] = 1.0  # omega map
        
        # Safe configuration unpacking for 3-axis odometry noise
        if len(self.cfg.odom_noise) == 2:
            odom_noise_padded = [self.cfg.odom_noise[0], self.cfg.odom_noise[0], self.cfg.odom_noise[1]]
        else:
            odom_noise_padded = self.cfg.odom_noise
            
        R_odom = np.diag([n**2 for n in odom_noise_padded])
        
        S = H @ self.Sigma @ H.T + R_odom
        K = self.Sigma @ H.T @ np.linalg.inv(S)
        
        self.X += K @ r
        self.X[2] = normalize_angle(self.X[2])
        
        I_KH = np.eye(len(self.X)) - K @ H
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + K @ R_odom @ K.T

    def _get_polyline_properties(self, s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float]:
        idx = np.clip(np.searchsorted(self.map_s, s) - 1, 0, len(self.map_s) - 2)
        denom = self.map_s[idx+1] - self.map_s[idx]
        u = (s - self.map_s[idx]) / denom if denom > 0 else 0.0

        p0, p1 = self.M[2*idx:2*idx+2], self.M[2*idx+2:2*idx+4]
        diff = p1 - p0
        dist = np.linalg.norm(diff)
        t = diff / dist if dist > 1e-6 else np.array([1.0, 0.0])
        n = np.array([-t[1], t[0]])
        
        return (1 - u) * p0 + u * p1, t, n, int(idx), int(idx + 1), u

    def update(self, obs: Observation):
        """Process local observations and update robot and map positions jointly."""
        if obs.loop_closure_triggered and not self.loop_closed:
            self._apply_loop_closure()
            self.loop_closed = True
            return

        if len(obs.local_points) == 0:
            return

        dists = np.linalg.norm(obs.local_points, axis=1)
        frenet_pts = obs.local_points[dists < 0.3]
        map_pts = obs.local_points[::8]

        if len(frenet_pts) < 2:
            return

        t_line, n_line, b_meas, theta_meas = compute_frenet_frame(frenet_pts)
        self._last_theta_meas = theta_meas  
        self._last_t_line = t_line
        z_meas = np.array([b_meas, theta_meas])

        self.current_s = self._project_onto_spline(self.pose[:2], self.current_s)
        C_star, t_m, n_m, idx0, idx1, u = self._get_polyline_properties(self.current_s)

        cos_e, sin_e = np.cos(self.pose[2]), np.sin(self.pose[2])
        R_T = np.array([[cos_e, sin_e], [-sin_e, cos_e]])

        C_local = R_T @ (C_star - self.pose[:2])
        t_m_local = R_T @ t_m
        n_m_local = R_T @ n_m

        z_pred = np.array([-np.dot(C_local, n_m_local), normalize_angle(-np.arctan2(t_m_local[1], t_m_local[0]))])
        r = z_meas - z_pred
        r[1] = normalize_angle(r[1])

        # Formulate full Joint Jacobian H = [H_robot, H_map] padded to match 6-DOF
        H_robot = np.array([
            [n_m[0], n_m[1], 0.0, 0.0, 0.0, 0.0], 
            [0.0,    0.0,    1.0, 0.0, 0.0, 0.0]
        ])
        
        H_map = np.zeros((2, len(self.M)))
        
        # Compute orientation sensitivities relative to map coordinates
        p0_geom = self.M[2*idx0 : 2*idx0 + 2]
        p1_geom = self.M[2*idx1 : 2*idx1 + 2]
        dx_g = p1_geom[0] - p0_geom[0]
        dy_g = p1_geom[1] - p0_geom[1]
        d2_g = dx_g**2 + dy_g**2 + 1e-9
        
        if self.loop_closed and idx1 == (len(self.M) // 2 - 1):
            # Cross-track rows
            H_map[0, 2*idx0 : 2*idx0 + 2] = -(1 - u) * n_m
            H_map[0, 0:2] += -u * n_m
            # Map Orientation rows (Row 1)
            H_map[1, 2*idx0 : 2*idx0 + 2] = np.array([-dy_g / d2_g, dx_g / d2_g])
            H_map[1, 0:2] += np.array([dy_g / d2_g, -dx_g / d2_g])
        else:
            # Cross-track rows
            H_map[0, 2*idx0 : 2*idx0 + 2] = -(1 - u) * n_m
            H_map[0, 2*idx1 : 2*idx1 + 2] = -u * n_m
            # Map Orientation rows (Row 1)
            H_map[1, 2*idx0 : 2*idx0 + 2] = np.array([-dy_g / d2_g, dx_g / d2_g])
            H_map[1, 2*idx1 : 2*idx1 + 2] = np.array([dy_g / d2_g, -dx_g / d2_g])

        H = np.zeros((2, len(self.X)))
        H[:, 0:6] = H_robot
        H[:, 6:] = H_map

        S = H @ self.Sigma @ H.T + self.R
        K = self.Sigma @ H.T @ np.linalg.inv(S)

        self.X += K @ r
        self.X[2] = normalize_angle(self.X[2])
        
        I_KH = np.eye(len(self.X)) - K @ H
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + K @ self.R @ K.T

        self._update_and_augment_map(map_pts)
        self._sync_loop_closure_seam()

    def _update_and_augment_map(self, local_obs_pts: np.ndarray):
        if len(local_obs_pts) == 0:
            return

        cos_e, sin_e = np.cos(self.pose[2]), np.sin(self.pose[2])
        R_pose = np.array([[cos_e, -sin_e], [sin_e, cos_e]])
        global_obs_pts = self.pose[:2] + local_obs_pts @ R_pose.T

        # --- 1. Isolated Heading update ---
        theta_track_global = normalize_angle(self.pose[2] + self._last_theta_meas)
        seg_idx = np.clip(np.searchsorted(self.map_s, self.current_s) - 1, 0, len(self.map_s) - 2)
        p0 = self.M[2*seg_idx:2*seg_idx+2]
        p1 = self.M[2*seg_idx+2:2*seg_idx+4]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        d2 = dx**2 + dy**2 + 1e-9
        r_theta = normalize_angle(theta_track_global - np.arctan2(dy, dx))
        
        J_theta = np.array([dy/d2, -dx/d2, -dy/d2, dx/d2])
        idx0_h, idx1_h = seg_idx, seg_idx + 1

        H_theta = np.zeros((1, len(self.X)))
        H_theta[0, 2] = -1.0  
        H_theta[0, 6 + 2*idx0_h : 6 + 2*idx0_h + 2] = J_theta[0:2]
        
        if self.loop_closed and idx1_h == (len(self.M) // 2 - 1):
            H_theta[0, 6:8] += J_theta[2:4]
        else:
            H_theta[0, 6 + 2*idx1_h : 6 + 2*idx1_h + 2] = J_theta[2:4]
        
        R_theta = self.cfg.point_cov[2]
        S_t = (H_theta @ self.Sigma @ H_theta.T)[0, 0] + R_theta
        K_t = (self.Sigma @ H_theta.T) / S_t
        
        # Apply change and IMMEDIATELY normalize angles safely
        self.X += K_t.flatten() * r_theta
        self.X[2] = normalize_angle(self.X[2]) 
        
        I_KH_t = np.eye(len(self.X)) - K_t @ H_theta
        self.Sigma = I_KH_t @ self.Sigma @ I_KH_t.T + R_theta * (K_t @ K_t.T)
        
        # Sync seam directly before proceeding to map expansion checks
        self._sync_loop_closure_seam()

        # --- 2. Check Augmentations FIRST ---
        for global_pt in global_obs_pts:
            self._check_augmentation(global_pt)

        # --- 3. Vectorized Batch Normal Update ---
        s_hint = self.current_s
        H_list = []
        r_list = []

        for local_pt, global_pt in zip(local_obs_pts, global_obs_pts):
            s_pt = self._project_onto_spline(global_pt, s_hint)
            s_hint = s_pt   
            
            C_s, t_m, n_m, idx0, idx1, u = self._get_polyline_properties(s_pt)

            xl, yl = local_pt
            G_r = np.array([
                [1.0, 0.0, -xl * sin_e - yl * cos_e],
                [0.0, 1.0,  xl * cos_e - yl * sin_e]
            ])

            r_n = float(n_m @ (global_pt - C_s))
            
            H_n_joint = np.zeros(len(self.X))
            H_n_joint[0:3] = -n_m @ G_r
            
            if self.loop_closed and idx1 == (len(self.M) // 2 - 1):
                H_n_joint[6 + 2*idx0 : 6 + 2*idx0 + 2] = n_m * (1 - u)
                H_n_joint[6 : 8] += n_m * u
            else:
                H_n_joint[6 + 2*idx0 : 6 + 2*idx0 + 2] = n_m * (1 - u)
                H_n_joint[6 + 2*idx1 : 6 + 2*idx1 + 2] = n_m * u
                
            H_list.append(H_n_joint)
            r_list.append(r_n)

        if len(H_list) > 0:
            H_batch = np.array(H_list)
            r_batch = np.array(r_list)
            
            K_pts = len(H_list)
            R_batch = np.eye(K_pts) * self.cfg.point_cov[1]
            
            Sigma_HT = self.Sigma @ H_batch.T
            S = H_batch @ Sigma_HT + R_batch
            
            K_gain = np.linalg.solve(S.T, Sigma_HT.T).T
            
            self.X += K_gain @ r_batch
            self.X[2] = normalize_angle(self.X[2])
            
            I_KH_loc = np.eye(len(self.X)) - K_gain @ H_batch
            self.Sigma = I_KH_loc @ self.Sigma @ I_KH_loc.T + K_gain @ R_batch @ K_gain.T
            
            self._sync_loop_closure_seam()
           
    def _check_augmentation(self, pt: np.ndarray):
        if self.cfg.trigger_dist_edge <= 0 or len(self.M) < 4 or self.loop_closed:
            return
        
        p_last, p_prev = self.M[-2:], self.M[-4:-2]
        t_last = p_last - p_prev
        t_last /= np.linalg.norm(t_last) + 1e-6

        v_obs = pt - p_last
        fwd_dist, lat_dist = np.dot(v_obs, t_last), np.dot(v_obs, np.array([-t_last[1], t_last[0]]))

        if fwd_dist > self.cfg.augment_dist and abs(lat_dist) < self.cfg.lane_width_limit:
            v_to_obs = pt - p_last
            dist_to_obs = np.linalg.norm(v_to_obs) + 1e-6
            p_new = p_last + self.cfg.augment_dist * (v_to_obs / dist_to_obs)

            alpha = self.cfg.augment_dist / dist_to_obs
            alpha = np.clip(alpha, 0.0, 1.0)

            old_dim = len(self.X)
            G = np.zeros((2, old_dim))

            cos_e, sin_e = np.cos(self.pose[2]), np.sin(self.pose[2])
            R_T = np.array([[cos_e, sin_e], [-sin_e, cos_e]])
            local_pt = R_T @ (pt - self.pose[:2])
            xl, yl = local_pt[0], local_pt[1]

            G_robot = np.array([
                [1.0, 0.0, -xl * sin_e - yl * cos_e],
                [0.0, 1.0,  xl * cos_e - yl * sin_e]
            ])
            G[:, 0:3] = alpha * G_robot
            G[:, old_dim - 2 : old_dim] = (1.0 - alpha) * np.eye(2)

            self.X = np.concatenate([self.X, p_new])
            self.map_s.append(self.map_s[-1] + np.linalg.norm(p_new - p_last))
            
            new_Sigma = np.zeros((old_dim + 2, old_dim + 2))
            new_Sigma[:old_dim, :old_dim] = self.Sigma
            
            Sigma_G_T = self.Sigma @ G.T
            new_Sigma[:old_dim, old_dim:] = Sigma_G_T
            new_Sigma[old_dim:, :old_dim] = Sigma_G_T.T
            
            R_noise = np.eye(2) * (self.cfg.augment_sigma**2)
            new_Sigma[old_dim:, old_dim:] = G @ Sigma_G_T + R_noise
            
            self.Sigma = new_Sigma

    def _apply_loop_closure(self):
        M_pts = self.M.reshape(-1, 2)
        start_anchor = M_pts[0]

        search_depth = min(50, len(M_pts) // 2)
        tail_idx = np.arange(len(M_pts) - search_depth, len(M_pts))

        dist_vals = np.linalg.norm(M_pts[tail_idx] - start_anchor, axis=1)
        closure_idx = int(tail_idx[np.argmin(dist_vals)])

        r = M_pts[0] - M_pts[closure_idx]

        H_lc = np.zeros((2, len(self.X)))
        H_lc[0, 6] = -1.0
        H_lc[1, 7] = -1.0
        H_lc[0, 6 + 2 * closure_idx] = 1.0
        H_lc[1, 7 + 2 * closure_idx] = 1.0

        R_lc = np.eye(2) * (1e-6 ** 2)
        S = H_lc @ self.Sigma @ H_lc.T + R_lc
        K = self.Sigma @ H_lc.T @ np.linalg.inv(S)

        self.X += K @ r
        self.X[2] = normalize_angle(self.X[2])
        
        I_KH = np.eye(len(self.X)) - K @ H_lc
        self.Sigma = I_KH @ self.Sigma @ I_KH.T

        trunc_nodes = closure_idx + 1
        new_dim = 6 + 2 * trunc_nodes
        
        self.X = self.X[:new_dim]
        self.Sigma = self.Sigma[:new_dim, :new_dim]
        
        # Enforce housekeeping immediately inside the frame before returning
        self._sync_loop_closure_seam()
        
        self.map_s = self._calc_arc_lengths(self.M.reshape(-1, 2))
        self.current_s = self._project_onto_spline(self.pose[:2], 0.0)

    @staticmethod
    def _calc_arc_lengths(P: np.ndarray) -> List[float]:
        s = [0.0]
        for i in range(1, len(P)):
            s.append(s[-1] + np.linalg.norm(P[i] - P[i-1]))
        return s

    def _project_onto_spline(self, point: np.ndarray, current_s: float, window=4) -> float:
        total_s = self.map_s[-1]
        num_segments = len(self.map_s) - 1

        if self.loop_closed and total_s > 0:
            current_s = current_s % total_s
            curr_idx = np.clip(np.searchsorted(self.map_s, current_s) - 1, 0, num_segments - 1)
            
            search_indices = []
            for offset in range(-window, window + 1):
                search_indices.append((curr_idx + offset) % num_segments)
                
            search_indices = list(dict.fromkeys(search_indices))
        else:
            curr_idx = np.clip(np.searchsorted(self.map_s, current_s) - 1, 0, num_segments - 1)
            st = max(0, curr_idx - window)
            end = min(num_segments, curr_idx + window + 1)
            search_indices = list(range(st, end))
        
        best_s, min_dist = current_s, float('inf')
        
        for i in search_indices:
            p0, p1 = self.M[2*i:2*i+2], self.M[2*i+2:2*i+4]
            v = p1 - p0
            v_len2 = np.dot(v, v)
            
            u = np.clip(0.0 if v_len2 < 1e-6 else np.dot(point - p0, v) / v_len2, 0.0, 1.0)
            dist = np.linalg.norm(point - (p0 + u * v))
            
            if dist < min_dist:
                min_dist = dist
                best_s = self.map_s[i] + u * (self.map_s[i+1] - self.map_s[i])
                
        return best_s


if __name__ == "__main__":
    np.random.seed(44)

    # Load reference track
    gt_track = np.load('custom_gt_track_reversed.npy')
    
    # Init configurations and simulation
    sim_cfg, ekf_cfg, ctrl_cfg = SimConfig(), EKFConfig(), PurePursuitConfig()
    sim = Sim(sim_cfg, gt_track)
    
    controller = PurePursuitController(ctrl_cfg) 
    obs_provider = SimCam(sim)
    
    init_pose = np.array([gt_track[0,0], gt_track[0,1], -np.pi/2])
    init_map = np.array([gt_track[0], gt_track[8], gt_track[16]])
    
    ekf = EKFSLACE(ekf_cfg, init_pose, init_map)
    
    # Init live visualizer engine
    visualizer = LiveVisualizer(gt_track, map_skip_val=50, record_video=False, video_path="slace.mp4")
    
    high_rate_steps = sim_cfg.sim_steps * sim.odom_stride  
    obs_packet = None

    # Initialize simulator and active tracking controller

    for step in range(high_rate_steps):
        # 1. Compute control actions independently of physics code
        cmd_vx, cmd_vy, cmd_w = controller.compute_commands(sim.true_pose, sim.gt_track, sim.closest_idx)
        # 2. Advance physics state by passing explicit velocity commands
        imu_packet, odom_packet = sim.step(cmd_vx, cmd_vy, cmd_w)
        # 3. Handle observer updates
        ekf.predict_via_imu(imu_packet)
        if odom_packet is not None:
            ekf.update_odometry(odom_packet)
        # 4. Handle asynchronous camera landmark observations

        if sim.is_camera_ready:
            obs_packet = obs_provider.get_observations()
            ekf.update(obs_packet)

            if obs_packet.loop_closure_triggered and not ekf.loop_closed:
                print(f"--- Step {step}: LOOP CLOSED SUCCESSFULLY ---")

        # 5. Refresh visualization display windows
        visualizer.update(step, sim, ekf, obs_packet)
        
    plt.ioff()
    plt.show()
    visualizer.close()
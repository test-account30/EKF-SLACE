import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from Sim import *
from Config import *

"""
SLACE.py
Description:
    EKF SLACE (Simultanious Localisation And Curve Estimation)
    Light(ish) weight curve-based SLAM using a deformable polyline representation
Author:
    Matthew Allen
Date Created:
    May 27, 2026
"""

class EKFSLACE:
    """Core EKF SLACE (Simultanious Localisation And Curve Estimation)"""
    def __init__(self, config: EKFConfig, initial_pose: np.ndarray, initial_map_pts: np.ndarray):
        self.cfg = config
        self.pose = initial_pose.copy()
        self.P = np.diag(config.init_p_diag)
        self.Q = np.diag(config.q_diag)
        self.R = np.diag(config.r_meas_diag)
        
        self.M = initial_map_pts.flatten()
        self.Sigma_M = np.diag([0.05**2, 0.05**2, 0.08**2, 0.08**2, 0.1**2, 0.1**2])
        self.map_s = self._calc_arc_lengths(initial_map_pts)
        self.current_s = 0.0
        self.loop_closed = False

    def predict(self, v: float, omega: float, dt: float):
        """Propagate state and covariance forward."""
        self.pose[0] += v * dt * np.cos(self.pose[2])
        self.pose[1] += v * dt * np.sin(self.pose[2])
        self.pose[2] = normalize_angle(self.pose[2] + omega * dt)

        F_x = np.array([
            [1.0, 0.0, -v * dt * np.sin(self.pose[2])],
            [0.0, 1.0,  v * dt * np.cos(self.pose[2])],
            [0.0, 0.0,  1.0]
        ])
        self.P = F_x @ self.P @ F_x.T + self.Q

    def update(self, obs: Observation):
        """Process local observations and optionally trigger loop closure."""
        if obs.loop_closure_triggered and not self.loop_closed:
            self._apply_loop_closure()
            self.loop_closed = True
            return

        if len(obs.local_points) == 0:
            return

        # 1. Update Robot Pose
        t_line, n_line, b_meas, theta_meas = compute_frenet_frame(obs.local_points)
        z_meas = np.array([b_meas, theta_meas])

        self.current_s = self._project_onto_spline(self.pose[:2], self.current_s)
        C_star, t_m, n_m, J_M_star = self._get_polyline_properties(self.current_s)

        cos_e, sin_e = np.cos(self.pose[2]), np.sin(self.pose[2])
        R_T = np.array([[cos_e, sin_e], [-sin_e, cos_e]])

        C_local = R_T @ (C_star - self.pose[:2])
        t_m_local = R_T @ t_m
        n_m_local = R_T @ n_m

        z_pred = np.array([-np.dot(C_local, n_m_local), normalize_angle(-np.arctan2(t_m_local[1], t_m_local[0]))])
        r = z_meas - z_pred
        r[1] = normalize_angle(r[1])

        H_robot = np.array([[n_m[0], n_m[1], 0.0], [0.0, 0.0, 1.0]])
        H_map = np.vstack((-n_m @ J_M_star, np.zeros(len(self.M))))

        S_map = H_map @ self.Sigma_M @ H_map.T
        S = H_robot @ self.P @ H_robot.T + self.R + S_map
        K = self.P @ H_robot.T @ np.linalg.inv(S)

        self.pose += K @ r
        self.pose[2] = normalize_angle(self.pose[2])
        I_KH = np.eye(3) - K @ H_robot
        self.P = I_KH @ self.P @ I_KH.T + K @ (self.R + S_map) @ K.T

        # 2. Update Map and Augment
        self._update_and_augment_map(obs.local_points)

    def _update_and_augment_map(self, local_obs_pts: np.ndarray):
        """Transform local points to global, check augmentation, update map nodes."""
        cos_e, sin_e = np.cos(self.pose[2]), np.sin(self.pose[2])
        R_pose = np.array([[cos_e, -sin_e], [sin_e, cos_e]])
        global_obs_pts = self.pose[:2] + local_obs_pts @ R_pose.T

        for local_pt, global_pt in zip(local_obs_pts, global_obs_pts):
            self._check_augmentation(global_pt)
            
            s_pt = self._project_onto_spline(global_pt, self.current_s)
            C_s, t_m, n_m, J_local, active_nodes = self._get_polyline_properties_local(s_pt)

            state_idx = np.ravel([[2*n, 2*n+1] for n in active_nodes])
            
            xl, yl = local_pt
            G_r = np.array([
                [1.0, 0.0, -xl * sin_e - yl * cos_e],
                [0.0, 1.0,  xl * cos_e - yl * sin_e]
            ])

            r = global_pt - C_s
            T = np.vstack([t_m, n_m])
            R_world = T.T @ np.diag(self.cfg.point_cov) @ T + (G_r @ self.P @ G_r.T)

            Sig_M_loc = self.Sigma_M[np.ix_(state_idx, state_idx)]
            S = J_local @ Sig_M_loc @ J_local.T + R_world
            K_loc = Sig_M_loc @ J_local.T @ np.linalg.inv(S)

            self.M[state_idx] += K_loc @ r
            
            # Isolated block update for numerical stability
            I_KH_loc = np.eye(len(state_idx)) - K_loc @ J_local
            self.Sigma_M[state_idx, :] = I_KH_loc @ self.Sigma_M[state_idx, :]
            self.Sigma_M[:, state_idx] = self.Sigma_M[:, state_idx] @ I_KH_loc.T
            self.Sigma_M[np.ix_(state_idx, state_idx)] = (
                I_KH_loc @ Sig_M_loc @ I_KH_loc.T + K_loc @ R_world @ K_loc.T
            )
            self.Sigma_M[state_idx, state_idx] += self.cfg.map_process_noise

    def _check_augmentation(self, pt: np.ndarray):
        if self.cfg.trigger_dist_edge <= 0 or len(self.M) < 4 or self.loop_closed:
            return
        
        p_last, p_prev = self.M[-2:], self.M[-4:-2]
        t_last = p_last - p_prev
        t_last /= np.linalg.norm(t_last) + 1e-6

        v_obs = pt - p_last
        fwd_dist, lat_dist = np.dot(v_obs, t_last), np.dot(v_obs, np.array([-t_last[1], t_last[0]]))

        if fwd_dist > self.cfg.augment_dist and abs(lat_dist) < self.cfg.lane_width_limit:
            p_new = p_last + self.cfg.augment_dist * t_last
            self.M = np.concatenate([self.M, p_new])
            self.map_s.append(self.map_s[-1] + np.linalg.norm(p_new - p_last))
            
            n = len(self.map_s) - 1
            new_Sigma = np.zeros((2*n + 2, 2*n + 2))
            new_Sigma[:2*n, :2*n] = self.Sigma_M
            
            idx = 2*n - 2
            new_Sigma[2*n:, :2*n] = self.Sigma_M[idx:, :]
            new_Sigma[:2*n, 2*n:] = self.Sigma_M[:, idx:]
            new_Sigma[2*n:, 2*n:] = self.Sigma_M[idx:, idx:] + np.eye(2) * (self.cfg.augment_sigma**2)
            self.Sigma_M = new_Sigma

    def _apply_loop_closure(self):
        """Applies loop closure correction internally."""
        M_pts = self.M.reshape(-1, 2)
        start_anchor = M_pts[0].copy()
        
        # Simple heuristic to find closure point from tail nodes
        tail_idx = np.arange(len(M_pts) - 4, len(M_pts))
        dists = np.linalg.norm(M_pts[tail_idx] - start_anchor, axis=1)
        closure_idx = tail_idx[np.argmin(dists)]
        end_frontier = M_pts[closure_idx].copy()
        
        gap_error = start_anchor - end_frontier
        t_start = M_pts[1] - M_pts[0]
        t_end = M_pts[closure_idx] - M_pts[closure_idx - 1]
        
        dtheta = normalize_angle(np.arctan2(t_start[1], t_start[0]) - np.arctan2(t_end[1], t_end[0]))
        R_full = np.array([[np.cos(dtheta), -np.sin(dtheta)], [np.sin(dtheta), np.cos(dtheta)]])

        for i in range(1, closure_idx + 1):
            w = (i / closure_idx) ** 2
            theta_i = w * dtheta
            t_i = w * gap_error
            c, s = np.cos(theta_i), np.sin(theta_i)
            R_i = np.array([[c, -s], [s, c]])
            
            M_pts[i] = R_i @ (M_pts[i] - end_frontier) + end_frontier + t_i
            idx = 2 * i
            self.Sigma_M[idx:idx+2, :] = R_i @ self.Sigma_M[idx:idx+2, :]
            self.Sigma_M[:, idx:idx+2] = self.Sigma_M[:, idx:idx+2] @ R_i.T

        M_pts[closure_idx] = start_anchor
        M_pts = M_pts[:closure_idx + 1]
        
        self.pose[:2] = R_full @ (self.pose[:2] - end_frontier) + end_frontier + gap_error
        self.pose[2] = normalize_angle(self.pose[2] + dtheta)
        
        N_new = len(M_pts)
        self.Sigma_M = self.Sigma_M[:2*N_new, :2*N_new] + np.eye(2*N_new) * 1e-3
        self.Sigma_M *= 0.2
        self.P[:2, :2] = R_full @ self.P[:2, :2] @ R_full.T
        self.P += np.diag([1e-3, 1e-3, 1e-3])
        self.P *= 3
        self.M = M_pts.flatten()
        self.map_s = self._calc_arc_lengths(M_pts)
        self.current_s = 0.0

    # Spline helpers
    @staticmethod
    def _calc_arc_lengths(P: np.ndarray) -> List[float]:
        s = [0.0]
        for i in range(1, len(P)):
            s.append(s[-1] + np.linalg.norm(P[i] - P[i-1]))
        return s

    def _project_onto_spline(self, point: np.ndarray, current_s: float, window=4) -> float:
        curr_idx = np.clip(np.searchsorted(self.map_s, current_s) - 1, 0, len(self.map_s) - 2)
        st, end = max(0, curr_idx - window), min(len(self.map_s) - 1, curr_idx + window + 1)
        
        best_s, min_dist = current_s, float('inf')
        for i in range(st, end):
            p0, p1 = self.M[2*i:2*i+2], self.M[2*i+2:2*i+4]
            v = p1 - p0
            v_len2 = np.dot(v, v)
            u = np.clip(0.0 if v_len2 < 1e-6 else np.dot(point - p0, v) / v_len2, 0.0, 1.0)
            dist = np.linalg.norm(point - (p0 + u * v))
            if dist < min_dist:
                min_dist, best_s = dist, self.map_s[i] + u * (self.map_s[i+1] - self.map_s[i])
        return best_s

    def _get_polyline_properties(self, s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = np.clip(np.searchsorted(self.map_s, s) - 1, 0, len(self.map_s) - 2)
        denom = self.map_s[idx+1] - self.map_s[idx]
        u = (s - self.map_s[idx]) / denom if denom > 0 else 0.0

        p0, p1 = self.M[2*idx:2*idx+2], self.M[2*idx+2:2*idx+4]
        diff = p1 - p0
        dist = np.linalg.norm(diff)
        t = diff / dist if dist > 1e-6 else np.array([1.0, 0.0])
        
        J = np.zeros((2, len(self.M)))
        J[:, 2*idx:2*idx+2] = (1 - u) * np.eye(2)
        J[:, 2*idx+2:2*idx+4] = u * np.eye(2)
        
        return (1 - u) * p0 + u * p1, t, np.array([-t[1], t[0]]), J

    def _get_polyline_properties_local(self, s: float, window=10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
        idx = np.clip(np.searchsorted(self.map_s, s) - 1, 0, len(self.map_s) - 2)
        denom = self.map_s[idx+1] - self.map_s[idx]
        u = (s - self.map_s[idx]) / denom if denom > 0 else 0.0

        p0, p1 = self.M[2*idx:2*idx+2], self.M[2*idx+2:2*idx+4]
        diff = p1 - p0
        dist = np.linalg.norm(diff)
        t = diff / dist if dist > 1e-6 else np.array([1.0, 0.0])

        active_nodes = list(range(max(0, idx - window), min(len(self.map_s), idx + window + 2)))
        J_local = np.zeros((2, 2 * len(active_nodes)))
        
        loc_0, loc_1 = active_nodes.index(idx), active_nodes.index(idx + 1)
        J_local[:, 2*loc_0:2*loc_0+2] = (1 - u) * np.eye(2)
        J_local[:, 2*loc_1:2*loc_1+2] = u * np.eye(2)

        return (1 - u) * p0 + u * p1, t, np.array([-t[1], t[0]]), J_local, active_nodes


if __name__ == "__main__":
    np.random.seed(42)
    
    # load reference track
    gt_track = np.load('track.npy')
    
    # init sim
    sim_cfg, ekf_cfg = SimConfig(), EKFConfig()
    sim = Sim(sim_cfg, gt_track)
    obs_provider = SimCam(sim)
    
    init_pose = np.array([gt_track[0,0] + 0.05, gt_track[0,1] - 0.05, -np.pi/2 + 0.02])
    init_map = np.array([gt_track[0], gt_track[2], gt_track[5]])
    
    ekf = EKFSLACE(ekf_cfg, init_pose, init_map)
    
    # init vis
    visualizer = LiveVisualizer(gt_track, map_skip_val=10, record_video=False, video_path="slace.mp4")
    
    # loop
    for step in range(sim_cfg.sim_steps):
        # simulate hardware, returns odometry
        v_enc, w_enc = sim.step() 
        
        # simulate CV data returns
        obs_packet = obs_provider.get_observations()
        
        # EKF core
        ekf.predict(v_enc, w_enc, sim_cfg.dt)
        ekf.update(obs_packet)

        if obs_packet.loop_closure_triggered and not ekf.loop_closed:
            print(f"--- Step {step}: LOOP CLOSED SUCCESSFULLY ---")

        # update plot
        visualizer.update(step, sim, ekf, obs_packet)
        
    plt.ioff()
    plt.show() # Keep the plot open at the end
    visualizer.close() # Stop recording
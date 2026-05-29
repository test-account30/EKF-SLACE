import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from Sim import *
from Config import *
from Controller import *
from utils import normalize_angle

"""
SLACE.py
Description:
    EKF SLACE (Simultaneous Localization And Curve Estimation)
    Light(ish) weight curve-based SLAM using a deformable polyline representation
Author:
    Matthew Allen
"""

class EKFSLACE:
    """Core EKF SLACE (Simultaneous Localization And Curve Estimation)"""
    # NOTE: This class will be heavily over-commented so I don't forget how the maths works
    def __init__(self, config: EKFConfig, initial_pose: np.ndarray, initial_map_pts: np.ndarray):
        self.cfg = config
        self.Q = np.diag(config.q_diag)
        self.R = np.diag(config.r_meas_diag)
        self.loop_closed = False
        self.current_s = 0
        self.seg_idx = 0
        # init the state vector of the EKF
        initial_map_flat = initial_map_pts.flatten()
        total_dim = 6 + len(initial_map_flat)
        self.X = np.zeros(total_dim)
        self.X[0:3] = initial_pose # x, y, theta
        self.X[3:6] = np.array([0.0, 0.0, 0.0])  # vx, vy, omega
        self.X[6:] = initial_map_flat # map points (x_1, y_1, x_2, y_2, ... ,x_n, y_n)

        # Init the covarience matrix
        P_init = np.diag(config.init_p_diag) # position covarience
        V_init = np.diag(config.init_v_diag) # velocity covarience
        Sigma_M_init = np.eye(len(initial_map_flat)) * (config.map_init_sigma**2) # init point covarience
        
        self.Sigma = np.zeros((total_dim, total_dim)) # Write the covariences of each component to the init covarience matrix
        self.Sigma[0:3, 0:3] = P_init
        self.Sigma[3:6, 3:6] = V_init
        self.Sigma[6:, 6:] = Sigma_M_init

        self.map_s = self._calc_arc_lengths(initial_map_pts) # Calculate the current length of the map 

        num_nodes = len(initial_map_flat) // 2 # Initalise the track width estimator
        self.W_est = np.ones(num_nodes) * config.default_width
        self.W_var = np.ones(num_nodes) * config.width_var_init

    # Getters
    @property
    def pose(self) -> np.ndarray:
        return self.X[0:3]

    @property
    def velocities(self) -> np.ndarray:
        return self.X[3:6]

    @property
    def M(self) -> np.ndarray:
        return self.X[6:]   

    @property
    def Sigma_M(self) -> np.ndarray:
        return self.Sigma[6:, 6:]

    def predict_imu(self, imu:IMUMeasurement):
        """Update the robot state w/ IMU measurements"""
        theta = self.X[2]
        vx, vy = self.X[3], self.X[4]
        ax, ay, dt = imu.acc_x, imu.acc_y, imu.dt # Read in the imu states

        self.X[0] += (self.X[3] * np.cos(theta) - self.X[4] * np.sin(theta)) * dt # Basic euler integration (NOTE: If this starts playing up on the actual robot, 445 dude said runge kutta is better)
        self.X[1] += (self.X[3] * np.sin(theta) + self.X[4] * np.cos(theta)) * dt
        self.X[2] = normalize_angle(theta + self.X[5] * dt)
        self.X[3] += ax * dt
        self.X[4] += ay * dt

        self.X[5] = imu.omega

        # State:
        # x = [x, y, theta, vx, vy, omega]

        # Motion model:
        # x' = x + (vx*cos(theta) - vy*sin(theta)) * dt
        # y' = y + (vx*sin(theta) + vy*cos(theta)) * dt
        # theta' = theta + omega * dt
        # vx' = vx + ...
        # vy' = vy + ...
        # omega' = omega
        # F_x = d x_{k+1} / d x_k
        F_x = np.eye(6)
        F_x[0, 2] = (-vx * np.sin(theta) - vy * np.cos(theta)) * dt # dx' / dtheta
        F_x[0, 3] = np.cos(theta) * dt # dx' / dvx
        F_x[0, 4] = -np.sin(theta) * dt # dx' / dvy
        F_x[1, 2] = (vx * np.cos(theta) - vy * np.sin(theta)) * dt # dy' / dtheta
        F_x[1, 3] = np.sin(theta) * dt # dy' / dvx
        F_x[1, 4] = np.cos(theta) * dt # dy' / dvy
        F_x[2, 5] = dt  # dtheta' / domega
        F_x[5, 5] = 0.0 # domega' / domega

        # Write to the covarience matrix
        self.Sigma[0:6, :] = F_x @ self.Sigma[0:6, :]
        self.Sigma[:, 0:6] = self.Sigma[:, 0:6] @ F_x.T
        self.Sigma[0:6, 0:6] += self.Q * dt

    def _update_width(self, idx: int, measured_w: float, R_meas: float, Q_process: float = 1e-4):
        """Update the map width"""

        P_pred = self.W_var[idx] + Q_process # predict step (assume constant width)
        
        y = measured_w - self.W_est[idx]  # update step
        S = P_pred + R_meas
        K = P_pred / S
        
        # Apply
        self.W_est[idx] = self.W_est[idx] + K * y
        self.W_var[idx] = (1.0 - K) * P_pred

    def update_odom(self, odom:OdomMeasurement):
        """Update the robot state with odometry measurements"""
        vx, vy, omega = odom.vx_enc, odom.vy_enc, odom.w_enc # pull out encoder velocities
        z_meas = np.array([vx, vy, omega])
        z_pred = np.array([self.X[3], self.X[4], self.X[5]])
        
        r = z_meas - z_pred # Define the innovation between measurement and predicted state
        r[2] = normalize_angle(r[2])
        
        # We are measuring velocity states directly so measurement matrix is just 1 at correspondiong elements
        H = np.zeros((3, len(self.X)))
        H[0, 3] = 1.0  # vx map
        H[1, 4] = 1.0  # vy map
        H[2, 5] = 1.0  # omega map
                   
        R_odom = np.diag(self.cfg.odom_noise) # assemble the odometry measurement noise covarience
        
        # Run a kalman update step on just the robot states of the state vector (first 6 states)
        S = H @ self.Sigma @ H.T + R_odom
        K = self.Sigma @ H.T @ np.linalg.inv(S)
        
        self.X += K @ r
        self.X[2] = normalize_angle(self.X[2]) # making sure to not blow up the angle
        
        I_KH = np.eye(len(self.X)) - K @ H
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + K @ R_odom @ K.T # Update the respective covarience matrix

    def update_SLACE(self, obs: Observation):
        """Update the SLACE curve and localise the robot on that curve"""

        if obs.loop_closure_triggered and not self.loop_closed: # Check if the vision system has fired off a loop close trigger
            self._apply_loop_closure()
            self.loop_closed = True
            return
        
        if len(obs.local_points) == 0: # If there aren't any points being picked up by vision, skip the map update
            return
        
        pts_xy = obs.local_points[:, :2]   # (x, y)
        dists = np.linalg.norm(pts_xy, axis=1)
        # SVD used to estimate robot offset and heading to line, pick a cluster of points near the robot to get approx. linear line segment
        frenet_pts = pts_xy[dists < self.cfg.frenet_sample_radius]

        # Down sample map update by decimating the full vision range
        map_pts = obs.local_points[::self.cfg.map_point_decimation_factor]



        if len(frenet_pts) < 2: # Need at least 2 points to run SVD 
            return
        
        # figure out the tangent and normal vectors and measure beta and theta values of robot to line
        t_line, n_line, b_meas, theta_meas = compute_frenet_frame(frenet_pts)
        self._last_theta_meas = theta_meas  
        self._last_t_line = t_line
        z_meas = np.array([b_meas, theta_meas]) # Stick those in measurement vector

        self.current_s = self._project_onto_spline(self.pose[:2], self.current_s) # Get current robot lateral distance along map curve
        C_star, t_m, n_m, idx0, idx1, u = self._get_polyline_properties(self.current_s) # Get the properties of that map line segment

        cos_e, sin_e = np.cos(self.pose[2]), np.sin(self.pose[2])
        R_T = np.array([[cos_e, sin_e], 
                        [-sin_e, cos_e]])

        C_local = R_T @ (C_star - self.pose[:2]) # Grab the distance error from robot -> line
        t_m_local = R_T @ t_m # Compute the tangent and normal components vector components
        n_m_local = R_T @ n_m

        z_pred = np.array([-np.dot(C_local, n_m_local), normalize_angle(-np.arctan2(t_m_local[1], t_m_local[0]))]) # Compute the estimated lateral and theta error of robot on line
        r = z_meas - z_pred # Compute innovation between measurement and estimate
        r[1] = normalize_angle(r[1])

        # We are measuring robot in lateral offset and theta, state is in x and y, so Jacobian turns out to be just the normal vector component in x and y.
        H_robot = np.array([
            # dr_cross / d[x, y, theta, vx, vy, omega]
            [n_m[0], n_m[1], 0.0, 0.0, 0.0, 0.0], 
            # dr_theta / d[x, y, theta, vx, vy, omega]
            [0.0,    0.0,    1.0, 0.0, 0.0, 0.0]
        ])
        
        # r = [r_cross, r_theta]
        # H_map = dr / d[map_nodes]
        #
        # cross track innov:
        # r_cross = n_m^T (p - C(s))
        # where C(s) = (1 - u)p0 + u p1
        #
        # Orientation innv:
        # r_theta = theta_obs - atan2(dy, dx)
        # where dx = x1 - x0, dy = y1 - y0
        H_map = np.zeros((2, len(self.M))) # Not measuring the map at all
        
        # Compute orientation sensitivities relative to map coordinates
        p0_geom = self.M[2*idx0 : 2*idx0 + 2]
        p1_geom = self.M[2*idx1 : 2*idx1 + 2]
        dx_g = p1_geom[0] - p0_geom[0]
        dy_g = p1_geom[1] - p0_geom[1]
        d2_g = dx_g**2 + dy_g**2 + 1e-9
        
        # Cross track:
        # dr_cross / dp0 = -(1 - u) * n_m
        # dr_cross / dp1 = -u * n_m
        #
        # If loop closed, p1 wraps to first node
        if self.loop_closed and idx1 == (len(self.M) // 2 - 1):
            # Cross-track rows
            H_map[0, 2*idx0 : 2*idx0 + 2] = -(1 - u) * n_m
            H_map[0, 0:2] += -u * n_m
            # Orientation Jacobian:
            # J_theta = d(atan2(dy, dx)) / d[p0, p1]
            H_map[1, 2*idx0 : 2*idx0 + 2] = np.array([-dy_g / d2_g, dx_g / d2_g])
            H_map[1, 0:2] += np.array([dy_g / d2_g, -dx_g / d2_g])
        else:
            # Cross-track rows
            H_map[0, 2*idx0 : 2*idx0 + 2] = -(1 - u) * n_m
            H_map[0, 2*idx1 : 2*idx1 + 2] = -u * n_m
            # Orientation Jacobian:
            # J_theta = d(atan2(dy, dx)) / d[p0, p1]
            H_map[1, 2*idx0 : 2*idx0 + 2] = np.array([-dy_g / d2_g, dx_g / d2_g])
            H_map[1, 2*idx1 : 2*idx1 + 2] = np.array([dy_g / d2_g, -dx_g / d2_g])

        # Full measurement Jacobian
        # H = dr / d[x, y, theta, map_nodes]
        H = np.zeros((2, len(self.X)))
        H[:, 0:6] = H_robot
        H[:, 6:] = H_map

        # Run the EKF update step on robot pose
        S = H @ self.Sigma @ H.T + self.R
        K = self.Sigma @ H.T @ np.linalg.inv(S)

        self.X += K @ r
        self.X[2] = normalize_angle(self.X[2])
        
        I_KH = np.eye(len(self.X)) - K @ H
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + K @ self.R @ K.T

        # Now its time to update the map!
        self._update_and_augment_map(map_pts)

    def _update_and_augment_map(self, local_obs_pts: np.ndarray):
        """Update SLACE Map curve from measurements"""
        if len(local_obs_pts) == 0:
            return
        
        pts_xy = local_obs_pts[:, :2]   # (x, y)
        pts_w  = local_obs_pts[:, 2]    # width

        # The basic layout for this section, is first map segment near robot is updated using robot rotation measurement,
        # Then in another update step the maps lateral error using the large map span measurements is updated.

        # Convert the local point observations from body frame to world frame
        cos_e, sin_e = np.cos(self.pose[2]), np.sin(self.pose[2])
        R_pose = np.array([[cos_e, -sin_e], [sin_e, cos_e]])
        global_obs_pts = self.pose[:2] + pts_xy @ R_pose.T 


        theta_track_global = normalize_angle(self.pose[2] + self._last_theta_meas) # Estimate global line orientation

        self.seg_idx = np.clip(np.searchsorted(self.map_s, self.current_s) - 1, 0, len(self.map_s) - 2) # Grab the map point cloest to robot
        p0 = self.M[2*self.seg_idx:2*self.seg_idx+2] # Grab the 2 points making up that segment
        p1 = self.M[2*self.seg_idx+2:2*self.seg_idx+4]

        dx, dy = p1[0] - p0[0], p1[1] - p0[1] # Compute the segment vector
        d2 = dx**2 + dy**2 + 1e-9 # Grab its length
        r_theta = normalize_angle(theta_track_global - np.arctan2(dy, dx)) # Compute the error between the current orientation and estimated orientation
        
        # theta_map = atan2(dy, dx)
        # dx = x1 - x0
        # dy = y1 - y0
        #
        # J_theta = d(theta_map) / d[p0_x, p0_y, p1_x, p1_y]
        J_theta = np.array([dy/d2, -dx/d2, -dy/d2, dx/d2]) # Jacobian for segment theta update
        idx0_h, idx1_h = self.seg_idx, self.seg_idx + 1

        # r_theta = theta_obs - theta_map
        # H_theta = dr_theta / d[x, y, theta, map_nodes]
        H_theta = np.zeros((1, len(self.X)))

        # robot heading contribution:
        # theta_map depends on theta, so dr/dtheta = -1
        H_theta[0, 2] = -1.0  

        # map segment orientation:
        # theta_map = atan2(dy, dx)
        # J_theta = d(theta_map) / d[p0, p1]
        H_theta[0, 6 + 2*idx0_h : 6 + 2*idx0_h + 2] = J_theta[0:2] # Assemble measurement jacobian H
        
        if self.loop_closed and idx1_h == (len(self.M) // 2 - 1): # Being careful about indexing when the path becomes a loop
            H_theta[0, 6:8] += J_theta[2:4]
        else:
            H_theta[0, 6 + 2*idx1_h : 6 + 2*idx1_h + 2] = J_theta[2:4]
        
        # Perform EKF Update on map
        R_theta = self.cfg.point_cov[2]
        S_t = (H_theta @ self.Sigma @ H_theta.T)[0, 0] + R_theta
        K_t = (self.Sigma @ H_theta.T) / S_t
        
        self.X += K_t.flatten() * r_theta
        self.X[2] = normalize_angle(self.X[2]) # Normalise angle
        
        I_KH_t = np.eye(len(self.X)) - K_t @ H_theta
        self.Sigma = I_KH_t @ self.Sigma @ I_KH_t.T + R_theta * (K_t @ K_t.T) # Update map covarience

        
        for global_pt in global_obs_pts:
            self._check_augmentation(global_pt) # Check if a new map point needs to be added 

        # OK! Now its time to update the maps lateral error off the path using the spanning measurements
        s_hint = self.current_s
        H_list = []
        r_list = []

        for local_pt, global_pt, w_meas in zip(pts_xy, global_obs_pts, pts_w):
            s_pt = self._project_onto_spline(global_pt, s_hint) # Get the corresponding path index of the current measurement point
            s_hint = s_pt   
            
            C_s, t_m, n_m, idx0, idx1, u = self._get_polyline_properties(s_pt) # Get the properties of that point

            base_R = self.cfg.width_meas_variance
            base_Q = self.cfg.width_process_noise
            self._update_width(idx0, w_meas, base_R / (1.0 - u + 1e-3), base_Q)
            self._update_width(idx1, w_meas, base_R / (u + 1e-3), base_Q)

            xl, yl = local_pt
            # p_world = [x, y] + R(theta) * p_local
            # dp_world / d[x, y, theta]
            G_r = np.array([ # Jacobian describing process covarience of projecting points from local to world space using pose estimate
                [1.0, 0.0, -xl * sin_e - yl * cos_e],
                [0.0, 1.0,  xl * cos_e - yl * sin_e]
            ])

            r_n = float(n_m @ (global_pt - C_s)) # Compute the normal vector component of the path error at that point. 
            #We DON'T use the longitudional path error because the camera can't directly measure it, it causes the map to stretch and warp weirdly.
            
            #Jacobian time!
            # r_n = n_m^T (p_world - C(s))
            # H = dr_n / d[x, y, theta, map_nodes]

            # robot part:
            # p_world = [x, y] + R(theta) * p_local
            # dr_n / d[x, y, theta] = -n_m^T * dp_world / d[x, y, theta]
            H_n_joint = np.zeros(len(self.X)) # Make a blank jacobian vector
            H_n_joint[0:3] = -n_m @ G_r # Given we are only updating in the normal vector direction, the robot measurement covarience should only apply in the normal direction too
            
            # map part:
            # C(s) = (1 - u) * p0 + u * p1
            # dr_n / dp0 = (1 - u) * n_m
            # dr_n / dp1 = u * n_m
            if self.loop_closed and idx1 == (len(self.M) // 2 - 1): # Again, being super careful about indexing after loop closure (I have trauma from trying to figure this out)
                H_n_joint[6 + 2*idx0 : 6 + 2*idx0 + 2] = n_m * (1 - u)
                H_n_joint[6 : 8] += n_m * u
            else:
                H_n_joint[6 + 2*idx0 : 6 + 2*idx0 + 2] = n_m * (1 - u)
                H_n_joint[6 + 2*idx1 : 6 + 2*idx1 + 2] = n_m * u
                
            H_list.append(H_n_joint)
            r_list.append(r_n)

        if len(H_list) > 0: # Now we have assembled a full list of measurement jacobians for each measurement point, batch apply the update
            # This is called a gauss-newton iteration btw
            H_batch = np.array(H_list)
            r_batch = np.array(r_list)
            # batch all measurements and innovations into a big block matrix, run the EKF update in one big batch
            K_pts = len(H_list)
            R_batch = np.eye(K_pts) * self.cfg.point_cov[1]
            
            Sigma_HT = self.Sigma @ H_batch.T
            S = H_batch @ Sigma_HT + R_batch
            
            # Given this is a BIG matrix, solve is faster apparently according to google. No idea what the equivalent is in Go so good luck Dan :)
            K_gain = np.linalg.solve(S.T, Sigma_HT.T).T
            
            self.X += K_gain @ r_batch
            self.X[2] = normalize_angle(self.X[2])
            
            I_KH_loc = np.eye(len(self.X)) - K_gain @ H_batch
            self.Sigma = I_KH_loc @ self.Sigma @ I_KH_loc.T + K_gain @ R_batch @ K_gain.T
           
    def _check_augmentation(self, pt: np.ndarray):
        """Add a new node to the map if measurements exceed frontier"""
        if self.cfg.trigger_dist_edge <= 0 or len(self.M) < 4 or self.loop_closed:
            return
        
        p_last, p_prev = self.M[-2:], self.M[-4:-2] # Grab the latest point in the map
        t_last = p_last - p_prev
        t_last /= np.linalg.norm(t_last) + 1e-6

        v_obs = pt - p_last
        fwd_dist, lat_dist = np.dot(v_obs, t_last), np.dot(v_obs, np.array([-t_last[1], t_last[0]])) # Compute 2 metrics, forward and lateral distance

        if fwd_dist > self.cfg.augment_dist and abs(lat_dist) < self.cfg.lane_width_limit: # If the map point is within a certain distance past the map fronter, add a new point
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
            # p_world = [x, y] + R(theta) * p_local
            # dp_world / d[x, y, theta]
            G_robot = np.array([
                [1.0, 0.0, -xl * sin_e - yl * cos_e],
                [0.0, 1.0,  xl * cos_e - yl * sin_e]
            ])
            G[:, 0:3] = alpha * G_robot
            G[:, old_dim - 2 : old_dim] = (1.0 - alpha) * np.eye(2)

            # update map vector
            self.X = np.concatenate([self.X, p_new])
            self.map_s.append(self.map_s[-1] + np.linalg.norm(p_new - p_last))
            
            # update map width vector
            self.W_est = np.append(self.W_est, self.cfg.default_width)
            self.W_var = np.append(self.W_var, self.cfg.width_var_init)
            
            new_Sigma = np.zeros((old_dim + 2, old_dim + 2))
            new_Sigma[:old_dim, :old_dim] = self.Sigma
            
            Sigma_G_T = self.Sigma @ G.T
            new_Sigma[:old_dim, old_dim:] = Sigma_G_T
            new_Sigma[old_dim:, :old_dim] = Sigma_G_T.T
            
            R_noise = np.eye(2) * (self.cfg.augment_sigma**2)
            new_Sigma[old_dim:, old_dim:] = G @ Sigma_G_T + R_noise
            
            self.Sigma = new_Sigma

    def _apply_loop_closure(self):
        """Snap the map into a closed curve"""
        M_pts = self.M.reshape(-1, 2)
        start_anchor = M_pts[0]

        search_depth = min(50, len(M_pts) // 2)
        tail_idx = np.arange(len(M_pts) - search_depth, len(M_pts))

        dist_vals = np.linalg.norm(M_pts[tail_idx] - start_anchor, axis=1)
        closure_idx = int(tail_idx[np.argmin(dist_vals)])
        
        # r_lc = p_start - p_closure
        r = M_pts[0] - M_pts[closure_idx] # grab the error between the first point and last point in map 
        # We trim off the last few points ahead of the robot, because those are duplicates of the first points in the map when loop closure is triggered.

        # H_lc = d(r_lc) / d[x, y, theta, vx, vy, omega, map_nodes]
        H_lc = np.zeros((2, len(self.X)))
        H_lc[0, 6] = -1.0
        H_lc[1, 7] = -1.0
        H_lc[0, 6 + 2 * closure_idx] = 1.0
        H_lc[1, 7 + 2 * closure_idx] = 1.0

        R_lc = np.eye(2) * (1e-6 ** 2) # Super high confidence which means points snap shut quick
        S = H_lc @ self.Sigma @ H_lc.T + R_lc
        K = self.Sigma @ H_lc.T @ np.linalg.inv(S)

        self.X += K @ r # Update the map
        self.X[2] = normalize_angle(self.X[2])
        
        I_KH = np.eye(len(self.X)) - K @ H_lc
        self.Sigma = I_KH @ self.Sigma @ I_KH.T

        trunc_nodes = closure_idx + 1 # Remove the nodes we don't want anymore
        new_dim = 6 + 2 * trunc_nodes

        # Update map width
        self.W_est = self.W_est[:trunc_nodes]
        self.W_var = self.W_var[:trunc_nodes]

        # Update state vector and covarience matrix w/ their new lengths
        self.X = self.X[:new_dim]
        self.Sigma = self.Sigma[:new_dim, :new_dim] 
                
        self.map_s = self._calc_arc_lengths(self.M.reshape(-1, 2)) # Let the map know how long it is now
        self.current_s = self._project_onto_spline(self.pose[:2], 0.0) # Find where we are on the new map now everything changed

    @staticmethod
    def _calc_arc_lengths(P: np.ndarray) -> List[float]:
        """Compute map length"""
        s = [0.0]
        for i in range(1, len(P)):
            s.append(s[-1] + np.linalg.norm(P[i] - P[i-1]))
        return s

    def _project_onto_spline(self, point: np.ndarray, current_s: float, window=4) -> float:
        """Compute where robot is on map"""
        total_s = self.map_s[-1]
        num_segments = len(self.map_s) - 1

        if self.loop_closed and total_s > 0:
            current_s = current_s % total_s
            curr_idx = np.clip(np.searchsorted(self.map_s, current_s) - 1, 0, num_segments - 1)
            
            search_indices = [] # use a constrained search so robot doesn't jump to a completely different temporal location on map.
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
    
    def _get_polyline_properties(self, s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float]:
        """Returns [x,y], tangent, normal, node i, node i+1, and interpolation u (0-1)."""
        idx = np.clip(np.searchsorted(self.map_s, s) - 1, 0, len(self.map_s) - 2) # grab current map index
        denom = self.map_s[idx+1] - self.map_s[idx]  
        u = (s - self.map_s[idx]) / denom if denom > 0 else 0.0 # Grab LERP factor (0->1)

        p0, p1 = self.M[2*idx:2*idx+2], self.M[2*idx+2:2*idx+4]
        diff = p1 - p0
        dist = np.linalg.norm(diff)
        t = diff / dist if dist > 1e-6 else np.array([1.0, 0.0])
        n = np.array([-t[1], t[0]])
        return (1 - u) * p0 + u * p1, t, n, int(idx), int(idx + 1), u # Return LERP'd line pos, t_vec, n_vec, node idxs, u

if __name__ == "__main__":
    np.random.seed(42)
    
    # load reference track
    gt_track = np.load("track_width.npz")

    center = gt_track["center"]
    left   = gt_track["left"]
    right  = gt_track["right"]
    width  = gt_track["width"]
    s      = gt_track["s"]
    # init sim
    sim_cfg, ekf_cfg, controller_config = SimConfig(), EKFConfig(), PurePursuitConfig()
    sim = Sim(sim_cfg, gt_track)
    obs_provider = SimCam(sim)
    
    init_pose = np.array([
        center[0, 0] + 0.05,
        center[0, 1] - 0.05,
        np.pi / 2 + 0.02
    ])

    init_map = np.array([
        center[0],
        center[10]
    ])
    
    ekf = EKFSLACE(ekf_cfg, init_pose, init_map)
    
    # init vis
    visualizer = LiveVisualizer(gt_track, map_skip_val=100, record_video=True, video_path="slace.mp4")
    controller = PurePursuitController(controller_config)
    obs_packet = None
    # loop
    cmd_v = [0,0,0]
    mpc_trajectory = None
    for step in range(sim_cfg.sim_steps):
        cmd_v = controller.compute_commands(ekf.pose, ekf.M, ekf.seg_idx)       

        imu_packet, odom_packet = sim.step(*cmd_v)

        ekf.predict_imu(imu_packet)

        obs_packet = obs_provider.get_observations()

        if odom_packet is not None:
            ekf.update_odom(odom_packet)
            
        if sim.is_camera_ready:
            ekf.update_SLACE(obs_packet)
            if obs_packet.loop_closure_triggered and not ekf.loop_closed:
                print(f"--- Step {step}: LOOP CLOSED SUCCESSFULLY ---")
        visualizer.update(step, sim, ekf, obs_packet, planned_path=mpc_trajectory)
    plt.ioff() 
    plt.show()
    visualizer.close()
import numpy as np
from typing import Tuple
from Sim import normalize_angle

class PurePursuitController:
    """
    Decoupled geometric path tracker. 
    Swap this class out entirely when transitioning to MPC.
    """
    def __init__(self, config):
        self.cfg = config

    def compute_commands(self, true_pose: np.ndarray, gt_track: np.ndarray, closest_idx: int) -> Tuple[float, float, float]:
        """Calculates body-frame velocity targets to track the lookahead point."""
        N = len(gt_track)
        target = gt_track[(closest_idx + self.cfg.lookahead_idx) % N]
        
        # Calculate heading error to target point
        heading_err = normalize_angle(
            np.arctan2(target[1] - true_pose[1], target[0] - true_pose[0]) - true_pose[2]
        )
        
        # Output target velocities: [cmd_vx, cmd_vy, cmd_w]
        cmd_vx = self.cfg.target_v
        cmd_vy = 0.0  # Pure pursuit does not natively utilize lateral strafe
        cmd_w = self.cfg.steering_gain * heading_err
        
        return cmd_vx, cmd_vy, cmd_w
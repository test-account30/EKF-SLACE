import numpy as np
from typing import Tuple
from utils import normalize_angle
import osqp
import scipy.sparse as sparse
from utils import *
from Config import *

class PurePursuitController:
    """
    Decoupled geometric path tracker. 
    """
    def __init__(self, config):
        self.cfg = config

    def compute_commands(
        self,
        pose: np.ndarray,
        map_nodes: np.ndarray,
        closest_idx: int
    ) -> Tuple[float, float, float]:
        """Pure pursuit controller using EKF pose + SLACE map."""
        
        M = map_nodes.reshape(-1, 2)
        N = len(M)

        if N < 2:
            return 0.0, 0.0, 0.0

        # pick lookahead target on map
        target = M[(closest_idx + self.cfg.lookahead_idx) % N]

        # heading error to target
        heading_err = normalize_angle(
            np.arctan2(target[1] - pose[1], target[0] - pose[0]) - pose[2]
        )

        # outputs
        cmd_vx = self.cfg.target_v
        cmd_vy = 0.0
        cmd_w = self.cfg.steering_gain * heading_err

        return cmd_vx, cmd_vy, cmd_w
    
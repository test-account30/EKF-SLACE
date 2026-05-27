# EKF-SLACE
# SLACE

SLACE (Simultaneous Localisation And Curve Estimation) is a lightweight EKF-based curve estimation system written in Python.

The system jointly estimates robot pose and a sparse centreline representation of the environment using local observations of a track boundary or line structure. Instead of discrete landmark points, the map is represented as a deformable polyline that evolves online as new measurements are incorporated.

A sliding-window update scheme is used so that each measurement only affects a local subset of nearby map nodes. This keeps computation bounded with respect to map length, enabling near constant-time updates in practice for long trajectories.

Rather than aiming for globally consistent Cartesian reconstruction of the environment, SLACE is designed to maintain a locally smooth and stable Frenet-frame representation of the observed path. This makes it well-suited for finite-horizon control applications such as MPC, where local curvature and tangent consistency are more important than global map accuracy.

https://github.com/user-attachments/assets/2ed6bc52-8736-458a-b4d4-5efe82431039

The current setup uses a simulated camera pipeline which:

* samples points from a ground truth track
* converts them into local camera observations
* fits a local line/curve estimate
* feeds the observations into the EKF

The EKF then:

* predicts robot motion from odometry
* projects observations onto the estimated curve
* updates robot pose
* updates nearby map nodes
* incrementally augments the map
* performs a simple loop closure correction once the lap closes

The estimator core is seperated from the simulation and visualisation layers so the simulated observation provider can later be replaced with real hardware or a CV pipeline.

## Running

Install dependencies:

```bash
pip install numpy matplotlib
winget install "FFmpeg (Essentials Build)" 
```

Run the simulator:

```bash
python SLACE.py
```

A live visualisation window should open showing:

* ground truth track
* estimated centreline map
* robot trajectory
* local observations
* covariance ellipses

## Notes

This project is mainly experimental/research-style code rather than a polished robotics framework. The current implementation uses a sparse polyline representation rather than splines or occupancy grids.

## Appendix

This section summarises the SLACE EKF formulation using a simplified and GitHub-safe notation.

---

### 1. State Representation

Robot state:

$$
x = [x, y, \theta]^T
$$

Map representation (polyline nodes):

$$
M = [x_1, y_1, x_2, y_2, ..., x_N, y_N]^T
$$

Covariances:

$$
P \in \mathbb{R}^{3 \times 3}, \quad S_M \in \mathbb{R}^{2N \times 2N}
$$

---

### 2. Motion Model (Prediction Step)

$$
x_{k+1} = x_k + v \Delta t \cos(\theta)
$$

$$
y_{k+1} = y_k + v \Delta t \sin(\theta)
$$

$$
\theta_{k+1} = \theta_k + \omega \Delta t
$$

Covariance propagation:

$$
P = F P F^T + Q
$$

Jacobian:

$$
F =
\begin{bmatrix}
1 & 0 & -v \Delta t \sin(\theta) \\
0 & 1 & v \Delta t \cos(\theta) \\
0 & 0 & 1
\end{bmatrix}
$$

---

### 3. Line Observation Model (Frenet Frame)

Measurement from local point cloud:

$$
z = [b, \theta_l]^T
$$

where:
- b = lateral offset to line
- θ_l = local line direction

Predicted measurement:

$$
z_{hat} = [-C \cdot n, -atan2(t_y, t_x)]^T
$$

Residual:

$$
r = z - z_{hat}
$$

---

### 4. EKF Update (Robot Pose)

Jacobian:

$$
H =
\begin{bmatrix}
n_x & n_y & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

Innovation covariance:

$$
S = H P H^T + R + S_M
$$

Kalman gain:

$$
K = P H^T S^{-1}
$$

State update:

$$
x = x + K r
$$

Covariance update:

$$
P = (I - K H) P (I - K H)^T + K R K^T
$$

---

### 5. Map Update (Sliding Window EKF)

Global observation transform:

$$
z_{global} = [x, y] + R(\theta) z_{local}
$$

Residual:

$$
r_M = z_{global} - C(s)
$$

Map update:

$$
M_{active} = M_{active} + K_M r_M
$$

---

### 6. Sliding Window

Only map nodes near current arc-length s are updated:

$$
W(s) = \{ i : |s_i - s| < w \}
$$

This gives constant-time updates:

$$
O(|W|) \approx O(1)
$$

---

### 7. Map Augmentation

New node is added when:

$$
d_{forward} > d_{add}, \quad |d_{lat}| < w_{lane}
$$

New point:

$$
p_{new} = p_{last} + d_{add} t
$$

---

### 8. Loop Closure (Simple Alignment)

Angle correction:

$$
d\theta = \theta_{start} - \theta_{end}
$$

Rotation:

$$
R =
\begin{bmatrix}
cos(d\theta) & -sin(d\theta) \\
sin(d\theta) & cos(d\theta)
\end{bmatrix}
$$

Position correction:

$$
p = R(p - p_a) + p_a + d_p
$$

---

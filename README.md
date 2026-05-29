# EKF-SLACE

SLACE (Simultaneous Localization And Curve Estimation) is a lightweight EKF-based curve estimation and localisation system written in Python.

The system jointly estimates robot pose, body-frame velocities and a sparse centreline representation of the environment using a single shared EKF state vector. Rather than representing the environment as discrete landmarks or occupancy grids, the map is modelled as a deformable polyline that evolves online as new observations are incorporated.

The estimator uses a multirate EKF structure to fuse asynchronous IMU, wheel odometry and map-based observations. High-rate IMU propagation is used for continuous motion prediction, wheel odometry updates constrain body-frame velocity estimates and lower-rate SLACE updates align local observations against the estimated map geometry.

The system is designed around maintaining a locally smooth and dynamically consistent Frenet-frame representation of the observed path rather than producing globally optimal Cartesian reconstruction. This makes it well suited for local planning and finite-horizon control applications such as MPC, where local curvature consistency and stable relative geometry are more important than globally drift-free mapping.



https://github.com/user-attachments/assets/d08a8cee-baf0-4849-90f0-afe031f50cbc




The current simulated perception pipeline:

* samples points from a ground truth track
* converts them into local camera-frame observations
* extracts local line geometry using SVD
* feeds Frenet-frame observations into the EKF

The EKF then:

* propagates pose using IMU acceleration and angular velocity
* estimates body-frame velocities
* fuses wheel odometry measurements
* projects observations onto the estimated polyline map
* jointly updates robot and map states
* incrementally augments the map online
* performs loop closure alignment once the lap closes

The estimator core is separated from the simulation and visualisation layers so the simulated observation provider can later be replaced with real hardware or a CV pipeline.

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

## System Overview

### State Vector

SLACE uses a single shared EKF state vector containing:

* robot pose
* body-frame velocity estimates
* deformable polyline map nodes

The joint formulation allows correlations between robot motion and map geometry to be preserved directly within the covariance matrix.

State structure:

$$
X =
[x,\ y,\ \theta,\ v_x,\ v_y,\ \omega,\ m_1^x,\ m_1^y,\ ...]^T
$$

### Sensor Fusion Structure

The estimator operates as a multirate EKF:

| Source                | Role                                  |
| --------------------- | ------------------------------------- |
| IMU                   | High-rate prediction                  |
| Wheel Odometry        | Velocity correction                   |
| SLACE Geometry Update | Map-relative localisation and mapping |

The IMU prediction step propagates pose and velocity continuously using measured accelerations and angular velocity.

Wheel odometry constrains body-frame velocity drift.

SLACE updates use local geometric observations extracted from nearby track structure to jointly correct both robot pose and map geometry.

### Map Representation

The environment is represented as a sparse polyline rather than:

* occupancy grids
* point clouds
* spline surfaces
* landmark graphs

Each segment stores local geometric structure while remaining lightweight enough for online joint optimisation inside the EKF.

Observations are projected onto nearby map segments using local arc-length parameterisation.

### Observation Model

Local point observations are converted into a Frenet-frame representation:

$$
z = [b,\ \theta_l]^T
$$

where:

* (b) is lateral offset from the local track estimate
* (\theta_l) is local line orientation

Local line geometry is extracted using SVD over nearby observed points.

### Loop Closure

When the estimated trajectory returns near the starting region, the system performs a lightweight loop closure correction by enforcing consistency between the map tail and the initial anchor region.

The map is then truncated and converted into a closed-loop representation.

## Notes

This project is primarily experimental/research-style code rather than a polished robotics framework.

The current implementation focuses on lightweight geometric SLAM using a sparse deformable polyline representation and joint EKF estimation rather than globally optimal graph-based optimisation.

The estimator architecture is intentionally modular so the simulated observation pipeline can later be replaced with real camera, lidar, or CV-based feature extraction systems.

## Appendix

This section summarises the simplified EKF formulation used by SLACE.

---

### 1. Joint State Representation

Robot pose and velocity:

$$
x_r =
[x,\ y,\ \theta,\ v_x,\ v_y,\ \omega]^T
$$

Polyline map:

$$
M =
[x_1,\ y_1,\ x_2,\ y_2,\ ...]^T
$$

Joint EKF state:

$$
X =
[x_r,\ M]^T
$$

Joint covariance:

$$
\Sigma \in \mathbb{R}^{N \times N}
$$

---

### 2. IMU Prediction Model

Pose propagation:

$$
x_{k+1} =
x_k + (v_x \cos\theta - v_y \sin\theta)\Delta t
$$

$$
y_{k+1} =
y_k + (v_x \sin\theta + v_y \cos\theta)\Delta t
$$

Heading update:

$$
\theta_{k+1} =
\theta_k + \omega \Delta t
$$

Velocity propagation:

$$
v_{x,k+1} = v_{x,k} + a_x \Delta t
$$

$$
v_{y,k+1} = v_{y,k} + a_y \Delta t
$$

Covariance propagation:

$$
\Sigma =
F \Sigma F^T + Q
$$

---

### 3. Odometry Velocity Update

Wheel odometry directly constrains body-frame velocity states:

$$
z_{odom} =
[v_x,\ v_y,\ \omega]^T
$$

Residual:

$$
r =
z_{odom} - \hat{z}_{odom}
$$

Kalman update:

$$
X = X + Kr
$$

---

### 4. Frenet Observation Model

Local line observations:

$$
z =
[b,\ \theta_l]^T
$$

Predicted observation:

$$
\hat{z} =
[-C \cdot n,\ -atan2(t_y,\ t_x)]^T
$$

Residual:

$$
r =
z - \hat{z}
$$

---

### 5. Joint EKF Update

The observation Jacobian spans both robot and map states:

$$
H =
[H_r\ \ H_m]
$$

Innovation covariance:

$$
S =
H \Sigma H^T + R
$$

Kalman gain:

$$
K =
\Sigma H^T S^{-1}
$$

State update:

$$
X =
X + Kr
$$

Joseph-form covariance update:

$$
\Sigma =
(I - KH)\Sigma(I - KH)^T + KRK^T
$$

---

### 6. Polyline Projection

Observed points are projected onto nearby polyline segments using local arc-length parameterisation:

$$
s^* =
argmin_s ||p - C(s)||
$$

where:

* (C(s)) is the current polyline estimate
* (s) is arc length along the map

---

### 7. Map Augmentation

New map nodes are added online when observations extend sufficiently beyond the current map frontier.

New node placement:

$$
p_{new} =
p_{last} + d_{aug} \hat{t}
$$

where:

* (d_{aug}) is augmentation spacing
* (\hat{t}) is estimated local tangent direction

---

### 8. Loop Closure Constraint

Loop closure applies a consistency constraint between the map tail and initial anchor:

$$
r_{lc} =
p_{start} - p_{end}
$$

A final EKF update distributes the correction through the joint covariance structure.

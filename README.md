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

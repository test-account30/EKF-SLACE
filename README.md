# EKF-SLACE
# SLACE

SLACE (Simultaneous Localisation And Curve Estimation) is a lightweight EKF-based curve SLAM project written in Python.

The system estimates both robot pose and a sparse centreline map at the same time using local observations of a track boundary/line. Instead of using traditional landmarks, the map is represented as a deformable polyline which is updated online as new observations arrive.

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

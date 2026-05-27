import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Load Existing Track
# -------------------------------------------------------------------------
track = np.load("custom_gt_track.npy")

print(f"Loaded track with {len(track)} points.")

# -------------------------------------------------------------------------
# Reverse Direction While Keeping Same Start Point
# -------------------------------------------------------------------------
# Remove duplicate endpoint if loop is explicitly closed
if np.allclose(track[0], track[-1]):
    track = track[:-1]

# Reverse traversal direction
track_reversed = track[::-1]

# Re-anchor so the first point is still at the origin
origin = track_reversed[0].copy()
track_reversed -= origin

# Close loop again
track_reversed = np.vstack([track_reversed, track_reversed[0]])

# -------------------------------------------------------------------------
# Save
# -------------------------------------------------------------------------
np.save("custom_gt_track_reversed.npy", track_reversed)

print("Saved reversed track to 'custom_gt_track_reversed.npy'")

# -------------------------------------------------------------------------
# Verification Plot
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 7))

plt.plot(
    track_reversed[:, 0],
    track_reversed[:, 1],
    'g-',
    label='Reversed Track'
)

# Direction arrows
step = 40
dx = np.diff(track_reversed[:, 0])
dy = np.diff(track_reversed[:, 1])

plt.quiver(
    track_reversed[:-1:step, 0],
    track_reversed[:-1:step, 1],
    dx[::step],
    dy[::step],
    color='red',
    scale=20,
    label='Direction'
)

# Start point
plt.scatter(
    track_reversed[0, 0],
    track_reversed[0, 1],
    color='blue',
    s=100,
    zorder=5,
    label='Start'
)

plt.axis('equal')
plt.grid(True, linestyle=':')
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.title("Reversed Track Verification")
plt.legend()
plt.show()
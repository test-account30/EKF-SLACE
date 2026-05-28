import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

REAL_WORLD_IMAGE_WIDTH_METERS = 7.0 

image_path = 'tools/image.png'

try:
    img = plt.imread(image_path)
    # img.shape[1] gives the width of the image in pixels
    img_pixel_width = img.shape[1]
    img_pixel_height = img.shape[0]
    print(f"Loaded image: {img_pixel_width}x{img_pixel_height} pixels.")
except FileNotFoundError:
    print(f"Warning: '{image_path}' not found. Using a blank canvas layout placeholder.")
    img = np.ones((600, 800, 3))
    img_pixel_width = 800
    img_pixel_height = 600

# Calculate exact conversion scale factor (meters per pixel)
scale_pixel_to_meter = REAL_WORLD_IMAGE_WIDTH_METERS / img_pixel_width
print(f"Calculated Scale Factor: {scale_pixel_to_meter:.5f} meters per pixel.")

# -------------------------------------------------------------------------
# 2. Interactive Point Collection
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 8))
ax.imshow(img)
ax.set_title(
    "LEFT-CLICK to place spline nodes along the track.\n"
    "RIGHT-CLICK (or Backspace) to undo. MIDDLE-CLICK (or Enter) to finish."
)

print("\nClick on the image window to trace the track centerline. Press Enter when finished.")
points = plt.ginput(n=-1, timeout=0, show_clicks=True)
plt.close()

if len(points) < 4:
    raise ValueError("Please click at least 4 points to form a valid track loop.")

# Convert to numpy array
points = np.array(points)

# -------------------------------------------------------------------------
# 3. Handle Direction Ordering Swap
# -------------------------------------------------------------------------
print(f"\nYou clicked {len(points)} points.")
swap_order = input("Do you want to reverse/swap the track progression direction? (y/n): ").strip().lower()

if swap_order == 'y':
    points = points[::-1, :]
    print("-> Track point ordering reversed.")
else:
    print("-> Original track point ordering retained.")

# Close the loop securely
points = np.vstack([points, points[0]])

# -------------------------------------------------------------------------
# 4. Arc-Length Parameterization & Spline Fitting
# -------------------------------------------------------------------------
dx = np.diff(points[:, 0])
dy = np.diff(points[:, 1])
distances = np.sqrt(dx**2 + dy**2)
s_accumulated = np.concatenate([[0], np.cumsum(distances)])

# Periodic cubic spline matching pixel-space constraints
spline_x = CubicSpline(s_accumulated, points[:, 0], bc_type="periodic")
spline_y = CubicSpline(s_accumulated, points[:, 1], bc_type="periodic")

# --- CHANGED: Sample 1000 dense tracking coordinates instead of 500 ---
s_dense = np.linspace(0, s_accumulated[-1], 1000)
dense_track = np.vstack([spline_x(s_dense), spline_y(s_dense)]).T

# -------------------------------------------------------------------------
# 5. Conversion to Metric Space & Coordinate Realignment
# -------------------------------------------------------------------------
# Flip the Y-axis calculation because image matrices count rows from the top-down,
# but physical standard simulations treat positive Y as going UP.
dense_track[:, 1] = img_pixel_height - dense_track[:, 1]

# Center coordinate system so first clicked point becomes origin
origin = dense_track[0].copy()
dense_track -= origin

# Apply calculated exact pixel-to-meter dimensioning ratio scale
dense_track *= scale_pixel_to_meter

# -------------------------------------------------------------------------
# 6. Save & Verification Plot
# -------------------------------------------------------------------------
np.save('custom_gt_track.npy', dense_track)
print(f"\nSuccess! Saved dense track layout to 'custom_gt_track.npy'.")
print(f"Track spans from X: [{dense_track[:,0].min():.2f}m to {dense_track[:,0].max():.2f}m]")
print(f"Track spans from Y: [{dense_track[:,1].min():.2f}m to {dense_track[:,1].max():.2f}m]")

# Verification Plot
plt.figure(figsize=(10, 7))
# Plot path with directional arrows to verify direction swap choice
plt.plot(dense_track[:, 0], dense_track[:, 1], 'g-', label='Track Centerline')

# --- CHANGED: Updated indexing step from 40 to 80 to maintain clean arrow density ---
plt.quiver(dense_track[:-1:80, 0], dense_track[:-1:80, 1], 
           np.diff(dense_track[::80, 0]), np.diff(dense_track[::80, 1]), 
           color='red', scale=20, label='Direction Vector')
plt.scatter(dense_track[0, 0], dense_track[0, 1], color='blue', s=100, zorder=5, label='Robot Start Location')

plt.axis('equal')
plt.grid(True, linestyle=':')
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.title("Processed Metric Track Ground Truth Verification (1000 Points)")
plt.legend()
plt.show()
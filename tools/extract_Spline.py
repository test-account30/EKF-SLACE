import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os

REAL_WORLD_IMAGE_WIDTH_METERS = 7.0
image_path = "tools/image.png"

CHECKPOINT_DIR = "track_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save(name, **data):
    path = os.path.join(CHECKPOINT_DIR, name)
    np.savez(path, **data)
    print(f"[saved] {path}")

# ============================================================
# LOAD IMAGE
# ============================================================
img = plt.imread(image_path)
h, w = img.shape[:2]
scale = REAL_WORLD_IMAGE_WIDTH_METERS / w

# ============================================================
# INPUT CENTRELINE (PIXELS)
# ============================================================
fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title("Click centreline (closed loop)")

pts_px = np.array(plt.ginput(n=-1, timeout=0))
plt.close()

if len(pts_px) < 4:
    raise ValueError("Need at least 4 points")

save("01_raw.npz", pts_px=pts_px)

# ============================================================
# WIDTH (PIXELS)
# ============================================================
width_px = np.ones(len(pts_px)) * (0.5 / scale)
save("02_width.npz", width_px=width_px)

# ============================================================
# ARC LENGTH (OPEN PARAM, CLEAN)
# ============================================================
d = np.diff(pts_px, axis=0)
ds = np.linalg.norm(d, axis=1)
s = np.concatenate([[0], np.cumsum(ds)])
save("03_arclength.npz", s=s)

# ============================================================
# SPLINE INPUT (FIXED CLOSED LOOP)
# IMPORTANT: duplicate first point ONLY HERE
# ============================================================
pts_closed = np.vstack([pts_px, pts_px[0]])
width_closed = np.append(width_px, width_px[0])
s_closed = np.append(s, s[-1] + np.linalg.norm(pts_px[0] - pts_px[-1]))

save("04_spline_inputs.npz",
     pts_closed=pts_closed,
     width_closed=width_closed,
     s_closed=s_closed)

# ============================================================
# DRAG EDITOR (PIXEL SPACE)
# ============================================================
def compute_normals(P):
    n = len(P)
    T = np.zeros_like(P)

    for i in range(n):
        prev = P[i - 1]
        nxt = P[(i + 1) % n]
        T[i] = nxt - prev

    T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    N = np.stack([-T[:, 1], T[:, 0]], axis=1)
    return N

class TrackEditor:
    def __init__(self, pts, width, img):
        self.pts = pts.copy()
        self.width = width.copy()
        self.img = img

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(img)
        self.ax.set_aspect("equal")

        self.drag_i = None

        self.line, = self.ax.plot([], [], "g-")
        self.sc_c = self.ax.scatter([], [], c="red", s=40)
        self.sc_w = self.ax.scatter([], [], c="blue", s=40)

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)

        self.update()

    def rebuild(self):
        self.N = compute_normals(self.pts)
        self.wpts = self.pts + self.width[:, None] * self.N

    def update(self):
        self.rebuild()
        self.line.set_data(self.pts[:, 0], self.pts[:, 1])
        self.sc_c.set_offsets(self.pts)
        self.sc_w.set_offsets(self.wpts)

        self.ax.set_xlim(0, self.img.shape[1])
        self.ax.set_ylim(self.img.shape[0], 0)

        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        d = np.linalg.norm(self.wpts - [event.xdata, event.ydata], axis=1)
        self.drag_i = np.argmin(d)

    def on_release(self, event):
        self.drag_i = None

    def on_move(self, event):
        if self.drag_i is None or event.inaxes != self.ax:
            return

        p = self.pts[self.drag_i]
        delta = np.array([event.xdata, event.ydata]) - p

        self.width[self.drag_i] = np.dot(delta, self.N[self.drag_i])
        self.update()

    def run(self):
        plt.show()
        return self.pts, self.width

editor = TrackEditor(pts_px, width_px, img)
pts_px, width_px = editor.run()

save("05_after_drag.npz", pts_px=pts_px, width_px=width_px)

# ============================================================
# FINAL CLOSED SPLINE (NO CRASH VERSION)
# ============================================================
pts_closed = np.vstack([pts_px, pts_px[0]])
width_closed = np.append(width_px, width_px[0])

d = np.diff(pts_closed, axis=0)
ds = np.linalg.norm(d, axis=1)
s_closed = np.concatenate([[0], np.cumsum(ds)])

L = s_closed[-1]

sx = CubicSpline(s_closed, pts_closed[:, 0], bc_type="periodic")
sy = CubicSpline(s_closed, pts_closed[:, 1], bc_type="periodic")
sw = CubicSpline(s_closed, width_closed, bc_type="periodic")

# ============================================================
# HIGH RES OUTPUT (>=1000)
# ============================================================
N_OUT = 1500
s_dense = np.linspace(0, L, N_OUT)

x = sx(s_dense)
y = sy(s_dense)

dx = sx(s_dense, 1)
dy = sy(s_dense, 1)

norm = np.sqrt(dx**2 + dy**2) + 1e-9
tx, ty = dx / norm, dy / norm
nx, ny = -ty, tx

w = sw(s_dense)

left = np.column_stack([x - w * nx, y - w * ny])
right = np.column_stack([x + w * nx, y + w * ny])

# ============================================================
# SHIFT ORIGIN (0,0 START)
# ============================================================
origin = np.array([x[0], y[0]])

x -= origin[0]
y -= origin[1]
left -= origin
right -= origin

# ============================================================
# METRES CONVERSION
# ============================================================
center_m = np.column_stack([x, y]) * scale
left_m = left * scale
right_m = right * scale
w_m = w * scale
s_m = s_dense * scale

save("06_output_m.npz",
     center=center_m,
     left=left_m,
     right=right_m,
     width=w_m,
     s=s_m)

# ============================================================
# PLOT CHECK
# ============================================================
plt.figure()
plt.plot(center_m[:, 0], center_m[:, 1], "g")
plt.plot(left_m[:, 0], left_m[:, 1], "c")
plt.plot(right_m[:, 0], right_m[:, 1], "b")
plt.axis("equal")
plt.grid(True)
plt.title("FINAL CLOSED TRACK (stable + draggable widths + no spline crash)")
plt.show()
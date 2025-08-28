import pandas as pd
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
USE_IMU = False  # <-- start with GPS-only; set True after you confirm <20 m
ACC_IN_G = False  # set True if your ax, ay are in "g" units, not m/s^2
ACC_CLIP = 5.0    # m/s^2; clip extreme spikes to avoid blow-ups
FALLBACK_DT = 0.02  # seconds (50 Hz) if no timestamps available

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/processed/cleaned_dataset_local.csv")

# Must have x_m, y_m (meters). Optional: time, ax, ay
if "x_m" not in df or "y_m" not in df:
    raise ValueError("cleaned_dataset_local.csv must contain 'x_m' and 'y_m' columns.")

# Load origin (used later by 04 script; not strictly needed here)
with open("data/processed/origin.txt", "r") as f:
    lat0, lon0 = map(float, f.read().split(","))

# -----------------------------
# TIMESTAMPS -> dt per step
# -----------------------------
if "time" in df.columns:
    # Try to build dt from timestamps
    t = pd.to_datetime(df["time"], errors="coerce")
    if t.isna().all():
        dt_series = pd.Series(FALLBACK_DT, index=df.index)
    else:
        t = t.fillna(method="ffill").fillna(method="bfill")
        dt_vals = (t.view("int64").to_numpy() / 1e9)
        dt_series = pd.Series(np.diff(dt_vals, prepend=dt_vals[0]), index=df.index)
        # clamp dt to reasonable bounds
        dt_series = dt_series.clip(lower=0.01, upper=0.2)  # 5–100 Hz
else:
    dt_series = pd.Series(FALLBACK_DT, index=df.index)

# -----------------------------
# OPTIONAL IMU PREP
# -----------------------------
if USE_IMU and ("ax" in df.columns) and ("ay" in df.columns):
    ax = df["ax"].astype(float).copy()
    ay = df["ay"].astype(float).copy()

    # If accelerometer is in "g", convert to m/s^2
    if ACC_IN_G:
        ax *= 9.80665
        ay *= 9.80665

    # Simple de-bias using first N samples (assumes near-stationary start)
    N_bias = min(250, len(df))  # ~5s at 50 Hz
    ax_bias = ax.iloc[:N_bias].median()
    ay_bias = ay.iloc[:N_bias].median()
    ax = (ax - ax_bias).clip(-ACC_CLIP, ACC_CLIP)
    ay = (ay - ay_bias).clip(-ACC_CLIP, ACC_CLIP)

    imu_available = True
else:
    imu_available = False

# -----------------------------
# KALMAN FILTER SETUP (CV model)
# State: [x, y, vx, vy]
# -----------------------------
# Initialize at first GPS point; velocity zero
x = np.array([[df["x_m"].iloc[0]],
              [df["y_m"].iloc[0]],
              [0.0],
              [0.0]])

# Initial covariance
P = np.diag([10.0, 10.0, 10.0, 10.0])

# Measurement matrix: GPS observes position only
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]], dtype=float)

# Measurement noise (trust GPS strongly)
gps_std = 1.5  # meters 1-sigma; tune if your GPS is better/worse
R = np.diag([gps_std**2, gps_std**2])

fused_x = []
fused_y = []

# -----------------------------
# FILTER LOOP
# -----------------------------
for i in range(len(df)):
    dt = float(dt_series.iloc[i])

    # State transition (CV)
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]], dtype=float)

    # Process noise: nearly CV; small acceleration uncertainty
    # (tuned for stability; not too small to avoid divergence)
    accel_std = 1.0  # m/s^2 nominal
    q = accel_std**2
    Q = np.array([
        [0.25*dt**4*q, 0,            0.5*dt**3*q, 0           ],
        [0,            0.25*dt**4*q, 0,            0.5*dt**3*q],
        [0.5*dt**3*q,  0,            dt**2*q,     0           ],
        [0,            0.5*dt**3*q,  0,            dt**2*q    ]
    ], dtype=float)

    # Control input (accel) – DISABLED by default
    if USE_IMU and imu_available:
        B = np.array([[0.5*dt**2, 0],
                      [0,          0.5*dt**2],
                      [dt,         0],
                      [0,          dt       ]], dtype=float)
        u = np.array([[ax.iloc[i]],
                      [ay.iloc[i]]], dtype=float)
    else:
        B = np.zeros((4, 2), dtype=float)
        u = np.zeros((2, 1), dtype=float)

    # Predict
    x = F @ x + B @ u
    P = F @ P @ F.T + Q

    # GPS measurement
    xm = df["x_m"].iloc[i]
    ym = df["y_m"].iloc[i]
    if not (np.isnan(xm) or np.isnan(ym)):
        z = np.array([[xm], [ym]], dtype=float)
        y_res = z - (H @ x)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y_res
        P = (np.eye(4) - K @ H) @ P

    fused_x.append(float(x[0, 0]))
    fused_y.append(float(x[1, 0]))

# Save fused results
df["fused_x"] = fused_x
df["fused_y"] = fused_y
df.to_csv("data/processed/fused_estimates.csv", index=False)
print("✅ Fused estimates saved to data/processed/fused_estimates.csv")
print(f"   USE_IMU={USE_IMU}, gps_std={gps_std} m, accel_std=0.5 m/s²")


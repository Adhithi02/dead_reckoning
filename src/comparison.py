import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
np.random.seed(42)
df = pd.read_csv("data/data.csv")

# Convert to UTM
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
utm = df.apply(lambda row: transformer.transform(row["longitude"], row["latitude"]), axis=1)
utm = np.array(utm.tolist())
df["x"] = utm[:, 0]
df["y"] = utm[:, 1]

x_gps = df["x"].values
y_gps = df["y"].values

# EKF simulation (rolling average)
x_ekf = pd.Series(x_gps).rolling(window=5, min_periods=1).mean().values
y_ekf = pd.Series(y_gps).rolling(window=5, min_periods=1).mean().values

# ML-only simulation
dx = np.diff(x_gps, prepend=x_gps[0])
dy = np.diff(y_gps, prepend=y_gps[0])
x_ml = [x_gps[0]]
y_ml = [y_gps[0]]
for i in range(1, len(dx)):
    x_ml.append(x_ml[-1] + dx[i] + np.random.normal(0, 5))
    y_ml.append(y_ml[-1] + dy[i] + np.random.normal(0, 5))
x_ml = np.array(x_ml)
y_ml = np.array(y_ml)

# Hybrid = EKF + ML correction
x_hybrid = 0.7 * x_ekf + 0.3 * x_ml
y_hybrid = 0.7 * y_ekf + 0.3 * y_ml

# Plot
plt.figure(figsize=(12, 8))
plt.plot(x_gps - x_gps[0], y_gps - y_gps[0], label="Raw GPS", alpha=0.6, linewidth=2)
plt.plot(x_ml - x_ml[0], y_ml - y_ml[0], '--', label="ML Only", alpha=0.6)
plt.plot(x_ekf - x_ekf[0], y_ekf - y_ekf[0], '--', label="EKF Only", alpha=0.8)
plt.plot(x_hybrid - x_hybrid[0], y_hybrid - y_hybrid[0], label="Hybrid (EKF + ML)", linewidth=3)

plt.title("Trajectory Comparison: Raw GPS vs ML vs EKF vs Hybrid")
plt.xlabel("Relative X (meters)")
plt.ylabel("Relative Y (meters)")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()

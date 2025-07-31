import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
from sklearn.metrics import mean_squared_error

# === Load base GPS data ===
df = pd.read_csv("data/data.csv")

# === Convert GPS (lat, lon) to UTM coordinates ===
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
utm_coords = df.apply(lambda row: transformer.transform(row["longitude"], row["latitude"]), axis=1)
utm_coords = np.array(utm_coords.tolist())
x_gps, y_gps = utm_coords[:, 0], utm_coords[:, 1]

# === Load predictions ===
ml = pd.read_csv("output/ml_predictions.csv")
ekf = pd.read_csv("output/ekf_predictions.csv")
hybrid = pd.read_csv("output/hybrid_predictions.csv")

x_ml, y_ml = ml["x"].values, ml["y"].values
x_ekf, y_ekf = ekf["x"].values, ekf["y"].values
x_hybrid, y_hybrid = hybrid["x"].values, hybrid["y"].values

# === Adjust GPS length to match predictions ===
gps_start_index = 10  # for ML and EKF alignment
gps_hybrid_start_index = 11  # for hybrid model (SEQ_LEN + 1)

x_gps_ml = x_gps[gps_start_index: gps_start_index + len(x_ml)]
y_gps_ml = y_gps[gps_start_index: gps_start_index + len(y_ml)]

x_gps_hybrid = x_gps[gps_hybrid_start_index: gps_hybrid_start_index + len(x_hybrid)]
y_gps_hybrid = y_gps[gps_hybrid_start_index: gps_hybrid_start_index + len(y_hybrid)]

# === RMSEs ===
def rmse(pred, true):
    return np.sqrt(mean_squared_error(true, pred))

print(f"RMSE (ML only):     {rmse(x_ml, x_gps_ml):.2f} m")
print(f"RMSE (EKF only):    {rmse(x_ekf[:len(x_gps)], x_gps):.2f} m")
print(f"RMSE (Hybrid):      {rmse(x_hybrid, x_gps_hybrid):.2f} m")

# === Plot all trajectories ===
plt.figure(figsize=(12, 8))
plt.plot(x_gps - x_gps[0], y_gps - y_gps[0], label="Raw GPS", alpha=0.6, linewidth=2)
plt.plot(x_ml - x_ml[0], y_ml - y_ml[0], '--', label="ML Only", alpha=0.6)
plt.plot(x_ekf - x_ekf[0], y_ekf - y_ekf[0], '--', label="EKF Only", alpha=0.8)
plt.plot(x_hybrid - x_hybrid[0], y_hybrid - y_hybrid[0], label="Hybrid (EKF + ML)", linewidth=3)

plt.title("Trajectory Comparison: GPS vs ML vs EKF vs Hybrid")
plt.xlabel("Relative X (meters)")
plt.ylabel("Relative Y (meters)")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import cos, radians

# Load fused estimates
df = pd.read_csv("data/processed/fused_estimates.csv")

# Load the same origin as used in 02
with open("data/processed/origin.txt", "r") as f:
    lat0, lon0 = map(float, f.read().split(","))

# Calculate Euclidean error in meters
df["error_m"] = np.sqrt((df["fused_x"] - df["x_m"])**2 +
                        (df["fused_y"] - df["y_m"])**2)

# Error metrics
rmse = np.sqrt(np.mean(df["error_m"]**2))
median_err = np.median(df["error_m"])
max_err = np.max(df["error_m"])

print("📊 Error Metrics:")
print(f"  RMSE   : {rmse:.3f} m")
print(f"  Median : {median_err:.3f} m")
print(f"  Max    : {max_err:.3f} m")

# Print all coordinates in meters
print("\n📍 All Coordinates (Raw vs Fused) [in meters]:")
for i in range(len(df)):
    print(f"Raw: ({df['x_m'].iloc[i]:.3f}, {df['y_m'].iloc[i]:.3f})  "
          f"Fused: ({df['fused_x'].iloc[i]:.3f}, {df['fused_y'].iloc[i]:.3f})  "
          f"Error: {df['error_m'].iloc[i]:.3f} m")

# Convert fused X/Y back to lat/lon for comparison
meters_per_deg_lat = 111320.0
meters_per_deg_lon = 111320.0 * cos(radians(lat0))

df["fused_lat"] = lat0 + df["fused_y"] / meters_per_deg_lat
df["fused_lon"] = lon0 + df["fused_x"] / meters_per_deg_lon

print("\n📍 All Coordinates (Raw vs Fused) [lat/lon]:")
for i in range(len(df)):
    print(f"Raw: ({df['lat'].iloc[i]:.6f}, {df['lon'].iloc[i]:.6f})  "
          f"Fused: ({df['fused_lat'].iloc[i]:.6f}, {df['fused_lon'].iloc[i]:.6f})  "
          f"Error: {df['error_m'].iloc[i]:.3f} m")

# ---- PLOTS ----
plt.figure(figsize=(10, 6))
plt.plot(df["x_m"], df["y_m"], label="Raw GPS", alpha=0.7)
plt.plot(df["fused_x"], df["fused_y"], label="Fused GPS", alpha=0.7)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.title("Raw GPS vs Fused GPS Path")
plt.grid(True)
plt.savefig("data/processed/path_comparison.png", dpi=200)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(df.index, df["error_m"], label="Error (m)", color="red")
plt.xlabel("Sample Index")
plt.ylabel("Error (m)")
plt.title("Error Over Time")
plt.grid(True)
plt.savefig("data/processed/error_over_time.png", dpi=200)
plt.show()

plt.figure(figsize=(6, 4))
plt.hist(df["error_m"], bins=50, color="purple", alpha=0.7)
plt.xlabel("Error (m)")
plt.ylabel("Frequency")
plt.title("Error Distribution")
plt.grid(True)
plt.savefig("data/processed/error_histogram.png", dpi=200)
plt.show()

# Optional: Google Maps links
# for i in range(len(df)):
#     raw_url = f"https://www.google.com/maps?q={df['lat'].iloc[i]},{df['lon'].iloc[i]}"
#     fused_url = f"https://www.google.com/maps?q={df['fused_lat'].iloc[i]},{df['fused_lon'].iloc[i]}"
#     print(f"Raw: {raw_url}")
#     print(f"Fused: {fused_url}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer

# Load your data
df = pd.read_csv("data/data.csv")

# Plot GPS trajectory (lat/lon)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(df["longitude"], df["latitude"], marker='.', linewidth=0.5)
plt.title("Raw GPS (Lat vs Lon)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)

# Convert to UTM for distance/speed analysis
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
utm = df.apply(lambda row: transformer.transform(row["longitude"], row["latitude"]), axis=1)
utm = np.array(utm.tolist())
df["x"] = utm[:, 0]
df["y"] = utm[:, 1]

# Compute displacement (euclidean distance)
dx = df["x"].diff().fillna(0)
dy = df["y"].diff().fillna(0)
dist = np.sqrt(dx**2 + dy**2)
df["distance"] = dist

# Plot total displacement
plt.subplot(1, 2, 2)
df["distance"].plot(title="Stepwise Displacement (meters)")
plt.xlabel("Sample Index")
plt.ylabel("Δ Distance (meters)")
plt.grid(True)

plt.tight_layout()
plt.show()

# Print stats
print("\n=== MOTION STATS ===")
print(f"Total GPS samples: {len(df)}")
print(f"Total displacement: {df['distance'].sum():.2f} meters")
print(f"Max step distance: {df['distance'].max():.2f} meters")
print(f"Mean step distance: {df['distance'].mean():.2f} meters")

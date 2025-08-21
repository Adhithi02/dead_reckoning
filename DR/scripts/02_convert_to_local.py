import pandas as pd
import numpy as np
from math import cos, radians

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_dataset.csv")

# Use correct column names from your file
lat_col = "lat"
lon_col = "lon"

# Reference point (first GPS reading)
lat0 = df[lat_col].iloc[0]
lon0 = df[lon_col].iloc[0]

# Save origin for later use
with open("data/processed/origin.txt", "w") as f:
    f.write(f"{lat0},{lon0}")

# Conversion constants
meters_per_deg_lat = 111320.0
meters_per_deg_lon = 111320.0 * cos(radians(lat0))

# Add local coordinates
df["x_m"] = (df[lon_col] - lon0) * meters_per_deg_lon
df["y_m"] = (df[lat_col] - lat0) * meters_per_deg_lat

# Save
df.to_csv("data/processed/cleaned_dataset_local.csv", index=False)
print("✅ Local coordinates added and saved to data/processed/cleaned_dataset_local.csv")
print(f"✅ Origin saved to data/processed/origin.txt -> ({lat0}, {lon0})")

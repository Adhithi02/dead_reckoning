import re
import pandas as pd

# Input/Output paths
infile = "data/raw/Serial_ACC_GNSS_data.txt"
outfile = "data/processed/cleaned_dataset.csv"

# Regex patterns
imu_re = re.compile(
    r'(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*ACCL: ICM20600 '
    r'ACC_X=(?P<ax>-?\d+) ACC_Y=(?P<ay>-?\d+) ACC_Z=(?P<az>-?\d+) '
    r'GYRO_X=(?P<gx>-?\d+) GYRO_Y=(?P<gy>-?\d+) GYRO_Z=(?P<gz>-?\d+)'
)

gps_re = re.compile(
    r'(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*GPS_SERVICE:.*lat:(?P<lat>[-+]?\d+\.\d+)\s+long:(?P<lon>[-+]?\d+\.\d+)'
)

imu_rows = []
gps_rows = []

# Parse file
with open(infile, "r", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        m_imu = imu_re.search(line)
        if m_imu:
            imu_rows.append(m_imu.groupdict())
            continue

        m_gps = gps_re.search(line)
        if m_gps:
            gps_rows.append(m_gps.groupdict())
            continue

# DataFrames
imu_df = pd.DataFrame(imu_rows)
gps_df = pd.DataFrame(gps_rows)

# Convert to datetime
imu_df["ts"] = pd.to_datetime(imu_df["ts"])
gps_df["ts"] = pd.to_datetime(gps_df["ts"])

# Convert numeric
for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
    imu_df[col] = pd.to_numeric(imu_df[col], errors="coerce")

gps_df["lat"] = pd.to_numeric(gps_df["lat"], errors="coerce")
gps_df["lon"] = pd.to_numeric(gps_df["lon"], errors="coerce")

# Merge nearest GPS into IMU
merged = pd.merge_asof(
    imu_df.sort_values("ts"),
    gps_df.sort_values("ts"),
    on="ts",
    direction="backward",
    tolerance=pd.Timedelta("500ms")
)

# Fill GPS missing values
merged["lat"] = merged["lat"].interpolate(method="linear").ffill().bfill()
merged["lon"] = merged["lon"].interpolate(method="linear").ffill().bfill()

# Save
merged.to_csv(outfile, index=False)
print(f"✅ Cleaned dataset saved to: {outfile}")

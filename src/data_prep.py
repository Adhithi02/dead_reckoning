import pandas as pd
import numpy as np
from pyproj import Transformer
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path, seq_len=10):
    df = pd.read_csv(file_path)

    # GPS to UTM
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
    utm_coords = df.apply(lambda row: transformer.transform(row['longitude'], row['latitude']), axis=1)
    utm_coords = np.array(utm_coords.tolist())
    df['x'] = utm_coords[:, 0]
    df['y'] = utm_coords[:, 1]

    # Compute delta changes
    df['dx'] = df['x'].diff().fillna(0)
    df['dy'] = df['y'].diff().fillna(0)

    # Normalize IMU data
    imu_cols = ['imu_acc_x', 'imu_acc_y', 'imu_acc_z', 'imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z']
    imu_scaler = StandardScaler()
    df[imu_cols] = imu_scaler.fit_transform(df[imu_cols])

    # Build sequences
    X_seq, y_seq = [], []
    for i in range(seq_len, len(df)):
        imu_seq = df[imu_cols].iloc[i-seq_len:i].values
        delta = df[['dx', 'dy']].iloc[i].values
        X_seq.append(imu_seq)
        y_seq.append(delta)

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Normalize delta labels
    delta_scaler = StandardScaler()
    y_seq = delta_scaler.fit_transform(y_seq)

    imu_data = df[imu_cols].values
    gps_data = df[['x', 'y']].values

    return X_seq, y_seq, imu_data, gps_data, df, delta_scaler
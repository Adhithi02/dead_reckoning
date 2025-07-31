import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from correction_train import CorrectionModel
from data_prep import preprocess_data

# === Load correction model ===
model = CorrectionModel(input_size=8)  # input_size=8 because IMU (6) + EKF delta (2)
model.load_state_dict(torch.load("model/ekf_correction_model.pt"))
model.eval()

# === Load data ===
SEQ_LEN = 10
_, _, imu_data, gps_data, df, _ = preprocess_data('data/data.csv', seq_len=SEQ_LEN)

# === Compute EKF path ===
ekf_path = [gps_data[0]]
P = np.eye(2) * 10
Q = np.eye(2) * 0.1
R = np.eye(2) * 5
I = np.eye(2)

for i in range(1, len(imu_data)):
    dt = 0.1
    acc_x, acc_y = imu_data[i, 0], imu_data[i, 1]
    dx = 0.5 * acc_x * dt**2
    dy = 0.5 * acc_y * dt**2
    x_pred = ekf_path[-1][0] + dx
    y_pred = ekf_path[-1][1] + dy
    x_state = np.array([x_pred, y_pred])
    P = P + Q
    z = gps_data[i]
    K = P @ np.linalg.inv(P + R)
    x_state = x_state + K @ (z - x_state)
    P = (I - K) @ P
    ekf_path.append(x_state)

ekf_path = np.array(ekf_path)

# === Build hybrid model input: IMU + EKF relative motion ===
X_seq = []
for i in range(SEQ_LEN, len(imu_data)):
    imu_seq = imu_data[i - SEQ_LEN:i]  # shape: (SEQ_LEN, 6)
    ekf_last = ekf_path[i - 1] - ekf_path[i - SEQ_LEN]  # shape: (2,)
    ekf_delta = np.tile(ekf_last, (SEQ_LEN, 1))         # shape: (SEQ_LEN, 2)
    input_seq = np.hstack([imu_seq, ekf_delta])         # shape: (SEQ_LEN, 8)
    X_seq.append(input_seq)

X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)

# === Predict corrections ===
with torch.no_grad():
    corrections = model(X_tensor).numpy()

# === Apply corrections to EKF path ===
x_ekf = df['x'].rolling(window=5, min_periods=1).mean().values
y_ekf = df['y'].rolling(window=5, min_periods=1).mean().values

x_hybrid = []
y_hybrid = []

for i in range(len(corrections)):
    corrected_x = x_ekf[i + SEQ_LEN] + corrections[i][0]
    corrected_y = y_ekf[i + SEQ_LEN] + corrections[i][1]
    x_hybrid.append(corrected_x)
    y_hybrid.append(corrected_y)

# === Ground truth GPS values ===
x_true = df['x'].iloc[SEQ_LEN + 1:].values
y_true = df['y'].iloc[SEQ_LEN + 1:].values

# === Align lengths ===
min_len = min(len(x_hybrid), len(x_true))
x_hybrid = np.array(x_hybrid[:min_len])
y_hybrid = np.array(y_hybrid[:min_len])
x_true = x_true[:min_len]
y_true = y_true[:min_len]

# === Compute RMSE ===
rmse = np.sqrt(((x_hybrid - x_true) ** 2 + (y_hybrid - y_true) ** 2).mean())
print(f"Hybrid (EKF + ML) RMSE: {rmse:.2f} meters")

# === Save hybrid path ===
pd.DataFrame({'x': x_hybrid, 'y': y_hybrid}).to_csv("output/hybrid_predictions.csv", index=False)

# === Plot ===
plt.figure(figsize=(10, 6))
plt.plot(df['x'] - df['x'][0], df['y'] - df['y'][0], label='Raw GPS', alpha=0.5)
plt.plot(x_hybrid - x_hybrid[0], y_hybrid - y_hybrid[0], '--', label='Hybrid Path (EKF + ML)')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.title('Hybrid Path vs Raw GPS')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()

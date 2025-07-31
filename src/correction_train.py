import torch
import torch.nn as nn
import torch.optim as optim
from data_prep import preprocess_data
import numpy as np

class CorrectionModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# Load data
SEQ_LEN = 10
X_seq, _, imu_data, gps_data, _, _ = preprocess_data('data/data.csv', seq_len=SEQ_LEN)


# Recompute EKF for training correction
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
corrections = gps_data - ekf_path

# Prepare training samples: IMU + EKF error
X_corr, y_corr = [], []
for i in range(SEQ_LEN, len(imu_data)):
    imu_seq = imu_data[i-SEQ_LEN:i]
    ekf_last = ekf_path[i-1] - ekf_path[i-SEQ_LEN]  # relative motion
    imu_plus_ekf = np.hstack([imu_seq, np.tile(ekf_last, (SEQ_LEN, 1))])
    X_corr.append(imu_plus_ekf)
    y_corr.append(corrections[i])

X_corr = torch.tensor(np.array(X_corr), dtype=torch.float32)
y_corr = torch.tensor(np.array(y_corr), dtype=torch.float32)

# Train model
model = CorrectionModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_corr)
    loss = criterion(outputs, y_corr)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), 'model/ekf_correction_model.pt')
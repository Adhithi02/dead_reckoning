import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_prep import preprocess_data

# Load preprocessed data
_, _, imu_data, gps_data, df, _ = preprocess_data('data/data.csv')


# EKF smoothing via rolling average
x_ekf = pd.Series(df['x']).rolling(window=5, min_periods=1).mean().values
y_ekf = pd.Series(df['y']).rolling(window=5, min_periods=1).mean().values

# True path
x_true = df['x'].values
y_true = df['y'].values

# RMSE
rmse = np.sqrt(((x_ekf - x_true)**2 + (y_ekf - y_true)**2).mean())
print(f"EKF RMSE: {rmse:.2f} meters")

# Save output
pd.DataFrame({'x': x_ekf, 'y': y_ekf}).to_csv("output/ekf_predictions.csv", index=False)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x_true - x_true[0], y_true - y_true[0], label='Raw GPS', alpha=0.5)
plt.plot(x_ekf - x_ekf[0], y_ekf - y_ekf[0], '--', label='EKF Path')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.title('EKF Path vs Raw GPS')
plt.legend()
plt.axis('equal')
plt.grid()
plt.tight_layout()
plt.show()

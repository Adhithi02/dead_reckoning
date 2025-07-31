import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_prep import preprocess_data
from model_train import LSTMModel

# Load model
model = LSTMModel()
model.load_state_dict(torch.load('model/lstm_model.pt'))
model.eval()

# Load data
X, y, _, _, df, delta_scaler = preprocess_data('data/data.csv')
X_tensor = torch.tensor(X, dtype=torch.float32)

# Predict delta x/y (normalized)
with torch.no_grad():
    preds_norm = model(X_tensor).numpy()

# Denormalize predictions
preds = delta_scaler.inverse_transform(preds_norm)

# Reconstruct path
x_pred, y_pred = [], []
for i in range(len(preds)):
    x_pred.append(df['x'].iloc[i + 9] + preds[i][0])
    y_pred.append(df['y'].iloc[i + 9] + preds[i][1])

# True path
x_true = df['x'].iloc[10:].values
y_true = df['y'].iloc[10:].values

# RMSE
rmse = np.sqrt(((np.array(x_pred) - x_true)**2 + (np.array(y_pred) - y_true)**2).mean())
print(f"ML-only RMSE: {rmse:.2f} meters")

# Save output
pd.DataFrame({'x': x_pred, 'y': y_pred}).to_csv("output/ml_predictions.csv", index=False)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df['x'] - df['x'][0], df['y'] - df['y'][0], label='Raw GPS', alpha=0.5)
plt.plot(np.array(x_pred) - x_pred[0], np.array(y_pred) - y_pred[0], '--', label='ML Predicted Path')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.title('ML-only Path vs Raw GPS')
plt.legend()
plt.axis('equal')
plt.grid()
plt.tight_layout()
plt.show()

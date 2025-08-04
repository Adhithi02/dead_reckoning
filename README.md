

# 🚗 Hybrid Vehicle Navigation Using GPS, IMU & Sensor Fusion

## 📌 Introduction

This project focuses on building a **robust vehicle tracking system** by combining:
- **GPS (Global Positioning System)** data
- **IMU (Inertial Measurement Unit)** sensor data — Accelerometer & Gyroscope
- **Sensor Fusion Techniques** like:
  - Dead Reckoning
  - Extended Kalman Filter (EKF)
  - Machine Learning (LSTM)
  - Hybrid EKF + ML Correction

---

## 🎯 Objective

The goal is to develop a hybrid navigation model that:
- Enhances **location accuracy** and **reliability** during GPS degradation
- **Fuses IMU + GPS** to predict position even when GPS fails
- **Learns correction patterns** using ML to improve EKF outputs
- Provides a **comparative evaluation** of:
  - GPS-only
  - ML-only
  - EKF-only
  - Hybrid EKF + ML

---

## 🔍 Why This Matters

Real-world environments like tunnels, cities with tall buildings, or forests can cause:
- GPS **signal loss** or **multipath errors**
- Erratic or delayed positioning

By fusing IMU data with GPS and applying **sensor fusion + machine learning**, we can:
- Predict short-term paths without GPS
- Smooth GPS noise
- Correct drift in real-time navigation systems

---

## 🧠 Goal

To accurately estimate the vehicle’s position **even when GPS signals are weak or noisy**, by combining:
- Accelerometer + Gyroscope (IMU) data
- GPS coordinates
- Sensor fusion and ML models

---

## 🗂️ Folder & File Structure Explained

```plaintext
dead reckoning/
├── data/
│   └── data.csv
│
├── model/
│   ├── lstm_model.pt
│   └── ekf_correction_model.pt
│
├── output/
│   ├── ml_predictions.csv
│   ├── ekf_predictions.csv
│   └── hybrid_predictions.csv
│
├── src/
│   ├── data_prep.py
│   ├── model_train.py
│   ├── model_eval.py
│   ├── ekf_eval.py
│   ├── correction_train.py
│   ├── hybrid_eval.py
│   └── compare_paths.py
│
├── requirements.txt
└── README.md
````

---

## 📁 Folder Details

### `data/`

* **`data.csv`**: Input dataset containing:

  * `timestamp`, `accel_x`, `accel_y`, `gyro_x`, `gyro_y` (IMU)
  * `latitude`, `longitude` (GPS)

---

### `model/`

* Stores trained PyTorch models:

  * **`lstm_model.pt`**: ML model trained to predict GPS from IMU.
  * **`ekf_correction_model.pt`**: LSTM model trained to correct EKF output.

---

### `output/`

* Prediction results are saved here after evaluation:

  * **`ml_predictions.csv`**: Path predicted using ML only.
  * **`ekf_predictions.csv`**: Path estimated via EKF only.
  * **`hybrid_predictions.csv`**: EKF path corrected by ML model.

---

## 🧾 Script Descriptions (Inside `src/`)

### ✅ `data_prep.py`

* Preprocesses `data.csv`:

  * Converts lat/lon to X/Y (meters)
  * Normalizes IMU values
  * Generates sequences for training

---

### ✅ `model_train.py`

* Trains an **LSTM model** using IMU data to predict relative movement (∆x, ∆y).
* Saves model to `model/lstm_model.pt`.

---

### ✅ `model_eval.py`

* Loads the LSTM model and:

  * Predicts path from IMU
  * Reconstructs trajectory from predicted ∆x, ∆y
  * Calculates RMSE vs. actual GPS
  * Plots: Raw GPS vs ML Path

---

### ✅ `ekf_eval.py`

* Implements a **simplified Extended Kalman Filter**:

  * Fuses IMU + GPS
  * Outputs filtered path
  * Calculates EKF RMSE

---

### ✅ `correction_train.py`

* Uses residuals between EKF and actual GPS to train a **correction model** (LSTM).
* Model learns to correct EKF errors.
* Output: `ekf_correction_model.pt`

---

### ✅ `hybrid_eval.py`

* Applies **correction model** on EKF path.
* Final path = EKF + ML correction
* Calculates hybrid RMSE
* Saves prediction to `hybrid_predictions.csv`
* Plots: GPS vs Hybrid

---

### ✅ `compare_paths.py`

* Loads all 3 paths: GPS, ML-only, EKF, Hybrid
* Plots them together for visual comparison
* Prints RMSE of:

  * Raw GPS
  * ML-only
  * EKF-only
  * EKF + ML

---

## 📦 Dependencies

Install all packages with:

```bash
pip install -r requirements.txt
```

Typical dependencies include:

* `numpy`
* `pandas`
* `matplotlib`
* `torch`
* `scikit-learn`

---

## ▶️ Running the Project (in order)

```bash
# 1. Preprocess the data
python src/data_prep.py

# 2. Train ML-only model
python src/model_train.py

# 3. Evaluate ML-only model
python src/model_eval.py

# 4. Evaluate EKF-only path
python src/ekf_eval.py

# 5. Train correction model for EKF
python src/correction_train.py

# 6. Evaluate hybrid EKF + ML path
python src/hybrid_eval.py

# 7. Compare all paths together
python src/compare_paths.py
```

---

## 📊 Sample RMSE Results

| Method          | RMSE (meters) |
| --------------- | ------------- |
| ML-only         | 4.39          |
| EKF-only        | 5.31          |
| Hybrid (EKF+ML) | 6.69          |

Note: Results may vary depending on dataset, model weights, and randomness.

---

## 🔬 Techniques Used

* **Dead Reckoning**: Predicts next position using IMU delta + velocity.
* **Kalman Filter**: Reduces noise and fuses GPS + IMU.
* **LSTM (RNN)**: Learns temporal dependencies from IMU data.
* **Hybrid Model**: ML corrects EKF trajectory using residual learning.

---

## 💡 Use Cases

* Vehicle tracking in **GPS-denied environments**
* Navigation in **urban canyons** or **underground**
* Integrating with **telematics devices**

---

## 🧠 Future Scope

* Real-time live sensor integration
* 3D path estimation (x, y, z)
* Integration with map APIs
* Deployment on embedded systems (Raspberry Pi, Jetson Nano)

---



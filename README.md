# Hybrid Vehicle Navigation Using GPS, IMU & Sensor Fusion

##  Introduction

This project focuses on building a **robust vehicle tracking system** by combining:
- **GPS (Global Positioning System)** data
- **IMU (Inertial Measurement Unit)** sensor data — Accelerometer & Gyroscope
- **Sensor Fusion Techniques** including:
  - Dead Reckoning
  - Extended Kalman Filter (EKF)
  - Machine Learning (LSTM)
  - Hybrid EKF + ML Correction

The project provides two distinct implementations for different use cases and complexity levels.

---

##  Objective

The goal is to develop navigation models that:
- Enhance **location accuracy** and **reliability** during GPS degradation
- **Fuse IMU + GPS** to predict position even when GPS fails
- **Learn correction patterns** using ML to improve filter outputs
- Provide a **comparative evaluation** of different approaches

---

##  Why This Matters

Real-world environments like tunnels, cities with tall buildings, or forests can cause:
- GPS **signal loss** or **multipath errors**
- Erratic or delayed positioning

By fusing IMU data with GPS and applying **sensor fusion techniques**, we can:
- Predict short-term paths without GPS
- Smooth GPS noise
- Correct drift in real-time navigation systems

---

##  Project Structure

This repository contains two main implementations:

###  **Main Branch** - Advanced Hybrid System
```plaintext
hybrid-vehicle-navigation/
├── data/
│   └── data.csv
├── model/
│   ├── lstm_model.pt
│   └── ekf_correction_model.pt
├── output/
│   ├── ml_predictions.csv
│   ├── ekf_predictions.csv
│   └── hybrid_predictions.csv
├── src/
│   ├── data_prep.py
│   ├── model_train.py
│   ├── model_eval.py
│   ├── ekf_eval.py
│   ├── correction_train.py
│   ├── hybrid_eval.py
│   └── compare_paths.py
├── requirements.txt
└── README.md
```

###  **DR-Only Branch** - Kalman Filter Focus
```plaintext
dr-only/
├── data/
│   ├── processed/
│   │   ├── cleaned_dataset.csv
│   │   ├── cleaned_dataset_local.csv
│   │   ├── fused_estimates.csv
│   │   ├── origin.txt
│   │   └── *.png (visualizations)
│   └── raw/
│       ├── data.csv
│       └── Serial_ACC_GNSS_data.txt
├── scripts/
│   ├── 01_extract_clean_dataset.py
│   ├── 02_convert_to_local.py
│   ├── 03_kalman_filter.py
│   └── 04_compare_and_plot.py
├── venv/
├── requirements.txt
└── README.md
```

---

##  Implementation Details

### **Main Branch Features**
- **Advanced ML Models**: LSTM networks for trajectory prediction
- **Hybrid Fusion**: Combines EKF with ML correction models
- **Multi-Method Comparison**: GPS-only vs ML-only vs EKF vs Hybrid
- **End-to-End Pipeline**: From data preprocessing to performance evaluation

### **DR-Only Branch Features**
- **Real Data Processing**: Handles raw sensor logs with regex parsing
- **Coordinate Conversion**: GPS to local metric coordinates
- **Kalman Filter Implementation**: Classical sensor fusion approach
- **Comprehensive Analysis**: Error metrics and visualization tools
- **Modular Pipeline**: Step-by-step processing with clear separation

---

## Dependencies

Both implementations require:
```bash
pip install -r requirements.txt
```

**Common dependencies:**
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `matplotlib` - Visualization
- `scikit-learn` - ML utilities

**Main branch additional:**
- `torch` - Deep learning framework

---

##  Quick Start

### **Main Branch Usage**
```bash
# Complete pipeline for hybrid navigation
python src/data_prep.py
python src/model_train.py
python src/model_eval.py
python src/ekf_eval.py
python src/correction_train.py
python src/hybrid_eval.py
python src/compare_paths.py
```

### **DR-Only Branch Usage**
```bash
# Step-by-step Kalman filter approach
python scripts/01_extract_clean_dataset.py
python scripts/02_convert_to_local.py
python scripts/03_kalman_filter.py
python scripts/04_compare_and_plot.py
```

---

##  Performance Comparison

### **Main Branch Results**
| Method          | RMSE (meters) |
| --------------- | ------------- |
| ML-only         | 4.39          |
| EKF-only        | 5.31          |
| Hybrid (EKF+ML) | 6.69          |

### **DR-Only Branch Results**
- **RMSE**: Typically 1-3 meters depending on GPS quality
- **Real-time capable**: Optimized for embedded systems
- **Configurable**: Easy parameter tuning for different scenarios

---

##  Techniques Implemented

### **Sensor Fusion Methods**
- **Dead Reckoning**: IMU-based position prediction
- **Kalman Filtering**: Optimal state estimation with uncertainty
- **LSTM Networks**: Deep learning for temporal pattern recognition
- **Hybrid Correction**: ML-enhanced classical filtering

### **Data Processing**
- **Coordinate Transformation**: GPS to local metric system
- **Sensor Calibration**: IMU bias correction and noise filtering
- **Time Synchronization**: Alignment of multi-rate sensor data
- **Interpolation**: Gap filling for missing GPS data

---

##  Use Cases

- **Autonomous Vehicles**: Navigation in GPS-denied environments
- **Fleet Management**: Enhanced tracking in urban canyons
- **Mobile Robotics**: Indoor/outdoor seamless navigation
- **IoT Tracking**: Low-power, high-accuracy positioning
- **Emergency Services**: Reliable location in challenging conditions

---

##  Getting Started

1. **Choose Your Branch**:
   - `main`: For comprehensive ML-based hybrid navigation
   - `dr-only`: For focused Kalman filter implementation

2. **Prepare Your Data**:
   - Main: CSV with IMU and GPS columns
   - DR-only: Raw sensor logs or pre-processed CSV

3. **Run the Pipeline**:
   - Follow the respective usage instructions above

4. **Analyze Results**:
   - Check output files and generated visualizations

---

##  Future Enhancements

### **Technical Roadmap**
- **Real-time Integration**: Live sensor data processing
- **3D Navigation**: Full 6-DOF state estimation
- **Map Integration**: Constraint-based filtering with OSM
- **Multi-Vehicle**: Cooperative localization algorithms

### **Deployment Options**
- **Edge Computing**: Raspberry Pi, Jetson Nano deployment
- **Cloud Integration**: Scalable fleet management
- **Mobile Apps**: Real-time navigation assistance
- **Embedded Systems**: Automotive-grade implementations

---

##  Documentation

Each branch contains detailed documentation:
- **Code Comments**: Inline explanations of algorithms
- **Configuration**: Tunable parameters and their effects
- **Error Analysis**: Comprehensive performance metrics
- **Visualization**: Clear plots for result interpretation

---

*This project demonstrates the power of combining classical sensor fusion with modern machine learning for robust vehicle navigation systems.*

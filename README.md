# 💤 Predicting Sleep Quality from Smartphone Sensor Data

![Focus](https://img.shields.io/badge/Focus-Neuroscience%20%26%20Machine%20Learning-blueviolet)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project aims to predict **subjective sleep quality** (rated 1–10) using passive smartphone sensor data and self-reported sleep questionnaires. 

The model predicts:
* **"Rate your overall sleep last night"** (1–10 scale)

Sleep quality was selected because it represents a holistic perception of restfulness, integrating physiological, behavioral, and psychological components of sleep. This is a **regression task**, predicting continuous values between 1 and 10.

### 🎯 Objective
To build a **causal, personalized regression model** for each student that predicts sleep quality using:
* Raw smartphone sensor data
* Previous night sleep duration (hours only)
* **Strict Causality:** No future information leakage

---

## 📊 Data Sources
Data was collected using the **BHQ mobile sensing platform** and includes:

* **📱 Movement Sensors:** Accelerometer (x, y, z)
* **🌤 Environment Sensors:** Light, Location
* **📡 Communication Sensors:** Screen usage, Wi-Fi scans, Calls

### 🧠 Methodology

1. **Data Cleaning (`clean.py`):** Numeric coercion for sensor values and removal of invalid/NaN entries.
2. **Time Window Segmentation (`get_sensor_window.py`):** For each label, a personalized window is computed:
    * **Full Window:** (Previous wake-up $\rightarrow$ current wake-up)
    * **Day / Pre-sleep (3h before) / Sleep**
3. **Feature Extraction (`feature_extraction_v2.py`):**
    * **Movement:** Mean/Std magnitude, restlessness ratio, entropy.
    * **Light:** Circadian misalignment, evening/morning exposure.
    * **Screen/Calls:** Night usage, last usage time, inter-event variability.
    * **Location/Wi-Fi:** Night home stability, unique access points.
4. **Train & Test Strategy:** Personalized prediction per student with careful separation to prevent "window flipping."
5. **Feature Selection:** Wrapper method with exhaustive search to justify the final feature count experimentally.

---

## 🗂 Project Structure

```text
├── main_1.py                # Main execution script
├── clean.py                 # Data cleaning logic
├── explore_sensors.py       # Data exploration
├── get_sensor_window.py     # Time segmentation
├── feature_extraction_v2.py # Feature engineering
├── data/                    # Raw data (Session A, B, C)
└── features_session_*.csv   # Extracted feature matrices
```

### 🚀 How to Run

1.  **Prepare Data:**
    Place all data inside a folder named `data/` with the following structure:
    ```text
    data/
      ├── Session A/
      ├── Session B/
      └── Session C/
    ```

2.  **Install Requirements:**
    ```bash
    pip install pandas numpy scipy openpyxl
    ```

3.  **Run the Pipeline:**
    ```bash
    python main_1.py
    ```

---

## 📈 Output
For each session, the script saves a `features_session_[ID].csv` containing:
* **UID**
* **Label date**
* **Sleep score**
* **Extracted & normalized features**

---

## ⚙️ Requirements
* `pandas`
* `numpy`
* `scipy`
* `openpyxl`

---

## 👩‍💻 Author
**Shir Frank** & **Yuval Berkowitch**

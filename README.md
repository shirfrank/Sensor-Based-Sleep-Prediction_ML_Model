# 💤 Predicting Sleep Quality from Smartphone Sensor Data

![Neuroscience](https://img.shields.io/badge/Focus-Neuroscience%20%26%20Machine%20Learning-blueviolet)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Personalized ML](https://img.shields.io/badge/Model-Personalized%20Regression-orange)

## 📌 Project Overview
This project implements a personalized machine learning pipeline to predict **subjective sleep quality** (1–10) using passive smartphone sensor data. Unlike general models, this system builds **individualized models** for each user, recognizing that behavioral impact on sleep varies significantly between people.

### 🎯 Objective
To predict a user's perceived sleep quality based on their previous day's behavioral and environmental context without **future information leakage** (strict causality).

---

## 🧠 Methodology & Pipeline

### 1. Data Cleaning & Signal Processing (`clean.py`)
* **Sensor Fusion:** Processes Accelerometer, Light, Screen, Calls, Wi-Fi, and Location data.
* **Imputation:** Implements linear interpolation and forward/backward filling for continuous sensors (Light, Accel, Wi-Fi) to handle sampling gaps.
* **Event Handling:** Treats Screen and Call logs as discrete event-driven signals.

### 2. Causal Time-Windowing (`get_sensor_window.py`)
A custom algorithm estimates the sleep/wake boundaries for each night to ensure no data from "the future" (post-wake-up) is used for prediction.
* **Full Window:** From previous day's wake-up to current wake-up.
* **Sub-windows:** Segments data into `Day`, `Pre-sleep` (3 hours before), and `Sleep` periods to capture circadian behavioral shifts.

### 3. Feature Engineering (`feature_extraction_v2.py`)
Extracted over 50+ features, including:
* **Physical Activity:** Movement ratios and longest still periods via Accelerometer magnitude.
* **Circadian Rhythm:** Circular mean centers for light and screen usage to calculate **Circadian Misalignment**.
* **Digital Hygiene:** Screen-on gaps before sleep and "Evening Stimulation" indices.
* **Environmental Stability:** Wi-Fi access point entropy and location-based mobility ratios.

### 4. Personalized Selection Pipeline
To prevent overfitting on small individual datasets, the project employs a multi-stage selection process:
1.  **Vetting (`vetting.py`):** Uses Spearman correlation to remove redundant features ($|r| > 0.8$) and ranks features by label relevance.
2.  **Exhaustive Selection (`feature_selection.py`):** A wrapper method using `TimeSeriesSplit` and Ridge Regression to find the optimal combination of features.
3.  **Merging (`merge_features.py`):** Identifies the most robust features across multiple temporal splits per user.

---

## 🗂 Project Structure

```text
├── main_1.py                # Pipeline orchestrator
├── clean.py                 # Signal cleaning & interpolation
├── get_sensor_window.py     # Causal time segmentation logic
├── feature_extraction_v2.py # Core behavioral feature engineering
├── train_test.py            # Temporal split generation
├── vetting.py               # Spearman-based redundancy removal
├── feature_selection.py     # Wrapper-based exhaustive search
├── build_final_user_files.py # Final CSV consolidation
├── data/                    # Raw session data (Excel/CSV)
└── final_user_csvs/         # Ready-to-model personalized datasets
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
    pip install pandas numpy scipy openpyxl scikit-learn tqdm
    ```

3.  **Run the Pipeline:**
    ```bash
    python main_1.py
    ```

---

## 📈 Output
The pipeline generates Personalized Feature Matrices in final_user_csvs/. Each file contains:

* Temporal Splits: Training and testing sets divided by time (no shuffling) to respect causality.
* Vetted Features: Only the most statistically significant, non-redundant features for that specific user.
* Evaluation Reports: A comparison of estimated sleep/wake times vs. self-reported user labels.

---

## ⚙️ Requirements
* `pandas`
* `numpy`
* `scipy`
* `openpyxl`

---

## 👩‍💻 Author
**Shir Frank** & **Yuval Berkowitch**

#  Flood Detection Using Sentinel-1 SAR Imagery

### Machine Learning & Deep Learning for Automated Flood Mapping

![Logo](docs/img/hearder.png)

---

##  Overview

Floods are among the most devastating natural disasters, representing **44% of global disaster impacts**. With the rise of satellite-based monitoring and AI, flood detection can now be automated—offering faster, more reliable insights during emergencies.

This project builds a **supervised machine learning pipeline** using **Sentinel-1 SAR imagery** to classify tiles as **Flooded (1)** or **Not Flooded (0)**.
We implement and compare **three models**:

*  **Convolutional Neural Network (CNN)**
*  **Random Forest (RF)**
*  **Support Vector Machine – RBF (SVM-RBF)**

The goal is to evaluate which approach provides the most reliable and generalizable performance for flood detection, with relevance to real-world decision-making and alignment with the Kingdom of Saudi Arabia’s **Vision 2030** for disaster preparedness and sustainable urban planning.

---

## 🛰️ Task Definition

**Input:** Sentinel-1 SAR features (VV, VH) + engineered bands
**Output:** Binary classification → `1 = Flooded`, `0 = Non-Flooded`
**Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC

A full pipeline—from preprocessing to model evaluation—is illustrated in the workflow diagram (Figure 1 in the report).

---

##  Dataset

We use the **SEN12-FLOOD dataset**, a large-scale benchmark combining:

* Sentinel-1 SAR (VV, VH)
* Flood masks in GeoJSON
* 3,331 tiles (525×614 px)
* Global coverage via Radiant Earth

### Why SEN12-FLOOD?

* Designed **specifically** for flood mapping
* Reliable under cloud/weather conditions
* Provides rich metadata (date, orbit, tile ID)
* Open, reproducible, widely used in academic research

---

##  Methods

Three classification approaches were implemented and compared:

### 1️ Convolutional Neural Network (CNN)

* End-to-end tile classification
* Learns spatial flood textures directly
* Uses 256×256×6 engineered feature tiles
* Best-performing model in this project

### 2️ Random Forest (RF)

* Classical non-deep learning baseline
* Operates on flattened feature vectors
* Strong recall, moderate precision
* Robust to SAR noise but prone to false positives

### 3️ SVM-RBF

* High-dimensional nonlinear classifier
* Requires StandardScaler + RBF feature mapping
* Weak performance on SAR flood features
* High FN rates → unsuitable for deployment

###  Feature Engineering

We generated **six SAR-based features**:
`VV_dB, VH_dB, VV+VH, VV−VH, VV/VH, VV×VH`

These features enhance flood vs. non-flood separability.

---

##  Experiment Pipeline

* Standardized preprocessing for all models
* Balanced training set; unbalanced validation/test for realism
* GPU-accelerated CNN training (Google Colab T4)
* Hyperparameter tuning for all models
* Unified evaluation protocol (Confusion Matrix, ROC, PR curves)

---

##  Results

| Metric    | CNN      | Random Forest | SVM-RBF |
| --------- | -------- | ------------- | ------- |
| Accuracy  | **0.82** | 0.73          | 0.60    |
| Precision | **0.84** | 0.54          | 0.32    |
| Recall    | **0.89** | 0.86          | 0.29    |
| F1-score  | **0.76** | 0.66          | 0.30    |
| ROC-AUC   | **0.91** | 0.82          | 0.51    |

###  Summary

* **CNN is the strongest model**, with excellent recall of flooded tiles
* RF performs reasonably but suffers from high false positives
* SVM-RBF fails to generalize, biased toward the Non-Flood class



## 📂 Project Structure

```
project/
│
├── Notebook/
│   ├── Flood_Detection_Notebook.ipynb       # Training, evaluation, metrics
│   └── Model_Live_Testing.ipynb             # Streamlit live testing
│
├── docs/
│   └── img/
│       └── header.png
│
├── mini_data/
│   ├── sen12floods_s1_labels_0020_....npz
│   ├── sen12floods_s1_labels_0140_....npz
│   ├── sen12floods_s1_labels_46_....npz
│   ├── sen12floods_s1_labels_54_....npz
│   └── sen12floods_s1_labels_58_....npz
│
├── models/
│   ├── best_model.keras                     # CNN model
│   ├── best_rf_model.pkl                    # Random Forest model
│   └── (SVM model stored externally due to size)
│
├── AUTHORS.md
└── README.md
```

> **Note:**
> The SVM model is large (≈3 GB) and is stored externally.
> You can download it here:
> **SVM-RBF Model Download:** *[SVM-RBF Model](https://drive.google.com/file/d/1sLP8f247J6LxzfvuPFhUHSI76yOvrMpB/view?usp=drive_link)*




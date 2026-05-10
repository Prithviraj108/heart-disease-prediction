# heart-disease-prediction
Machine learning project for predicting the likelihood of heart disease using patient health data and classification algorithms.

# ❤️ Heart Disease Prediction

A machine learning project that predicts the presence of heart disease in patients using clinical features. Built with Python, trained on the [DrivenData — Machine Learning with a Heart](https://www.drivendata.org/competitions/54/) dataset hosted on Kaggle.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models Used](#models-used)
- [Results](#results)
- [Setup & Usage](#setup--usage)
- [Requirements](#requirements)
- [Key Findings](#key-findings)

---

## Overview

Heart disease is the number one cause of death worldwide. This project builds and evaluates multiple classification models to predict whether a patient has heart disease based on 13 clinical measurements such as resting blood pressure, serum cholesterol, ECG results, and exercise-induced indicators.

**Key metric: Recall** — in a medical context, missing a sick patient (false negative) is far more costly than a false alarm. All models are evaluated primarily on recall score alongside accuracy and ROC-AUC.

---

## Dataset

**Source:** [Kaggle — Heart Disease Prediction](https://www.kaggle.com/datasets/prithvirajshukla/heart-disease-prediction)  
**Origin:** DrivenData Competition #54 — *Machine Learning with a Heart* (UCI Cleveland Clinic Foundation)

| File | Rows | Columns | Description |
|---|---|---|---|
| `values.csv` | 180 | 14 | Patient features (patient_id + 13 clinical features) |
| `labels.csv` | 180 | 2 | Target labels (patient_id + heart_disease_present) |

**Target variable:** `heart_disease_present` — `0` = no disease, `1` = disease present

### Features

| Column | Description |
|---|---|
| `age` | Patient age in years |
| `sex` | 0 = female, 1 = male |
| `chest_pain_type` | 1 = typical angina, 2 = atypical, 3 = non-anginal, 4 = asymptomatic |
| `resting_blood_pressure` | Resting BP on admission (mm Hg) |
| `serum_cholesterol_mg_per_dl` | Serum cholesterol (mg/dL) |
| `fasting_blood_sugar_gt_120_mg_per_dl` | Fasting blood sugar > 120 mg/dL (1 = true) |
| `resting_ekg_results` | 0 = normal, 1 = ST-T abnormality, 2 = LV hypertrophy |
| `max_heart_rate_achieved` | Maximum heart rate during stress test (bpm) |
| `exercise_induced_angina` | 0 = no, 1 = yes |
| `oldpeak_eq_st_depression` | ST depression induced by exercise |
| `slope_of_peak_exercise_st_segment` | 1 = upsloping, 2 = flat, 3 = downsloping |
| `num_major_vessels` | Major vessels coloured by fluoroscopy (0–3) |
| `thal` | Thallium stress test: normal, fixed_defect, reversible_defect |

---

## Project Structure

```
heart-disease-prediction/
│
├── Heart_Disease_Prediction.ipynb   # Main notebook (Google Colab)
├── README.md                        # This file
└── PRCP-1016-HeartDieseasePred.zip  # Raw dataset files
```

---

## Models Used

| Model | Description |
|---|---|
| Logistic Regression | Baseline linear classifier; strong interpretability |
| Logistic Regression + Bagging | Ensemble of 45 logistic regression models to improve recall |
| K-Nearest Neighbours (KNN) | Non-parametric classifier; optimal k selected via error rate plot |
| Random Forest | 200-tree ensemble; best overall performance |

**Preprocessing steps:**
- Safe label merge using `patient_id` (not positional concat)
- Manual encoding of `thal` (normal=2, reversible_defect=1, fixed_defect=0)
- IQR-based outlier detection and median imputation for `resting_bp`, `serum_cholesterol`, `oldpeak`
- StandardScaler applied **after** train/test split to prevent data leakage
- 5-fold cross-validation for reliable performance estimates on the small dataset (180 patients)

---

## Results

| Model | Test Accuracy | Recall | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~83% | ~87% | — |
| Logistic + Bagging | ~84% | ~89% | — |
| KNN (k=3) | ~80% | ~78% | — |
| **Random Forest** | **~87%** | **~88%** | **~0.93** |

> ✅ **Selected model: Random Forest** — best balance of accuracy, recall, and AUC.  
> Logistic Regression is retained as an interpretable alternative.

---

## Setup & Usage

### Run on Google Colab (recommended)

1. Open the notebook in Colab:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Prithviraj108/heart-disease-prediction/blob/main/Heart_Disease_Prediction.ipynb)

2. Get your Kaggle API key:
   - Go to [kaggle.com](https://kaggle.com) → Profile → Settings → API → **Create New Token**
   - This downloads `kaggle.json` to your computer

3. Run the first cell — it will prompt you to upload `kaggle.json`

4. The dataset downloads automatically from Kaggle — no manual file handling needed

5. Run all remaining cells: **Runtime → Run All**

> ⚠️ `kaggle.json` is personal — never commit it to GitHub. It is listed in `.gitignore`.

### Run locally (Anaconda / Jupyter)

```bash
# Clone the repo
git clone https://github.com/Prithviraj108/heart-disease-prediction.git
cd heart-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Set up Kaggle credentials
mkdir -p ~/.kaggle
cp /path/to/your/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Launch Jupyter
jupyter notebook Heart_Disease_Prediction.ipynb
```

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
sweetviz
kagglehub
jupyter
```

Install all at once:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy sweetviz kagglehub jupyter
```

---

## Key Findings

- **Chest pain type** and **thal (thallium test result)** are the strongest predictors of heart disease
- Patients with **asymptomatic chest pain** have the highest risk — counterintuitively, no pain does not mean no disease
- **Reversible defect** in the thal test significantly increases heart disease probability
- **Max heart rate below 140 bpm** and **oldpeak ST depression above 1** are strong risk indicators
- **Male patients** show higher incidence of heart disease in this dataset
- Random Forest achieves ~93% ROC-AUC — strong discriminative ability for a dataset of only 180 patients

---

## License

This project is for educational purposes. The dataset originates from the UCI Machine Learning Repository (Cleveland Clinic Foundation) and is used under CC0 (Public Domain) via DrivenData.

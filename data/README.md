# 📊 Data Directory

This directory contains all datasets used for training, validating, and evaluating the **Soil Health Assessment and Crop Recommendation System**.

The data is structured, cleaned, and curated to support reproducible machine learning workflows.

---

## 📁 Dataset Overview

### 1. `soil_health.csv`

**Purpose**  
Dataset used to train and evaluate the **soil health classification model**.

**Size**  
- **1,288 samples**
- **12 soil parameters**

**Features**
- Nutrients: Nitrogen (N), Phosphorus (P), Potassium (K), Sulfur (S)
- Micronutrients: Zinc (Zn), Iron (Fe), Copper (Cu), Manganese (Mn), Boron (B)
- Soil properties: pH, Electrical Conductivity (EC), Organic Carbon (OC)

**Target Variable**
- `health_class`
  - 0 → Poor  
  - 1 → Fair  
  - 2 → Good  

**Class Distribution**
- Poor: 401 samples (31.1%)
- Fair: 440 samples (34.2%)
- Good: 447 samples (34.7%)

**Source**  
Aggregated soil testing data from agricultural research stations in Tamil Nadu.

---

### 2. `Crop_recommendation.csv`

**Purpose**  
Dataset used to train the **crop recommendation model**.

**Size**
- **2,200 samples**
- **7 features**
- **22 crop classes**

**Features**
- Soil nutrients: N, P, K (kg/ha)
- Environmental factors: Temperature (°C), Humidity (%), Rainfall (mm)
- Soil pH

**Target Variable**
- `label`: Crop name

**Dataset Characteristics**
- Balanced dataset (100 samples per crop)
- Covers major food, fruit, and cash crops

**Source**  
Standard agricultural dataset commonly used for crop prediction tasks.

---

## 🧹 Data Preprocessing Summary

### Soil Health Dataset
- Feature scaling using **StandardScaler**
- Feature engineering (nutrient ratios, squared terms, binning)
- Stratified train-test split (80/20)
- Class imbalance handled using class weights

### Crop Recommendation Dataset
- Label encoding for crop names
- Train-test split (80/20)
- No missing values detected

---

## 📈 Data Quality & Validation

- No missing values in either dataset
- Outliers addressed during preprocessing
- Cross-validation applied during model training
- Consistent feature naming and data types enforced

---

## 📝 Usage

```python
import pandas as pd

soil_df = pd.read_csv('data/soil_health.csv')
crop_df = pd.read_csv('data/Crop_recommendation.csv')

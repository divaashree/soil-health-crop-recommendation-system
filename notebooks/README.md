## 📁 Notebook Overview

### 1. `soil_health_analysis.ipynb`

**Purpose**  
End-to-end development of a **soil health classification model** that predicts soil quality based on physicochemical soil parameters.

**Workflow & Contents**

- **Cell 1–2**: Data loading and initial inspection  
- **Cell 3**: Target variable analysis and class distribution  
- **Cell 4**: Feature exploration and visualization  
- **Cell 5**: Data preprocessing (scaling, encoding, train-test split)  
- **Cell 6**: Feature engineering (polynomial terms, ratios, binning)  
- **Cell 7**: Base model training (Random Forest, Gradient Boosting)  
- **Cell 8**: Ensemble learning using a Stacking Classifier  
- **Cell 9**: Model evaluation (accuracy, confusion matrix, reports)  
- **Cell 10**: Feature importance analysis  
- **Cell 11**: Model testing with diverse unseen inputs  
- **Cell 12**: Model persistence (saving trained artifacts)  
- **Cell 13**: Model verification and consistency checks  

**Key Results**

- **Final Accuracy**: **94.57%**  
- **Final Model**: Stacking Classifier with three base estimators  
- **Target Classes**:
  - Poor (0)
  - Fair (1)
  - Good (2)

**Generated Output Files** (saved to `models/`)

- `soil_health_stacking_model.pkl`
- `soil_label_encoder.pkl`
- `soil_feature_names.pkl`
- `soil_feature_scaler.pkl`
- `soil_stacking_model_info.json`

---

### 2. `crop_recommendation.ipynb`

**Purpose**  
Development of a **crop recommendation model** that suggests the most suitable crop based on soil nutrients and environmental conditions.

**Workflow & Contents**

- Dataset loading and exploratory analysis  
- Feature–target separation  
- Model training and comparison:
  - Naive Bayes
  - Random Forest
- Performance evaluation and model selection  
- Saving the final model and metadata  

**Key Results**

- **Naive Bayes Accuracy**: **99.09%**  
- **Random Forest Accuracy**: **99.55%**  
- **Selected Model**: Random Forest Classifier  
- **Number of Crop Classes**: 22  

**Generated Output Files** (saved to `models/`)

- `crop_model.pkl`
- `crop_label_encoder.pkl`
- `crop_feature_names.pkl`
- `crop_model_info.json`

---

## 🚀 Running the Notebooks

### Prerequisites

Install the required dependencies before execution:

```bash
pip install jupyter notebook pandas numpy matplotlib seaborn scikit-learn

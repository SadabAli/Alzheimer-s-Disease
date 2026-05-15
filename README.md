# Alzheimer’s Disease Prediction using Explainable AI (SHAP)

## Overview

This project focuses on predicting Alzheimer’s Disease using Machine Learning and Explainable AI techniques. Multiple ML models were implemented and compared using feature selection, hyperparameter tuning, ROC analysis, and SHAP-based explainability.

The project aims to improve prediction performance while making the model interpretable for healthcare-related decision-making.

---

# Features

* Alzheimer’s Disease Prediction
* Feature Selection Techniques
* Hyperparameter Tuning
* Explainable AI using SHAP
* ROC Curve Analysis
* Confusion Matrix Visualization
* Comparative Machine Learning Analysis
* Model Performance Evaluation

---

# Machine Learning Models Used

* Random Forest Classifier
* Gradient Boosting Classifier
* XGBoost Classifier
* CatBoost Classifier
* Support Vector Machine (SVM)

---

# Feature Selection Techniques

The following feature selection methods were used:

1. Feature Importance from Random Forest
2. Recursive Feature Elimination (RFE)
3. SelectFromModel
4. XGBoost Feature Importance

---

# Hyperparameter Tuning

Hyperparameter tuning was performed using:

* GridSearchCV
* RandomizedSearchCV

---

# Explainable AI (SHAP)

SHAP (SHapley Additive Explanations) was used to interpret model predictions and understand feature contributions.

## SHAP Visualizations

* SHAP Summary Plot
* SHAP Bar Plot
* SHAP Force Plot
* SHAP Waterfall Plot
* SHAP Dependence Plot

---

# Dataset

The dataset contains clinical, behavioral, cognitive, and health-related features associated with Alzheimer’s Disease prediction.

Example features:

* FunctionalAssessment
* ADL
* MMSE
* MemoryComplaints
* BehavioralProblems
* Age
* BMI
* Cholesterol Levels
* PhysicalActivity
* SleepQuality

---

# Project Workflow

1. Data Preprocessing
2. Feature Selection
3. Data Scaling
4. Model Training
5. Hyperparameter Tuning
6. Model Evaluation
7. SHAP Explainability
8. ROC Curve Analysis

---

# Evaluation Metrics

The following metrics were used:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

---

# Technologies Used

* Python
* Scikit-learn
* XGBoost
* CatBoost
* SHAP
* Pandas
* NumPy
* Matplotlib
* Seaborn

---

# SHAP Insights

SHAP analysis revealed that the most influential features for Alzheimer’s prediction were:

* FunctionalAssessment
* ADL
* MemoryComplaints
* MMSE
* BehavioralProblems

These features had the highest contribution toward model predictions.

---

# Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost shap
```

---

# Run the Project

```bash
python app.py
```

or run the Jupyter Notebook:

```bash
jupyter notebook
```

---

# Future Improvements

* Deep Learning Integration
* Deployment using Streamlit/Flask
* Real-time Clinical Prediction System
* Advanced Explainable AI Techniques
* Larger Medical Dataset Integration

---

# Author

Sadab Ali

---

# Research Focus

This project combines Machine Learning and Explainable AI for interpretable healthcare analytics and Alzheimer’s Disease prediction.

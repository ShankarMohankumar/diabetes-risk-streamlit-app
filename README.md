# Elevated Diabetes Risk Prediction App

Machine learning application developed as part of the DATA 690 Capstone Project to predict elevated diabetes risk using health, demographic, and behavioral indicators.

## Project Overview

Diabetes is a major public health concern that can lead to serious long-term complications if not identified and managed early. This project uses machine learning models trained on health survey data to predict whether an individual is at elevated risk for diabetes.

The application allows users to enter health-related information and receive predictions from multiple machine learning models along with SHAP-based explanations to improve transparency and interpretability.

## Research Question

Can demographic, behavioral, and health-related indicators be used to accurately predict elevated diabetes risk using machine learning techniques?

## Dataset

**Dataset:** Diabetes Health Indicators Dataset

**Source:** Kaggle

**Original Source:** Behavioral Risk Factor Surveillance System (BRFSS) 2015

Dataset Characteristics:

- 253,680 records
- 21 predictor variables
- Binary classification target
- Elevated Risk = Prediabetes + Diabetes
- Lower Risk = No Diabetes

The dataset contains demographic, lifestyle, and health-related indicators including:

- Age
- Sex
- BMI
- General Health
- Physical Health
- Mental Health
- Income
- Education
- Physical Activity
- Smoking Status
- High Blood Pressure
- High Cholesterol
- And additional health indicators

## Machine Learning Models

The application includes predictions from four machine learning models:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. Artificial Neural Network (ANN)

## Model Evaluation

The primary evaluation metric used for model comparison was:

- Balanced Accuracy

Additional evaluation metrics included:

- Recall
- Precision
- F1 Score
- ROC-AUC

Balanced Accuracy was selected because it provides a more appropriate evaluation for imbalanced healthcare datasets by accounting for both sensitivity and specificity.

## Explainability

The application includes SHAP (SHapley Additive exPlanations) to improve model interpretability.

SHAP explanations help identify which features contributed most to a prediction and provide greater transparency into model behavior.

## Technologies Used

- Python
- Streamlit
- Scikit-Learn
- XGBoost
- TensorFlow / Keras
- SHAP
- Pandas
- NumPy
- Joblib

## Live Application

Streamlit Deployment:

https://diabetes-risk-predictor-sm.streamlit.app

## Repository Structure

```text
diabetes-risk-streamlit-app/
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── ann_model.keras
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── shap background files
│
├── app.py
├── requirements.txt
├── runtime.txt
└── README.md
```

## Running Locally

Clone the repository:

```bash
git clone https://github.com/ShankarMohankumar/diabetes-risk-streamlit-app.git
```

Navigate to the project folder:

```bash
cd diabetes-risk-streamlit-app
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the Streamlit application:

```bash
streamlit run app.py
```

## Privacy Notice

Information entered into the application is used only to generate predictions and is not stored.

## Author

**Shankar Mohankumar**

DATA 690 Capstone Project

University of Maryland Global Campus (UMGC)

2026

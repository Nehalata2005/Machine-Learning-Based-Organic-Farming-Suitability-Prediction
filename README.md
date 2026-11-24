# Soil Fertility and Organic Farming Suitability Prediction Using Machine Learning

**Project Title:** Machine Learning-Based Organic Farming Suitability Prediction  
**Student:** Nehalata Maji  
**Roll No:** BWU/BCA/24/541  
**Programme:** Bachelor of Computer Applications (BCA)  
**Department:** Department of Computational Sciences, Brainware University, Kolkata  
**Subject:** Project Based Learning (PBL) - Machine Learning (Group II)  

---

## 1. Project Overview
This project develops a binary machine learning classification system to predict whether an agricultural field is **Suitable** or **Not Suitable** for organic farming based on soil nutrient concentrations, climatic factors, and past agro-chemical practices.

In conventional agriculture, synthetic fertilizers and pesticides leave residual chemicals that deplete beneficial soil microorganisms and organic carbon. Predicting suitability helps assess whether a field can support organic agriculture immediately or requires a remediation/conversion period.

```text
Dataset (1,000 source-calibrated rows)
      |
      +--> Exclude 'Sample_ID' and 'Suitability_Score' (prevents target leakage)
      |
      +--> Stratified 80/20 Train-Test Split (800 train, 200 test)
      |
      +--> Preprocessing Pipeline (Median/Mode Imputer, StandardScaler, OneHotEncoder)
      |
      +--> 5-Fold Stratified Cross-Validation on 6 candidate models:
      |      - Logistic Regression
      |      - Decision Tree
      |      - K-Nearest Neighbors (KNN)
      |      - Random Forest
      |      - Gradient Boosting (Selected Best Model)
      |      - Support Vector Machine (SVM)
      |
      +--> Held-out test set evaluation (Accuracy, Precision, Recall, Specificity, F1, ROC-AUC)
      |
      +--> Feature importance analysis (Permutation importance)
      |
      +--> Streamlit interactive web interface
```

---

## 2. Dataset Information and Provenance
- **Dataset File:** `data/organic_farming_suitability_1000_source_calibrated.csv` (1,000 instances, 24 attributes).
- **Provenance Statement:** This dataset is a source-calibrated synthetic dataset prepared for PBL educational purposes. The values are calibrated against parameters from India's Soil Health Card (SHC) programme, AgriNet Bareilly records, and the Pune SHC dataset.
- **Target Variable:** `Organic_Farming_Suitability` (`Suitable` vs `Not Suitable`).
- **Target Leakage Prevention:** `Suitability_Score` is directly used to generate the target labels (with a cutoff of 55.0). Therefore, `Suitability_Score` is strictly dropped from model inputs during training to prevent target leakage.

---

## 3. Model Comparison and Evaluation Results

Evaluated on the 200 held-out test samples (115 Not Suitable, 85 Suitable):

| Model | 5-Fold CV F1 | Test Accuracy | Test Precision | Test Recall | Test Specificity | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **0.8118** | **83.00%** | **0.8000** | **0.8000** | **0.8522** | **0.8000** | **0.9003** |
| **Logistic Regression** | 0.8002 | 80.50% | 0.7875 | 0.7412 | 0.8522 | 0.7636 | 0.8899 |
| **Support Vector Machine (SVM)** | 0.7973 | 79.50% | 0.7821 | 0.7176 | 0.8522 | 0.7485 | 0.8978 |
| **Random Forest** | 0.7696 | 77.50% | 0.7564 | 0.6941 | 0.8348 | 0.7239 | 0.8762 |
| **K-Nearest Neighbors (KNN)** | 0.7105 | 77.00% | 0.7600 | 0.6706 | 0.8435 | 0.7125 | 0.8362 |
| **Decision Tree** | 0.7031 | 72.00% | 0.6593 | 0.7059 | 0.7304 | 0.6818 | 0.7863 |

**Selected Model:** Gradient Boosting achieved the highest overall test accuracy (83.00%), F1-score (0.8000), and ROC-AUC (0.9003), while maintaining high specificity (85.22%) to minimize false positive predictions.

---

## 4. Feature Importance Findings
Using permutation importance on the held-out test data, the most influential features are:
1. **Soil Organic Carbon (`Organic_Carbon_pct`):** Largest impact on prediction performance. Fields with low organic carbon (<0.60%) are consistently classified as not suitable.
2. **Soil pH:** Favorable range is neutral (6.5 to 7.5). Extreme acidity or alkalinity lowers suitability.
3. **Available Phosphorus (`Available_P_kg_ha`):** Important for root development and energy storage in organic cultivation.
4. **Years of Chemical Input (`Years_Chemical_Input`):** Fields with more than 8 years of continuous chemical input have significantly lower suitability rates (<18%).

---

## 5. Project File Structure

```text
Machine-Learning-Based-Organic-Farming-Suitability-Prediction-main/
|-- README.md                                # Project summary and instructions
|-- requirements.txt                         # Required Python libraries
|-- run_pipeline.py                          # Full machine learning pipeline runner
|-- app.py                                   # Streamlit web application
|
|-- data/                                    # Dataset and documentation
|   |-- organic_farming_suitability_1000_source_calibrated.csv
|   |-- organic_farming_dataset_README.txt
|
|-- src/                                     # Source code modules
|   |-- __init__.py
|   |-- config.py                            # Constants, paths, and feature lists
|   |-- data_loader.py                       # Ingestion, validation, and train-test split
|   |-- preprocessing.py                     # Scikit-Learn ColumnTransformer pipeline
|   |-- model_trainer.py                     # Cross-validation and model training
|   |-- eda_engine.py                        # Exploratory data analysis charts
|   |-- evaluation.py                        # Metrics, confusion matrix, ROC plots
|   |-- explainability.py                    # Feature importance analysis
|
|-- notebooks/                               # Jupyter notebook
|   |-- Organic_Farming_Suitability_Prediction.ipynb
|
|-- tests/                                   # Unit and pipeline tests
|   |-- test_pipeline.py
|
|-- models/                                  # Trained model files
|   |-- best_organic_farming_pipeline.joblib
|   |-- model_metadata.json
|
|-- outputs/                                 # Output charts and report files
|   |-- figures/                             # 14 evaluation and EDA plots
|   |-- reports/                             # CSV and JSON benchmark results
|
`-- docs/                                    # Documentation and reports
    |-- PROJECT_REPORT.md                    # Detailed academic report
    |-- DATA_DICTIONARY.md                   # Attribute definitions and ranges
    `-- VIVA_QUESTIONS_ANSWERS.md            # Viva-voce questions and answers
```

---

## 6. How to Run and Test

### 1. Activate Environment
```bash
cd Machine-Learning-Based-Organic-Farming-Suitability-Prediction-main
source .venv/bin/activate
```

### 2. Run Automated Tests
```bash
python -m unittest tests/test_pipeline.py
```

### 3. Run Machine Learning Pipeline
```bash
python run_pipeline.py
```

### 4. Launch Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 5. Open Jupyter Notebook
```bash
jupyter notebook notebooks/Organic_Farming_Suitability_Prediction.ipynb
```

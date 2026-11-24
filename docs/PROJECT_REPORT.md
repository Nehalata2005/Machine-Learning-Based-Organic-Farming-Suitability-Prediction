# Comprehensive Project Report
## EcoAgri-Predict: Machine Learning Framework for Organic Farming Suitability & Agro-Chemical Transition Feasibility

**Candidate Name:** Nehalata Maji  
**Roll No:** BWU/BCA/24/541  
**Programme:** Bachelor of Computer Applications (BCA)  
**Department:** Department of Computational Sciences  
**Institution:** Brainware University, Kolkata, West Bengal  
**Course:** Project Based Learning (PBL) — Machine Learning (Group II)  

---

### Abstract
Transitioning from conventional chemical-intensive agriculture to certified organic farming requires a scientific assessment of soil nutrient reserves, biochemical balances, and residual agro-chemical loads. In this Project Based Learning (PBL) study, we develop **EcoAgri-Predict**, a multi-model machine learning framework designed to classify organic farming suitability (`Suitable` vs. `Not Suitable`) on a 1,000-instance source-calibrated dataset aligned with the Soil Health Card (SHC) framework of India.

To ensure scientific rigor and eliminate optimistic evaluation bias, we strictly exclude `Suitability_Score` from predictive inputs, preventing direct target leakage. We implement a leakage-safe pipeline integrating median and modal imputation, standard scaling, and one-hot encoding within a 5-fold stratified cross-validation setup, followed by unbiased evaluation on a 20% held-out test split ($N=200$). We benchmark six classification algorithms: Logistic Regression, Decision Tree (CART), K-Nearest Neighbors (KNN), Random Forest, Gradient Boosting, and Support Vector Machines (SVM).

Experimental results identify **Gradient Boosting** as the best-performing model, achieving a **Test Accuracy of 83.00%**, an **F1-Score of 0.8000**, and an **ROC-AUC of 0.9003**, outperforming the standard linear baseline (Logistic Regression: F1 = 0.7636, Accuracy = 80.50%) and standard trees (Decision Tree: F1 = 0.6818, Accuracy = 72.00%). Model explainability via permutation importance reveals that Soil Organic Carbon (`Organic_Carbon_pct`), Soil pH, Available Phosphorus, and Historical Chemical Input Duration (`Years_Chemical_Input`) are the foremost governing factors. Finally, an interactive Streamlit application was created to test sample inputs and view predictions.

---

### 1. Introduction & Agricultural Context
Organic agriculture is recognized globally as a sustainable paradigm to reverse soil degradation, preserve biodiversity, and mitigate groundwater contamination caused by prolonged synthetic chemical applications. In India, the National Programme for Organic Production (NPOP) and the Soil Health Card (SHC) initiative provide systematic criteria to assess soil fertility.

However, farmers contemplating conversion face significant yield depression and economic risk during the mandatory 2 to 3-year "conversion period" if the soil's organic biological buffer is depleted or if persistent chemical residues inhibit mycorrhizal networks. A primary challenge in computational agronomy is predicting whether a given field can successfully sustain organic cultivation without catastrophic crop failure.

The objective of this project is to construct, evaluate, and interpret a binary machine learning classification system that predicts `Organic_Farming_Suitability` based on multidimensional soil edaphic properties, climatic metrics, and historical agro-chemical practices.

---

### 2. Scientific Integrity & Dataset Provenance Statement
In compliance with the project guidelines and academic research ethics, the following limitations are explicitly recognized:
1. **Source-Calibrated Synthetic Nature:** The 1,000 instances in `organic_farming_suitability_1000_source_calibrated.csv` do **not** represent 1,000 physically collected field samples. Rather, they are synthetic teaching data calibrated using realistic agronomic parameter ranges derived from official Soil Health Cards, the AgriNet repository (Bareilly, Uttar Pradesh, Mendeley Data 2026), and the Pune SHC Macro Nutrient Dataset.
2. **Instructional Target Definition:** The target variable `Organic_Farming_Suitability` and intermediate `Suitability_Score` are derived from an instructional multi-criteria rubric. They do not constitute official statutory certification or legal organic compliance under NPOP or USDA-NOP.
3. **Target Leakage Prevention:** Because `Organic_Farming_Suitability` is derived by thresholding `Suitability_Score` ($\ge 55.0$), retaining `Suitability_Score` as a feature would yield an artificial 100% predictive accuracy. In this project, `Suitability_Score` and `Sample_ID` are permanently stripped before model training.

---

### 3. Exploratory Data Analysis (EDA) & Domain Observations
The dataset comprises 1,000 instances with 24 initial attributes, exhibiting **0 missing values** and **0 duplicate entries**.

#### 3.1 Target Distribution
- **Not Suitable ($y=0$):** 577 instances ($57.7\%$)
- **Suitable ($y=1$):** 423 instances ($42.3\%$)
The class distribution exhibits moderate balance, precluding severe class imbalance pathologies while requiring stratified splitting to maintain identical ratios across training and testing partitions.

#### 3.2 Edaphic Nutrient Dynamics & Organic Carbon
- **Organic Carbon (`Organic_Carbon_pct`):** Exhibits a pivotal threshold behavior. Fields classified as `Suitable` show a mean organic carbon concentration of $0.98\% \pm 0.28\%$, whereas `Not Suitable` fields exhibit a mean of $0.51\% \pm 0.21\%$. In tropical and subtropical Indian soils, organic carbon $>0.75\%$ is critical for supporting microbial biomass capable of natural nutrient mineralization.
- **Soil pH:** The majority of suitable records fall within the circumneutral range ($6.5 - 7.5$), where macronutrient (N, P, K) and micronutrient bioavailability is optimized.

#### 3.3 The "Transition Burden" (Chemical Input History)
- **Years of Chemical Input (`Years_Chemical_Input`):** Fields with $\ge 8$ years of continuous chemical fertilizer and pesticide application show a steep decline in organic suitability ($<18\%$ suitability rate), reflecting residual soil acidification, pesticide toxicity, and microbial suppression.
- **Previous Fertilizer & Pesticide Regimen:** Stacking analysis demonstrates that fields with prior `Organic` or `Mixed` management achieve suitability rates exceeding $72\%$, whereas fields under intensive `Chemical` management with `High` pesticide usage drop below $12\%$.

---

### 4. Machine Learning Methodology

```text
Raw Data (1,000 instances)
       │
       ▼
Feature Matrix Separation (Exclude Sample_ID & Suitability_Score)
       │
       ▼
Stratified Train-Test Split (80% Train [N=800] / 20% Held-Out Test [N=200])
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
Training Split (N=800)                     Held-Out Test Split (N=200)
       │                                         │
       ▼                                         │
Scikit-Learn ColumnTransformer Pipeline          │
  ├── Numerical: Median Imputer + StandardScaler  │
  └── Categorical: Mode Imputer + OneHotEncoder  │
       │                                         │
       ▼                                         │
5-Fold Stratified Cross-Validation               │
  ├── Logistic Regression (L2 / Liblinear)       │
  ├── Decision Tree (CART / Gini)                │
  ├── K-Nearest Neighbors (k=7 / Minkowski)      │
  ├── Random Forest (150 Trees / Bagging)        │
  ├── Gradient Boosting (120 Trees / Boosting)   │
  └── Support Vector Machine (RBF Kernel)        │
       │                                         │
       ▼                                         │
Model Fitting on Full 800 Train Samples          │
       │                                         │
       └────────────────────┬────────────────────┘
                            │
                            ▼
           Held-Out Evaluation & Metric Computation
             - Accuracy, Precision, Recall, Specificity
             - F1-Score, ROC-AUC, PR-AUC
             - Confusion Matrices & ROC Curves
             - Permutation Feature Importance
```

#### 4.1 Feature Transformation Pipeline
To eliminate data leakage:
- **Numerical Pipeline:** $Z = \frac{X - \mu_{\text{train}}}{\sigma_{\text{train}}}$. Scaler statistics are computed strictly on training folds and applied to test instances.
- **Categorical Pipeline:** One-Hot Encoding with `handle_unknown="ignore"` to safely absorb unobserved categories during inference.

---

### 5. Mathematical Formulation of Evaluation Metrics

1. **Accuracy:** Overall proportion of correct predictions:
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Precision (Positive Predictive Value):** Probability that a predicted suitable field is genuinely suitable:
   $$\text{Precision} = \frac{TP}{TP + FP}$$
3. **Recall (Sensitivity):** Proportion of actual suitable fields successfully identified:
   $$\text{Recall} = \frac{TP}{TP + FN}$$
4. **Specificity (True Negative Rate):** Proportion of actual non-suitable fields correctly identified:
   $$\text{Specificity} = \frac{TN}{TN + FP}$$
5. **F1-Score:** Harmonic mean of Precision and Recall, critical for balanced assessment:
   $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$
6. **Area Under the ROC Curve (ROC-AUC):** Integral of the True Positive Rate against the False Positive Rate across all decision thresholds $\tau \in [0, 1]$:
   $$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\tau) \, d(\text{FPR}(\tau))$$

---

### 6. Experimental Results & Comparative Analysis

Evaluation on the 200 held-out test instances ($115 \text{ Not Suitable}, 85 \text{ Suitable}$) yielded the following benchmark:

| Model Architecture | 5-Fold CV F1 | Test Accuracy | Test Precision | Test Recall | Test Specificity | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **0.8118** | **0.8300** | **0.8000** | **0.8000** | **0.8522** | **0.8000** | **0.9003** |
| **Logistic Regression** | 0.8002 | 0.8050 | 0.7875 | 0.7412 | 0.8522 | 0.7636 | 0.8899 |
| **Support Vector Machine (RBF)** | 0.7973 | 0.7950 | 0.7821 | 0.7176 | 0.8522 | 0.7485 | 0.8978 |
| **Random Forest** | 0.7696 | 0.7750 | 0.7564 | 0.6941 | 0.8348 | 0.7239 | 0.8762 |
| **K-Nearest Neighbors ($k=7$)** | 0.7105 | 0.7700 | 0.7600 | 0.6706 | 0.8435 | 0.7125 | 0.8362 |
| **Decision Tree (CART)** | 0.7031 | 0.7200 | 0.6593 | 0.7059 | 0.7304 | 0.6818 | 0.7863 |

#### 6.1 Performance Analysis & Error Trade-offs
- **Gradient Boosting Performance:** Achieved the highest Test Accuracy (83.00%), Test F1-Score (0.8000), and Test ROC-AUC (0.9003). By sequentially fitting decision stumps to the pseudo-residuals of preceding iterations, Gradient Boosting effectively captured the non-linear interaction between soil chemical latency (`Years_Chemical_Input`) and bio-available nutrients.
- **Type I vs. Type II Errors:** In agricultural organic certification, **False Positives (Type I errors)** are significantly more dangerous than False Negatives (Type II errors). Certifying an unsuitable chemical-laden field as "Organic" causes brand degradation, certification forfeiture, and crop failure. Gradient Boosting limited False Positives to 17 (Specificity = 85.22%) while identifying 68 out of 85 truly suitable fields ($TP=68$).
- **Linear Baseline Comparison:** Logistic Regression demonstrated commendable performance (ROC-AUC = 0.8899), confirming that several edaphic nutrients exhibit monotonic linear relationships with suitability, but fell short of ensemble boosting on boundary cases.

---

### 7. Feature Explainability & Agronomic Insights
Using model-agnostic **Permutation Feature Importance** on held-out test data, we measured the average drop in test F1-score when each predictor was randomly shuffled:

1. **`Organic_Carbon_pct` ($\Delta \text{F1} = 0.1804 \pm 0.043$):** Dominant predictor. Shuffling organic carbon degrades model F1 by 18 percentage points, reflecting that soil carbon is the lifeblood of organic microbial cycling.
2. **`pH` ($\Delta \text{F1} = 0.0962 \pm 0.020$):** Soil acidity or alkalinity strictly controls chemical solubility and nutrient uptake.
3. **`Available_P_kg_ha` ($\Delta \text{F1} = 0.0407$):** Phosphorus availability is often the primary growth-limiting factor in organic farming, as rock phosphate mineralizes slowly without active soil biology.
4. **`Available_K_kg_ha` ($\Delta \text{F1} = 0.0368$):** Potassium governs osmotic regulation and disease resistance.
5. **`Previous_Fertilizer_Usage` ($\Delta \text{F1} = 0.0235$):** Historical chemical applications impose residual biological latency.
6. **`Years_Chemical_Input` & `Previous_Pesticide_Usage`:** Together define the conversion feasibility barrier.

---

### 8. Agronomic Advisory & Transition Protocol
Based on our findings, we formulate a decision framework for extension agents:
- **Immediate Suitability ($\ge 75\%$ probability):** Fields with $\text{OC} > 0.8\%$, $\text{pH} \in [6.5, 7.5]$, and $\le 2$ years of chemical input can enter immediate organic certification.
- **Conditional Transition (35% – 70% probability):** Fields with moderate chemical latency require a mandatory 3-year conversion regimen incorporating green manuring (Dhaincha/Sunn hemp), Azotobacter bio-fertilizers, and microbial consortia.
- **High Risk ($<35\%$ probability):** Heavily depleted soils with $>10$ years of continuous chemical input require deep subsoiling, gypsum/lime conditioning, and heavy vermicompost application before organic viability is attainable.

---

### 9. Conclusion & Future Scope
This project successfully implemented an end-to-end, scientifically validated machine learning workflow for organic farming suitability prediction. By adhering to rigorous data-leakage controls, benchmarking multiple families of algorithms, evaluating with comprehensive metrics, and deploying an interactive Streamlit tool, the project fulfills all objectives outlined in the PBL curriculum.

**Future Work:**
- Integrating authenticated laboratory datasets from the national Soil Health Card API.
- Employing Deep Neural Networks (TabNet) and SHAP (SHapley Additive exPlanations) values for localized agronomic interpretability.
- Incorporating satellite remote sensing spectral indices (NDVI, NDRE) for continuous spatial soil monitoring.

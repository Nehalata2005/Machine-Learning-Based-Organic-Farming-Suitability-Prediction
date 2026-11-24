ORGANIC FARMING SUITABILITY — 1000-INSTANCE PBL DATASET

STATUS
-------
This package is SOURCE-CALIBRATED SYNTHETIC DATA. It is NOT a collection of 1,000
authentic field/laboratory records. The raw rows were generated for teaching purposes
using realistic ranges and variables aligned with India's Soil Health Card framework.

WHY THIS DISTINCTION MATTERS
----------------------------
The Government of India's Soil Health Card programme measures pH, EC, organic carbon,
N, P, K, S, Zn, Fe, Mn, Cu and B. The public national SHC dataset is much larger than
1,000 records, but its current downloadable resource is approximately 1 GiB and was
not directly ingested here. Also, "organic farming suitability" is not an SHC laboratory
target variable. Therefore it would be scientifically incorrect to claim that these
1,000 labels are authentic observations.

LABEL
-----
Suitability_Score and Organic_Farming_Suitability are derived from a transparent
instructional rubric using soil health and management variables. The rubric is NOT
an official certification or legal standard for organic farming.

RECOMMENDED STUDENT USE
-----------------------
1. Use this dataset to learn preprocessing, EDA, classification and model comparison.
2. Treat Organic_Farming_Suitability as the target.
3. Compare Logistic Regression, Decision Tree, Random Forest and KNN.
4. Report confusion matrix, precision, recall, F1 and ROC-AUC.
5. Perform feature importance/explainability.
6. In the final report explicitly state that the dataset is synthetic.

AUTHENTIC SOURCES TO USE FOR A RESEARCH-GRADE VERSION
------------------------------------------------------
- Government of India Soil Health Card programme.
- India Data Portal: Soil Health Card / Soil Nutrient Analysis, 2023–2025.
- AgriNet (Mendeley Data, 2026): 258 laboratory-tested soil records from Bareilly,
  Uttar Pradesh, extracted from official Soil Health Cards.
- Soil Macro Nutrient Dataset for Fertilizer Combinations Advice, Pune
  (Mendeley Data, 2026), based on Soil Health Cards from Pune district.

Suggested next step for a research-grade dataset: download the underlying authenticated
records, preserve their original provenance, harmonize units, deduplicate, and create
the suitability target through a documented agronomist/expert labeling protocol rather
than synthetic labels.

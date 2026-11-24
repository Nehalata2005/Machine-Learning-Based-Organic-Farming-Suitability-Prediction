# Comprehensive Data Dictionary
## EcoAgri-Predict: Organic Farming Suitability & Transition Dataset

This data dictionary provides complete technical and agronomic specifications for all 24 attributes present in `organic_farming_suitability_1000_source_calibrated.csv`.

---

### 1. Attribute Taxonomy Overview

| Category | Column Count | Features Included |
| :--- | :---: | :--- |
| **Identifier & Provenance** | 1 | `Sample_ID` |
| **Contextual & Categorical** | 3 | `State`, `Soil_Type`, `Crop` |
| **Soil Edaphic & Nutrients** | 12 | `pH`, `Organic_Carbon_pct`, `Available_N_kg_ha`, `Available_P_kg_ha`, `Available_K_kg_ha`, `Available_S_mg_kg`, `Available_Zn_mg_kg`, `Available_Fe_mg_kg`, `Available_Mn_mg_kg`, `Available_Cu_mg_kg`, `Available_B_mg_kg`, `EC_dS_m` |
| **Agro-Climatic & Environmental** | 3 | `Annual_Rainfall_mm`, `Mean_Temperature_C`, `Soil_Moisture_pct` |
| **Agro-Chemical Latency** | 3 | `Previous_Fertilizer_Usage`, `Previous_Pesticide_Usage`, `Years_Chemical_Input` |
| **Target Construction Field** | 1 | `Suitability_Score` *(Excluded from ML modeling to avoid target leakage)* |
| **Target Variable** | 1 | `Organic_Farming_Suitability` |

---

### 2. Detailed Column Specifications

#### 2.1 Identifiers & Metadata
- **`Sample_ID`**
  - **Type:** String / Identifier (e.g., `OF-0001` to `OF-1000`)
  - **Description:** Unique synthetic record identifier.
  - **Role in Modeling:** Excluded (`drop`). Identifiers carry zero generalized statistical signal and risk spurious memorization.

#### 2.2 Geographical & Crop Context
- **`State`**
  - **Type:** Categorical (Nominal)
  - **Values:** Andhra Pradesh, Bihar, Haryana, Karnataka, Madhya Pradesh, Rajasthan, Tamil Nadu, Telangana, West Bengal, etc.
  - **Description:** Indian state where soil agro-climatic conditions are calibrated.
  - **Preprocessing:** One-Hot Encoded with `handle_unknown='ignore'`.
- **`Soil_Type`**
  - **Type:** Categorical (Nominal)
  - **Values:** `Alluvial`, `Black`, `Red`, `Loamy`, `Sandy`, `Laterite`, `Clay Loam`
  - **Description:** Soil texture and pedological classification. Directly influences drainage, cation exchange capacity (CEC), and water holding capacity.
  - **Preprocessing:** One-Hot Encoded.
- **`Crop`**
  - **Type:** Categorical (Nominal)
  - **Values:** `Rice`, `Wheat`, `Tomato`, `Pulses`, `Sugarcane`, `Soybean`, `Mustard`, `Chickpea`, `Groundnut`
  - **Description:** Target cultivated crop, defining baseline nutrient demand.
  - **Preprocessing:** One-Hot Encoded.

#### 2.3 Soil Chemical & Nutrient Attributes (Edaphic Factors)
- **`pH`**
  - **Type:** Continuous Float (Typical range: $4.5 - 9.5$)
  - **Unit:** Standard pH scale ($-\log[H^+]$)
  - **Agronomic Role:** Master variable governing soil chemical equilibrium. Values between $6.5$ and $7.5$ maximize availability of primary macronutrients; extreme acidity ($<5.5$) induces aluminum toxicity, while extreme alkalinity ($>8.5$) precipitates phosphorus and micronutrients.
  - **Preprocessing:** Median Imputed $\rightarrow$ `StandardScaler`.
- **`Organic_Carbon_pct`**
  - **Type:** Continuous Float (Typical range: $0.1\% - 2.5\%$)
  - **Unit:** Percentage ($w/w$)
  - **Agronomic Role:** The bedrock of organic agriculture. Reflects humus content, soil microbial biomass, and intrinsic biological buffering capacity. Values $>0.75\%$ indicate high organic potential.
  - **Preprocessing:** Median Imputed $\rightarrow$ `StandardScaler`.
- **`Available_N_kg_ha`**
  - **Type:** Continuous Float (Range: $50 - 550$ kg/ha)
  - **Unit:** Kilograms per hectare
  - **Agronomic Role:** Readily mineralizable nitrogen available for vegetative plant expansion.
  - **Preprocessing:** Median Imputed $\rightarrow$ `StandardScaler`.
- **`Available_P_kg_ha`**
  - **Type:** Continuous Float (Range: $2 - 100$ kg/ha)
  - **Unit:** Kilograms per hectare (Olsen or Bray extractable phosphorus)
  - **Agronomic Role:** Essential for root development, adenosine triphosphate (ATP) synthesis, and early crop vigor.
  - **Preprocessing:** Median Imputed $\rightarrow$ `StandardScaler`.
- **`Available_K_kg_ha`**
  - **Type:** Continuous Float (Range: $40 - 550$ kg/ha)
  - **Unit:** Kilograms per hectare (Ammonium acetate extractable potassium)
  - **Agronomic Role:** Regulates stomatal conductance, water stress resilience, and enzymatic activation.
  - **Preprocessing:** Median Imputed $\rightarrow$ `StandardScaler`.
- **`Available_S_mg_kg`**
  - **Type:** Continuous Float (Range: $2 - 50$ mg/kg)
  - **Unit:** Milligrams per kilogram (ppm)
  - **Agronomic Role:** Crucial for synthesis of sulfur-containing amino acids (methionine, cysteine) and chlorophyll synthesis, particularly in oilseeds and pulses.
  - **Preprocessing:** Median Imputed $\rightarrow$ `StandardScaler`.
- **`Available_Zn_mg_kg`, `Available_Fe_mg_kg`, `Available_Mn_mg_kg`, `Available_Cu_mg_kg`, `Available_B_mg_kg`**
  - **Type:** Continuous Float
  - **Unit:** Milligrams per kilogram (DTPA-extractable micronutrients)
  - **Agronomic Role:** Essential trace cofactors for enzymatic reactions. Deficiencies severely depress organic yield potential.
  - **Preprocessing:** Median Imputed $\rightarrow$ `StandardScaler`.
- **`EC_dS_m`**
  - **Type:** Continuous Float (Range: $0.05 - 3.5$ dS/m)
  - **Unit:** DeciSiemens per meter (Electrical Conductivity)
  - **Agronomic Role:** Metric of soluble salt concentration. Values $>1.5$ dS/m indicate salinity hazards that retard seedling emergence.
  - **Preprocessing:** Median Imputed $\rightarrow$ `StandardScaler`.

#### 2.4 Agro-Climatic & Environmental Metrics
- **`Annual_Rainfall_mm`**
  - **Type:** Continuous Float (Range: $300 - 2500$ mm)
  - **Unit:** Millimeters per annum
  - **Agronomic Role:** Dictates soil moisture regimes and leaching dynamics.
  - **Preprocessing:** Median Imputed $\rightarrow$ `StandardScaler`.
- **`Mean_Temperature_C`**
  - **Type:** Continuous Float (Range: $15 - 38$ °C)
  - **Unit:** Degrees Celsius
  - **Agronomic Role:** Dictates soil microbial decomposition rates and evapotranspiration.
  - **Preprocessing:** Median Imputed $\rightarrow$ `StandardScaler`.
- **`Soil_Moisture_pct`**
  - **Type:** Continuous Float (Range: $8\% - 55\%$)
  - **Unit:** Volumetric percentage
  - **Agronomic Role:** Determines aerobic versus anaerobic micro-environments in the rhizosphere.
  - **Preprocessing:** Median Imputed $\rightarrow$ `StandardScaler`.

#### 2.5 Agro-Chemical Latency & Transition Predictors
- **`Previous_Fertilizer_Usage`**
  - **Type:** Categorical (Ordinal/Nominal)
  - **Values:** `Organic`, `Mixed`, `Chemical`
  - **Agronomic Role:** Identifies historical fertilization practices. Prior organic use indicates active microbial consortia.
  - **Preprocessing:** One-Hot Encoded.
- **`Previous_Pesticide_Usage`**
  - **Type:** Categorical (Ordinal/Nominal)
  - **Values:** `Low`, `Moderate`, `High`
  - **Agronomic Role:** Indicates pesticide residue risk and non-target beneficial insect suppression.
  - **Preprocessing:** One-Hot Encoded.
- **`Years_Chemical_Input`**
  - **Type:** Integer / Continuous (Range: $0 - 20$ years)
  - **Unit:** Cumulative years of continuous synthetic chemical application
  - **Agronomic Role:** Direct proxy for the **Transition Latency**. High values necessitate extended detox/remediation conversion periods.
  - **Preprocessing:** Median Imputed $\rightarrow$ `StandardScaler`.

#### 2.6 Target Construction Field & Binary Target
- **`Suitability_Score`**
  - **Type:** Continuous Float (Range: $4.5 - 89.3$)
  - **Description:** Synthetic scoring rubric aggregating nutrients and management factors.
  - **CRITICAL STATUS:** **EXCLUDED FROM ALL MODEL PREDICTORS** to eliminate target leakage, as `Organic_Farming_Suitability` is derived directly from this score ($\ge 55.0$).
- **`Organic_Farming_Suitability` (TARGET)**
  - **Type:** Binary Categorical
  - **Values:**
    - `0` / `Not Suitable`: Field requires remediation, nutrient supplementation, or multi-year conversion detox before organic standards can be met.
    - `1` / `Suitable`: Field exhibits sufficient biological buffer, favorable pH, adequate nutrients, and manageable chemical legacy for organic cultivation.

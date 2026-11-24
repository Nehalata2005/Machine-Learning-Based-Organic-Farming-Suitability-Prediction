from pathlib import Path

# Project Root Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Directories & Files
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "organic_farming_suitability_1000_source_calibrated.csv"
README_PATH = DATA_DIR / "organic_farming_dataset_README.txt"

# Output Directories
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

# Target & Feature Definitions
TARGET_COL = "Organic_Farming_Suitability"
TARGET_MAP = {"Not Suitable": 0, "Suitable": 1}
TARGET_INV_MAP = {0: "Not Suitable", 1: "Suitable"}

# Columns excluded from model features
# NOTE: Suitability_Score MUST be excluded to prevent target leakage,
# as Organic_Farming_Suitability is derived directly from it (threshold >= 55.0).
EXCLUDED_COLS = ["Sample_ID", "Suitability_Score"]

# Categorical Predictors
CATEGORICAL_COLS = [
    "State",
    "Soil_Type",
    "Crop",
    "Previous_Fertilizer_Usage",
    "Previous_Pesticide_Usage"
]

# Numerical Predictors
NUMERICAL_COLS = [
    # Soil Chemical & Nutrient Properties
    "pH",
    "Organic_Carbon_pct",
    "Available_N_kg_ha",
    "Available_P_kg_ha",
    "Available_K_kg_ha",
    "Available_S_mg_kg",
    "Available_Zn_mg_kg",
    "Available_Fe_mg_kg",
    "Available_Mn_mg_kg",
    "Available_Cu_mg_kg",
    "Available_B_mg_kg",
    "EC_dS_m",
    # Environmental & Climatic Metrics
    "Annual_Rainfall_mm",
    "Mean_Temperature_C",
    "Soil_Moisture_pct",
    # Agro-Chemical Latency / Conversion Burden
    "Years_Chemical_Input"
]

# All Feature Columns
FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS

# Experiment Reproducibility Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS_CV = 5

import unittest
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from src.config import RAW_DATA_PATH, MODELS_DIR, FEATURE_COLS, TARGET_MAP
from src.data_loader import load_raw_dataset, prepare_features_and_target, get_stratified_train_test_split
from src.preprocessing import build_preprocessor


class TestEcoAgriPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_raw_dataset()
        cls.model_path = MODELS_DIR / "best_organic_farming_pipeline.joblib"
        cls.pipeline = joblib.load(cls.model_path) if cls.model_path.exists() else None

    def test_01_dataset_integrity(self):
        """Verify dataset file exists, has correct shape and zero missing values."""
        self.assertTrue(RAW_DATA_PATH.exists(), "Dataset CSV file not found")
        self.assertEqual(len(self.df), 1000, "Dataset should contain exactly 1000 instances")
        self.assertEqual(self.df.isnull().sum().sum(), 0, "Dataset must not have missing values")
        self.assertIn("Organic_Farming_Suitability", self.df.columns)
        self.assertIn("Suitability_Score", self.df.columns)

    def test_02_target_leakage_prevention(self):
        """Ensure Suitability_Score and Sample_ID are excluded from model features."""
        X, y = prepare_features_and_target(self.df)
        self.assertNotIn("Suitability_Score", X.columns, "CRITICAL: Suitability_Score must be excluded to prevent leakage")
        self.assertNotIn("Sample_ID", X.columns, "Sample_ID must be excluded")
        self.assertEqual(X.shape[1], len(FEATURE_COLS))
        self.assertEqual(len(y), 1000)

    def test_03_stratified_split_ratios(self):
        """Verify train-test split maintains exact 80/20 stratified proportions."""
        X_train, X_test, y_train, y_test = get_stratified_train_test_split(self.df)
        self.assertEqual(len(X_train), 800)
        self.assertEqual(len(X_test), 200)
        
        # Check class balance preservation
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        self.assertAlmostEqual(train_ratio, test_ratio, delta=0.02)

    def test_04_preprocessor_pipeline(self):
        """Verify ColumnTransformer transforms features without error."""
        X, y = prepare_features_and_target(self.df)
        preprocessor = build_preprocessor()
        X_trans = preprocessor.fit_transform(X)
        self.assertEqual(X_trans.shape[0], 1000)
        self.assertTrue(np.all(np.isfinite(X_trans)), "Transformed array contains non-finite numbers")

    def test_05_saved_model_prediction(self):
        """Verify saved champion pipeline can load and make valid predictions."""
        self.assertIsNotNone(self.pipeline, "Champion model pipeline file does not exist")
        
        sample_input = pd.DataFrame([{
            "State": "West Bengal",
            "Soil_Type": "Alluvial",
            "Crop": "Rice",
            "pH": 6.8,
            "Organic_Carbon_pct": 0.95,
            "Available_N_kg_ha": 250.0,
            "Available_P_kg_ha": 35.0,
            "Available_K_kg_ha": 200.0,
            "Available_S_mg_kg": 18.0,
            "Available_Zn_mg_kg": 1.1,
            "Available_Fe_mg_kg": 12.0,
            "Available_Mn_mg_kg": 8.0,
            "Available_Cu_mg_kg": 0.8,
            "Available_B_mg_kg": 0.6,
            "EC_dS_m": 0.35,
            "Annual_Rainfall_mm": 1200.0,
            "Mean_Temperature_C": 27.0,
            "Soil_Moisture_pct": 32.0,
            "Previous_Fertilizer_Usage": "Organic",
            "Previous_Pesticide_Usage": "Low",
            "Years_Chemical_Input": 1
        }])

        prediction = self.pipeline.predict(sample_input)[0]
        self.assertIn(prediction, [0, 1], "Prediction must be binary (0 = Not Suitable, 1 = Suitable)")

        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(sample_input)[0]
            self.assertEqual(len(probs), 2)
            self.assertAlmostEqual(sum(probs), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()

"""
EcoAgri-Predict: End-to-End Machine Learning Pipeline Execution Script
Developed by: Nehalata Maji (Roll No: BWU/BCA/24/541, BCA, Brainware University)
"""

import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.data_loader import (
    load_raw_dataset,
    audit_dataset,
    get_stratified_train_test_split
)
from src.eda_engine import generate_all_eda_figures
from src.model_trainer import train_and_benchmark_models
from src.evaluation import generate_evaluation_visualizations
from src.explainability import analyze_feature_importance


def run_pipeline():
    print("=" * 75)
    print("ECOAGRI-PREDICT: MACHINE LEARNING WORKFLOW EXECUTION")
    print("Student: Nehalata Maji | Roll No: BWU/BCA/24/541 | BCA, Brainware University")
    print("=" * 75)

    # 1. Ingestion & Audit
    print("\n>>> STEP 1: Loading Dataset & Running Integrity Audit...")
    df = load_raw_dataset()
    audit = audit_dataset(df)
    print(f"Dataset Loaded: {audit['total_records']} rows, {audit['total_columns']} columns.")
    print(f"Class Balance: {audit['target_distribution']} ({audit['target_percentages']}%)")
    print(f"Missing Values: {audit['has_missing_values']} | Duplicates: {audit['duplicate_rows']}")

    # 2. Exploratory Data Analysis
    print("\n>>> STEP 2: Generating Exploratory Visualizations...")
    generate_all_eda_figures(df)

    # 3. Train-Test Split (Leakage-Safe)
    print("\n>>> STEP 3: Performing Stratified 80/20 Train-Test Split...")
    X_train, X_test, y_train, y_test = get_stratified_train_test_split(df)
    print(f"Training Set: {X_train.shape[0]} samples | Test Set: {X_test.shape[0]} samples")

    # 4. Multi-Model Training & Benchmark
    print("\n>>> STEP 4: Training & Benchmarking Classification Models...")
    results_df, fitted_pipelines, best_model_name = train_and_benchmark_models(
        X_train, y_train, X_test, y_test
    )
    print("\nBenchmark Summary (Ranked by Test F1-Score):")
    print(results_df[["Model", "CV_F1_Mean", "Test_Accuracy", "Test_F1_Score", "Test_ROC_AUC"]].to_string(index=False))

    # 5. Visual Evaluation
    print("\n>>> STEP 5: Generating Evaluation Visualizations...")
    generate_evaluation_visualizations(
        results_df, fitted_pipelines, X_test, y_test, df_raw=df
    )

    # 6. Feature Explainability
    print("\n>>> STEP 6: Conducting Feature Importance & Agronomic Analysis...")
    analyze_feature_importance(fitted_pipelines, X_test, y_test, best_model_name)

    print("\n" + "=" * 75)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("All models, reports, and publication figures have been generated.")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline()

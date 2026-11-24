import json
import warnings
warnings.filterwarnings("ignore")
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.config import (
    RANDOM_STATE,
    N_SPLITS_CV,
    MODELS_DIR,
    REPORTS_DIR
)
from src.preprocessing import build_preprocessor


def get_candidate_models() -> dict:
    """Instantiates the candidate classifiers with controlled random seeds."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=0.8,
            solver="liblinear",
            random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="minkowski"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            random_state=RANDOM_STATE
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            C=1.5,
            probability=True,
            random_state=RANDOM_STATE
        )
    }


def train_and_benchmark_models(X_train, y_train, X_test, y_test):
    """
    Trains each classifier inside a dedicated Pipeline containing the preprocessor,
    computes 5-fold cross-validation metrics, tests on held-out test data,
    and returns comprehensive benchmark results and fitted pipelines.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    models_dict = get_candidate_models()
    benchmark_records = []
    fitted_pipelines = {}
    detailed_reports = {}

    cv_strategy = StratifiedKFold(
        n_splits=N_SPLITS_CV,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    print("=" * 70)
    print("BEGINNING MACHINE LEARNING BENCHMARK (5-FOLD CV + TEST EVALUATION)")
    print("=" * 70)

    for name, clf in models_dict.items():
        print(f"\n[Training & Evaluating] {name}...")

        # Construct end-to-end pipeline: Preprocessing -> Estimator
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", clf)
            ]
        )

        # 5-Fold Stratified Cross Validation on Training Data
        scoring = ["accuracy", "f1", "roc_auc", "precision", "recall"]
        cv_res = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=-1
        )

        mean_cv_acc = np.mean(cv_res["test_accuracy"])
        mean_cv_f1 = np.mean(cv_res["test_f1"])
        mean_cv_auc = np.mean(cv_res["test_roc_auc"])

        # Fit on full training split
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Predictions on held-out test split
        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_proba = pipeline.decision_function(X_test)
        else:
            y_proba = y_pred

        # Compute held-out evaluation metrics
        test_acc = accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred, zero_division=0)
        test_rec = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        test_auc = roc_auc_score(y_test, y_proba)

        # Confusion Matrix components (TN, FP, FN, TP)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        record = {
            "Model": name,
            "CV_Accuracy_Mean": round(float(mean_cv_acc), 4),
            "CV_F1_Mean": round(float(mean_cv_f1), 4),
            "CV_ROC_AUC_Mean": round(float(mean_cv_auc), 4),
            "Test_Accuracy": round(float(test_acc), 4),
            "Test_Precision": round(float(test_prec), 4),
            "Test_Recall": round(float(test_rec), 4),
            "Test_Specificity": round(float(specificity), 4),
            "Test_F1_Score": round(float(test_f1), 4),
            "Test_ROC_AUC": round(float(test_auc), 4),
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn)
        }
        benchmark_records.append(record)

        # Store classification report
        clf_rep = classification_report(
            y_test,
            y_pred,
            target_names=["Not Suitable", "Suitable"],
            output_dict=True
        )
        detailed_reports[name] = clf_rep

        print(
            f"   -> CV F1: {mean_cv_f1:.4f} | Test Acc: {test_acc:.4f} | "
            f"Test F1: {test_f1:.4f} | Test ROC-AUC: {test_auc:.4f}"
        )

    # Convert to DataFrame and sort by Test F1-Score, then ROC-AUC
    results_df = pd.DataFrame(benchmark_records)
    results_df = results_df.sort_values(
        by=["Test_F1_Score", "Test_ROC_AUC", "Test_Accuracy"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    # Save benchmark table
    results_df.to_csv(REPORTS_DIR / "model_benchmark_results.csv", index=False)
    with open(REPORTS_DIR / "model_benchmark_results.json", "w") as f:
        json.dump(benchmark_records, f, indent=4)
    with open(REPORTS_DIR / "classification_reports.json", "w") as f:
        json.dump(detailed_reports, f, indent=4)

    # Select best model
    best_model_name = results_df.iloc[0]["Model"]
    best_pipeline = fitted_pipelines[best_model_name]

    print("\n" + "=" * 70)
    print(f"Best performing model: {best_model_name}")
    print(
        f"Metrics -> Accuracy: {results_df.iloc[0]['Test_Accuracy']} | "
        f"F1: {results_df.iloc[0]['Test_F1_Score']} | "
        f"ROC-AUC: {results_df.iloc[0]['Test_ROC_AUC']}"
    )
    print("=" * 70)

    # Save best model artifact
    joblib.dump(best_pipeline, MODELS_DIR / "best_organic_farming_pipeline.joblib")

    # Save metadata
    metadata = {
        "best_model_name": best_model_name,
        "selection_criteria": "Highest Test F1-Score with ROC-AUC tie-breaker",
        "performance_metrics": results_df.iloc[0].to_dict(),
        "random_state": RANDOM_STATE,
        "target_mapping": {"Not Suitable": 0, "Suitable": 1},
        "all_ranked_models": results_df["Model"].tolist()
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return results_df, fitted_pipelines, best_model_name

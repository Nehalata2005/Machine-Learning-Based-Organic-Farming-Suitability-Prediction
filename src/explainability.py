import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from src.config import FIGURES_DIR, REPORTS_DIR, RANDOM_STATE
from src.preprocessing import get_feature_names_after_preprocessing


def analyze_feature_importance(fitted_pipelines, X_test, y_test, best_model_name):
    """
    Computes both Tree-based MDI importance and Model-Agnostic Permutation Importance
    to understand key agronomic drivers of organic farming suitability.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[Explainability Engine] Computing feature importance metrics...")

    # Identify best tree model or random forest
    target_pipe = fitted_pipelines.get("Random Forest", fitted_pipelines[best_model_name])
    preprocessor = target_pipe.named_steps["preprocessor"]
    classifier = target_pipe.named_steps["classifier"]
    
    transformed_feature_names = get_feature_names_after_preprocessing(preprocessor)

    # 1. Random Forest Gini / MDI Feature Importance
    rf_pipe = fitted_pipelines.get("Random Forest")
    if rf_pipe and hasattr(rf_pipe.named_steps["classifier"], "feature_importances_"):
        rf_clf = rf_pipe.named_steps["classifier"]
        importances = rf_clf.feature_importances_
        
        fi_df = pd.DataFrame({
            "Feature": transformed_feature_names,
            "Importance_MDI": importances
        }).sort_values(by="Importance_MDI", ascending=False)

        # Plot Top 15 Features
        top_15_mdi = fi_df.head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=top_15_mdi,
            x="Importance_MDI",
            y="Feature",
            palette="viridis",
            hue="Feature",
            legend=False,
            edgecolor="#333",
            ax=ax
        )
        ax.set_title("Top 15 Agronomic & Chemical Features (Random Forest MDI Importance)", pad=15)
        ax.set_xlabel("Mean Decrease in Impurity (Gini Importance)")
        ax.set_ylabel("Predictor Variable")
        plt.savefig(FIGURES_DIR / "12_random_forest_feature_importance.png", dpi=200)
        plt.close()
    else:
        fi_df = pd.DataFrame({"Feature": transformed_feature_names})

    # 2. Model-Agnostic Permutation Importance on Held-Out Test Set
    perm_result = permutation_importance(
        target_pipe,
        X_test,
        y_test,
        n_repeats=15,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring="f1"
    )

    perm_df = pd.DataFrame({
        "Original_Feature": X_test.columns,
        "Permutation_Mean_F1_Drop": perm_result.importances_mean,
        "Permutation_Std": perm_result.importances_std
    }).sort_values(by="Permutation_Mean_F1_Drop", ascending=False)

    # Plot Top 15 Permutation Features
    top_15_perm = perm_df.head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=top_15_perm,
        x="Permutation_Mean_F1_Drop",
        y="Original_Feature",
        palette="magma",
        hue="Original_Feature",
        legend=False,
        edgecolor="#333",
        ax=ax
    )
    ax.errorbar(
        x=top_15_perm["Permutation_Mean_F1_Drop"],
        y=range(len(top_15_perm)),
        xerr=top_15_perm["Permutation_Std"],
        fmt="none",
        c="black",
        capsize=3
    )
    ax.set_title(
        f"Top Feature Importance via Permutation Testing on Held-Out Test Data ({best_model_name})",
        pad=15
    )
    ax.set_xlabel("Mean Drop in Test F1-Score Upon Feature Shuffle")
    ax.set_ylabel("Original Input Variable")
    plt.savefig(FIGURES_DIR / "13_permutation_importance_held_out.png", dpi=200)
    plt.close()

    # Save comprehensive feature importance table
    perm_df.to_csv(REPORTS_DIR / "feature_importance_analysis.csv", index=False)
    print("[Explainability Engine] Feature importance analysis complete.")
    return perm_df

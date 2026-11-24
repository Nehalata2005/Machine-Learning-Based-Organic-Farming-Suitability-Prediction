import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from src.config import FIGURES_DIR, REPORTS_DIR


def generate_evaluation_visualizations(results_df, fitted_pipelines, X_test, y_test, df_raw=None):
    """
    Generates comprehensive, multi-model evaluation graphics:
    1. Multi-metric comparison bar plot
    2. 2x3 grid of confusion matrices
    3. Combined ROC-AUC curves
    4. Combined Precision-Recall curves
    5. Transition Feasibility Matrix
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print("[Evaluation Engine] Generating model performance figures...")

    # 1. Multi-Metric Benchmark Comparison Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    melted_df = results_df.melt(
        id_vars=["Model"],
        value_vars=["Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1_Score", "Test_ROC_AUC"],
        var_name="Metric",
        value_name="Score"
    )
    metric_labels = {
        "Test_Accuracy": "Accuracy",
        "Test_Precision": "Precision",
        "Test_Recall": "Recall",
        "Test_F1_Score": "F1-Score",
        "Test_ROC_AUC": "ROC-AUC"
    }
    melted_df["Metric"] = melted_df["Metric"].map(metric_labels)

    palette = sns.color_palette("Set2", 5)
    sns.barplot(
        data=melted_df,
        x="Model",
        y="Score",
        hue="Metric",
        palette=palette,
        edgecolor="#333",
        linewidth=0.8,
        ax=ax
    )
    ax.set_ylim(0.5, 1.05)
    ax.set_title("Comparative Evaluation Across Machine Learning Algorithms (Held-Out Test Set)", pad=15)
    ax.set_xlabel("Candidate Algorithm")
    ax.set_ylabel("Metric Score (0.0 - 1.0)")
    ax.legend(loc="lower right", frameon=True)
    plt.xticks(rotation=15, ha="right")
    plt.savefig(FIGURES_DIR / "08_model_accuracy_f1_roc_comparison.png", dpi=200)
    plt.close()

    # 2. Confusion Matrices in 2x3 Grid
    models = list(fitted_pipelines.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, name in enumerate(models):
        pipe = fitted_pipelines[name]
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            annot_kws={"size": 13, "weight": "bold"},
            xticklabels=["Not Suitable", "Suitable"],
            yticklabels=["Not Suitable", "Suitable"]
        )
        axes[idx].set_title(f"{name}\n(Acc: {results_df[results_df['Model'] == name]['Test_Accuracy'].values[0]:.3f})")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")

    plt.suptitle("Confusion Matrix Comparison Across All Six Evaluated Models", y=1.02, fontsize=15)
    plt.savefig(FIGURES_DIR / "09_confusion_matrices_all_models.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 3. Combined ROC-AUC Curves
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = sns.color_palette("tab10", len(models))

    for idx, name in enumerate(models):
        pipe = fitted_pipelines[name]
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe, "decision_function"):
            y_proba = pipe.decision_function(X_test)
        else:
            y_proba = pipe.predict(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc_val = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc_val:.3f})",
            color=colors[idx],
            linewidth=2.2
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance Level (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) Benchmark Curves", pad=15)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    plt.savefig(FIGURES_DIR / "10_combined_roc_auc_curves.png", dpi=200)
    plt.close()

    # 4. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(9, 7))
    for idx, name in enumerate(models):
        pipe = fitted_pipelines[name]
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe, "decision_function"):
            y_proba = pipe.decision_function(X_test)
        else:
            y_proba = pipe.predict(X_test)

        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        avg_prec = average_precision_score(y_test, y_proba)
        ax.plot(
            rec,
            prec,
            label=f"{name} (AP = {avg_prec:.3f})",
            color=colors[idx],
            linewidth=2.0
        )

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title("Precision-Recall Benchmark Curves", pad=15)
    ax.legend(loc="lower left", frameon=True, fontsize=10)
    plt.savefig(FIGURES_DIR / "11_precision_recall_curves.png", dpi=200)
    plt.close()

    # 5. Transition Feasibility Matrix (Agro-Chemical Latency Analysis)
    if df_raw is not None:
        df_copy = df_raw.copy()
        bins = [-1, 2, 6, 10, 25]
        labels = ["0-2 Years (Low Latency)", "3-6 Years (Moderate)", "7-10 Years (High)", "11+ Years (Severe)"]
        df_copy["Chemical_Years_Group"] = pd.cut(df_copy["Years_Chemical_Input"], bins=bins, labels=labels)
        
        matrix = df_copy.pivot_table(
            index="Previous_Fertilizer_Usage",
            columns="Chemical_Years_Group",
            values="Organic_Farming_Suitability",
            aggfunc=lambda x: (x == "Suitable").mean() * 100,
            observed=False
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".1f",
            cmap="YlGn",
            cbar_kws={"label": "Suitability Feasibility Rate (%)"},
            linewidths=1.0,
            ax=ax
        )
        ax.set_title("Agro-Chemical Transition Feasibility Matrix (% Suitable by Regimen & Latency)", pad=15)
        ax.set_xlabel("Years of Chemical Input Latency")
        ax.set_ylabel("Previous Fertilizer Regime")
        plt.savefig(FIGURES_DIR / "14_transition_feasibility_matrix.png", dpi=200)
        plt.close()

    print("[Evaluation Engine] Model performance figures generated successfully.")

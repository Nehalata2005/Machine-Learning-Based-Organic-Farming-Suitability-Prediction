import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.config import (
    FIGURES_DIR,
    TARGET_COL,
    NUMERICAL_COLS,
    CATEGORICAL_COLS
)


def set_plotting_style():
    """Sets a refined, academic visual theme for all figures."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "figure.titlesize": 14,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 10,
        "figure.autolayout": True
    })


def generate_all_eda_figures(df: pd.DataFrame):
    """Generates and saves publication-quality exploratory data analysis figures."""
    set_plotting_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    palette = {"Suitable": "#2E7D32", "Not Suitable": "#C62828"}

    print("[EDA Engine] Generating exploratory figures...")

    # 1. Target Class Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df[TARGET_COL].value_counts()
    percentages = (counts / len(df) * 100).round(1)
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[palette[k] for k in counts.index],
        edgecolor="#333333",
        linewidth=1.2,
        width=0.55
    )
    for bar, pct in zip(bars, percentages.values):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 10,
            f"{int(yval)} ({pct}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10.5
        )
    ax.set_ylim(0, max(counts.values) + 80)
    ax.set_title("Target Distribution: Organic Farming Suitability (N=1000)", pad=15)
    ax.set_xlabel("Classification Label")
    ax.set_ylabel("Number of Field Records")
    plt.savefig(FIGURES_DIR / "01_target_class_distribution.png", dpi=200)
    plt.close()

    # 2. Correlation Heatmap for Numerical Features
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df[NUMERICAL_COLS].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.8,
        vmin=-0.8,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75, "label": "Pearson Correlation (r)"},
        ax=ax
    )
    ax.set_title("Inter-Feature Pearson Correlation Heatmap (Edaphic & Climatic Attributes)", pad=15)
    plt.savefig(FIGURES_DIR / "02_soil_nutrients_correlation_heatmap.png", dpi=200)
    plt.close()

    # 3. Organic Carbon % vs Suitability
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(
        x=TARGET_COL,
        y="Organic_Carbon_pct",
        data=df,
        palette=palette,
        hue=TARGET_COL,
        legend=False,
        ax=ax1,
        width=0.45
    )
    ax1.set_title("Organic Carbon (%) by Suitability Class")
    ax1.set_xlabel("Suitability Status")
    ax1.set_ylabel("Organic Carbon (%)")

    sns.kdeplot(
        data=df,
        x="Organic_Carbon_pct",
        hue=TARGET_COL,
        palette=palette,
        common_norm=False,
        fill=True,
        alpha=0.35,
        ax=ax2
    )
    ax2.set_title("Kernel Density of Organic Carbon (%)")
    ax2.set_xlabel("Organic Carbon (%)")
    ax2.set_ylabel("Density")
    plt.savefig(FIGURES_DIR / "03_organic_carbon_vs_suitability.png", dpi=200)
    plt.close()

    # 4. Years of Chemical Input vs Suitability (Transition Barrier)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.violinplot(
        x=TARGET_COL,
        y="Years_Chemical_Input",
        data=df,
        palette=palette,
        hue=TARGET_COL,
        legend=False,
        inner="quartile",
        ax=ax1
    )
    ax1.set_title("Chemical Input Duration Distribution (Transition Burden)")
    ax1.set_xlabel("Suitability Status")
    ax1.set_ylabel("Years of Chemical Input")

    sns.histplot(
        data=df,
        x="Years_Chemical_Input",
        hue=TARGET_COL,
        palette=palette,
        multiple="dodge",
        bins=10,
        shrink=0.8,
        ax=ax2
    )
    ax2.set_title("Frequency by Years of Chemical Input")
    ax2.set_xlabel("Years of Chemical Input")
    ax2.set_ylabel("Count")
    plt.savefig(FIGURES_DIR / "04_chemical_input_years_vs_suitability.png", dpi=200)
    plt.close()

    # 5. Agricultural Practice Impact (Fertilizer and Pesticide Usage)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fert_ct = pd.crosstab(
        df["Previous_Fertilizer_Usage"],
        df[TARGET_COL],
        normalize="index"
    ) * 100
    fert_ct[["Suitable", "Not Suitable"]].plot(
        kind="bar",
        stacked=True,
        color=["#2E7D32", "#C62828"],
        edgecolor="#333",
        ax=ax1,
        rot=0
    )
    ax1.set_title("Suitability Ratio by Previous Fertilizer Usage")
    ax1.set_xlabel("Previous Fertilizer Regimen")
    ax1.set_ylabel("Percentage (%)")
    ax1.legend(title="Status", loc="upper right")

    pest_ct = pd.crosstab(
        df["Previous_Pesticide_Usage"],
        df[TARGET_COL],
        normalize="index"
    ) * 100
    pest_ct[["Suitable", "Not Suitable"]].plot(
        kind="bar",
        stacked=True,
        color=["#2E7D32", "#C62828"],
        edgecolor="#333",
        ax=ax2,
        rot=0
    )
    ax2.set_title("Suitability Ratio by Previous Pesticide Pressure")
    ax2.set_xlabel("Previous Pesticide Usage Level")
    ax2.set_ylabel("Percentage (%)")
    ax2.legend(title="Status", loc="upper right")
    plt.savefig(FIGURES_DIR / "05_pesticide_fertilizer_transition_impact.png", dpi=200)
    plt.close()

    # 6. Soil pH and Electrical Conductivity (EC) Distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(
        data=df,
        x="pH",
        hue=TARGET_COL,
        palette=palette,
        kde=True,
        element="step",
        ax=ax1
    )
    ax1.axvline(6.5, color="gray", linestyle="--", alpha=0.7, label="Neutral Min (6.5)")
    ax1.axvline(7.5, color="gray", linestyle="--", alpha=0.7, label="Neutral Max (7.5)")
    ax1.set_title("Distribution of Soil pH by Suitability")
    ax1.set_xlabel("Soil pH")
    ax1.legend()

    sns.histplot(
        data=df,
        x="EC_dS_m",
        hue=TARGET_COL,
        palette=palette,
        kde=True,
        element="step",
        ax=ax2
    )
    ax2.set_title("Distribution of Electrical Conductivity (EC)")
    ax2.set_xlabel("EC (dS/m)")
    plt.savefig(FIGURES_DIR / "06_soil_ph_and_ec_distributions.png", dpi=200)
    plt.close()

    # 7. Suitability Rates by Crop and Soil Type
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    crop_order = df.groupby("Crop")[TARGET_COL].apply(
        lambda s: (s == "Suitable").mean()
    ).sort_values(ascending=False).index

    sns.barplot(
        x="Crop",
        y=df[TARGET_COL].map({"Suitable": 1, "Not Suitable": 0}),
        data=df,
        order=crop_order,
        palette="Greens_r",
        hue="Crop",
        legend=False,
        ax=ax1
    )
    ax1.set_title("Organic Farming Suitability Rate by Crop Type")
    ax1.set_xlabel("Target Crop")
    ax1.set_ylabel("Proportion Suitable")
    ax1.tick_params(axis="x", rotation=35)

    soil_order = df.groupby("Soil_Type")[TARGET_COL].apply(
        lambda s: (s == "Suitable").mean()
    ).sort_values(ascending=False).index

    sns.barplot(
        x="Soil_Type",
        y=df[TARGET_COL].map({"Suitable": 1, "Not Suitable": 0}),
        data=df,
        order=soil_order,
        palette="Blues_r",
        hue="Soil_Type",
        legend=False,
        ax=ax2
    )
    ax2.set_title("Organic Farming Suitability Rate by Soil Classification")
    ax2.set_xlabel("Soil Type")
    ax2.set_ylabel("Proportion Suitable")
    ax2.tick_params(axis="x", rotation=35)
    plt.savefig(FIGURES_DIR / "07_state_and_crop_suitability_rates.png", dpi=200)
    plt.close()

    print("[EDA Engine] All exploratory charts generated successfully.")

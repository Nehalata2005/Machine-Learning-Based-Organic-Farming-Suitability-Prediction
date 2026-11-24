import json
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import (
    RAW_DATA_PATH,
    TARGET_COL,
    TARGET_MAP,
    EXCLUDED_COLS,
    FEATURE_COLS,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    RANDOM_STATE,
    TEST_SIZE,
    REPORTS_DIR
)


def load_raw_dataset(filepath=RAW_DATA_PATH) -> pd.DataFrame:
    """Loads the source-calibrated organic farming suitability dataset."""
    df = pd.read_csv(filepath)
    df.columns = [col.strip() for col in df.columns]
    return df


def audit_dataset(df: pd.DataFrame) -> dict:
    """Performs integrity checks and audits dataset quality."""
    audit = {
        "total_records": int(len(df)),
        "total_columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values_per_column": df.isnull().sum().to_dict(),
        "has_missing_values": bool(df.isnull().sum().sum() > 0),
        "target_distribution": df[TARGET_COL].value_counts().to_dict(),
        "target_percentages": (
            df[TARGET_COL].value_counts(normalize=True) * 100
        ).round(2).to_dict(),
        "numerical_features_count": len(NUMERICAL_COLS),
        "categorical_features_count": len(CATEGORICAL_COLS),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "data_audit_summary.json", "w") as f:
        json.dump(audit, f, indent=4)

    return audit


def prepare_features_and_target(df: pd.DataFrame):
    """
    Separates features and target while strictly excluding leakage variables.
    Target is mapped to binary integer (0 = Not Suitable, 1 = Suitable).
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].map(TARGET_MAP).astype(int)
    return X, y


def get_stratified_train_test_split(df: pd.DataFrame):
    """
    Splits features and target into stratified train and test sets (80/20).
    Stratification ensures identical class proportions in both sets.
    """
    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    return X_train, X_test, y_train, y_test

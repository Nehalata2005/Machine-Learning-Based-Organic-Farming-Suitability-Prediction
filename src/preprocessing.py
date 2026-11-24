from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from src.config import NUMERICAL_COLS, CATEGORICAL_COLS


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a leakage-free Scikit-Learn ColumnTransformer:
    1. Numerical pipeline: Median imputation + StandardScaler (zero mean, unit variance).
    2. Categorical pipeline: Most-frequent imputation + OneHotEncoder with unseen category handling.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERICAL_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS)
        ],
        remainder="drop"
    )

    return preprocessor


def get_feature_names_after_preprocessing(preprocessor: ColumnTransformer) -> list:
    """Extracts transformed feature names after fitting the preprocessor."""
    feature_names = []
    
    # Numeric features retain original names
    feature_names.extend(NUMERICAL_COLS)
    
    # Categorical features get one-hot encoded names
    cat_transformer = preprocessor.named_transformers_["cat"]
    onehot_step = cat_transformer.named_steps["onehot"]
    cat_names = onehot_step.get_feature_names_out(CATEGORICAL_COLS).tolist()
    feature_names.extend(cat_names)
    
    return feature_names

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

CATEGORICAL_FEATURES = ["cut", "color", "clarity"]
TARGET = "outcome"


def get_preprocessor(feature_names):
    """Return a ColumnTransformer for the given feature names.

    Categorical features are one-hot encoded; all remaining features are
    standard-scaled.
    """
    numeric_features = [f for f in feature_names
                        if f not in CATEGORICAL_FEATURES and f != TARGET]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

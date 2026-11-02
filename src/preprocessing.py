from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["ano", "x", "y"]
CATEGORICAL_FEATURES = ["mes", "dia_da_sem", "uc"]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

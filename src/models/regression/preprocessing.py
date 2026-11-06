from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["ano", "mes_num", "x_cell", "y_cell"]
CATEGORICAL_FEATURES = ["mes", "estacao", "periodo_chuva", "uc"]


def build_preprocessor():
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

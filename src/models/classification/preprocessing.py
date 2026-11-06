from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["ano", "mes_num", "dia_ano", "fim_de_semana", "x", "y"]
CATEGORICAL_FEATURES = ["mes", "dia_da_sem", "estacao", "periodo_chuva", "uc"]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

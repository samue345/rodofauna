import pandas as pd

from src.features.feature_engineering import add_temporal_features


CLASSIFICATION_FEATURES = [
    "ano",
    "mes_num",
    "dia_ano",
    "fim_de_semana",
    "dia_da_sem",
    "estacao",
    "periodo_chuva",
    "uc",
    "x",
    "y",
]


def classification_dataset(df: pd.DataFrame):
    data = df.loc[df["classe"].ne("nao_identificado")].copy()
    data = add_temporal_features(data)
    X = data[CLASSIFICATION_FEATURES].copy()
    y = data["classe"].copy()
    return X, y

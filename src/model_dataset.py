import pandas as pd


CLASSIFICATION_FEATURES = [
    "ano",
    "mes",
    "dia_da_sem",
    "uc",
    "x",
    "y",
]


def classification_dataset(df: pd.DataFrame):
    data = df.loc[df["classe"].ne("nao_identificado")].copy()
    X = data[CLASSIFICATION_FEATURES].copy()
    y = data["classe"].copy()
    return X, y

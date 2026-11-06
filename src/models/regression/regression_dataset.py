import pandas as pd

from src.data.data_preparation import filter_wildlife, load_raw_data
from src.features.feature_engineering import add_temporal_features


GRID_SIZE_METERS = 500


def load_regression_dataset():
    """Aggregate wildlife records by 500 m spatial cell and month."""
    wildlife = add_temporal_features(filter_wildlife(load_raw_data()))
    wildlife["x_cell"] = (wildlife["x"] // GRID_SIZE_METERS).astype(int)
    wildlife["y_cell"] = (wildlife["y"] // GRID_SIZE_METERS).astype(int)

    grouped = (
        wildlife.groupby(
            ["ano", "mes_num", "mes", "x_cell", "y_cell"],
            as_index=False,
        )
        .agg(
            atropelamentos=("id", "size"),
            uc=("uc", lambda values: values.mode().iat[0]),
            estacao=("estacao", "first"),
            periodo_chuva=("periodo_chuva", "first"),
        )
    )
    return grouped


def regression_dataset():
    data = load_regression_dataset()
    features = [
        "ano",
        "mes_num",
        "mes",
        "x_cell",
        "y_cell",
        "estacao",
        "periodo_chuva",
        "uc",
    ]
    return data[features], data["atropelamentos"]

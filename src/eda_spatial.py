import pandas as pd


def coordinate_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "atributo": ["x", "y"],
            "minimo": [df["x"].min(), df["y"].min()],
            "maximo": [df["x"].max(), df["y"].max()],
            "valores_unicos": [df["x"].nunique(), df["y"].nunique()],
        }
    )


def grid_counts(df: pd.DataFrame, cell_size: int = 500) -> pd.DataFrame:
    result = df[["x", "y"]].copy()
    result["x_celula"] = (result["x"] // cell_size) * cell_size
    result["y_celula"] = (result["y"] // cell_size) * cell_size
    return (
        result.groupby(["x_celula", "y_celula"], as_index=False)
        .size()
        .rename(columns={"size": "quantidade"})
        .sort_values("quantidade", ascending=False)
        .reset_index(drop=True)
    )

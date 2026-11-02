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
    result["x_indice"] = result["x"] // cell_size
    result["y_indice"] = result["y"] // cell_size
    return (
        result.groupby(["x_indice", "y_indice"], as_index=False)
        .size()
        .rename(columns={"size": "quantidade"})
        .sort_values("quantidade", ascending=False)
    )


def plot_spatial(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["x"], df["y"], s=5, alpha=0.35)
    ax.set_title("Distribuição espacial das ocorrências")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    return fig, ax

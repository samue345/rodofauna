import pandas as pd


def inventory(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "atributo": df.columns,
            "tipo": [str(dtype) for dtype in df.dtypes],
            "registros": len(df),
            "valores_preenchidos": [df[column].notna().sum() for column in df.columns],
            "valores_unicos": [df[column].nunique(dropna=True) for column in df.columns],
        }
    )


def quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "atributo": df.columns,
            "valores_ausentes": [df[column].isna().sum() for column in df.columns],
            "percentual_ausente": [df[column].isna().mean() * 100 for column in df.columns],
            "valores_nao_identificados": [
                df[column].astype("string").str.lower().eq("nao_identificado").sum()
                for column in df.columns
            ],
        }
    )


def duplicate_count(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def count_by_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        df[column]
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="quantidade")
    )


def monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    months = [
        "janeiro", "fevereiro", "marco", "abril", "maio", "junho",
        "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
    ]
    result = count_by_column(df, "mes")
    result["mes"] = pd.Categorical(result["mes"], categories=months, ordered=True)
    return result.sort_values("mes").reset_index(drop=True)


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
        .reset_index(drop=True)
    )


def plot_counts(summary: pd.DataFrame, category: str, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(summary[category].astype(str), summary["quantidade"])
    ax.set_title(title)
    ax.set_xlabel(category)
    ax.set_ylabel("quantidade")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig, ax


def plot_spatial(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["x"], df["y"], s=5, alpha=0.35)
    ax.set_title("Distribuição espacial das ocorrências")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    return fig, ax

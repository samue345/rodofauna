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
    result["x_celula"] = (result["x"] // cell_size) * cell_size
    result["y_celula"] = (result["y"] // cell_size) * cell_size
    return (
        result.groupby(["x_celula", "y_celula"], as_index=False)
        .size()
        .rename(columns={"size": "quantidade"})
        .sort_values("quantidade", ascending=False)
        .reset_index(drop=True)
    )

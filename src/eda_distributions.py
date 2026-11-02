import pandas as pd


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

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

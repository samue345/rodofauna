import pandas as pd


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

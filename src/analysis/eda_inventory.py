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

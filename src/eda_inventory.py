from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "ProgramaRodofaunaAtropelada.xlsx"


def load_data(path: Path = RAW_FILE) -> pd.DataFrame:
    return pd.read_excel(path)


def select_wildlife(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["silvestre_"].eq("silvestre")].copy()


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


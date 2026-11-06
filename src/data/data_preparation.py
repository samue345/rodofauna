from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = ROOT / "data" / "raw" / "ProgramaRodofaunaAtropelada.xlsx"
TECHNICAL_COLUMNS = ["OBJECTID"]


def load_raw_data(path: Path = RAW_FILE) -> pd.DataFrame:
    data = pd.read_excel(path)
    return clean_raw_data(data)


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove GIS technical fields and duplicated exported occurrences."""
    data = df.drop(columns=TECHNICAL_COLUMNS, errors="ignore").copy()
    return data.drop_duplicates().reset_index(drop=True)


def filter_wildlife(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["silvestre_"].eq("silvestre")].copy()


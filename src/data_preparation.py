from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "ProgramaRodofaunaAtropelada.xlsx"


def load_raw_data(path: Path = RAW_FILE) -> pd.DataFrame:
    return pd.read_excel(path)


def filter_wildlife(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["silvestre_"].eq("silvestre")].copy()


if __name__ == "__main__":
    data = filter_wildlife(load_raw_data())
    print(f"Registros silvestres: {len(data)}")

"""Inventário inicial da base do RodoFauna."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "ProgramaRodofaunaAtropelada.xlsx"


def load_data(path: Path = RAW_FILE) -> pd.DataFrame:
    """Carrega a planilha original sem modificar o arquivo de origem."""
    return pd.read_excel(path)


def select_wildlife(df: pd.DataFrame) -> pd.DataFrame:
    """Seleciona somente registros classificados como fauna silvestre."""
    return df.loc[df["silvestre_"].eq("silvestre")].copy()


def inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna um inventário simples dos campos e seus tipos."""
    return pd.DataFrame(
        {
            "atributo": df.columns,
            "tipo": [str(dtype) for dtype in df.dtypes],
            "registros": len(df),
            "valores_preenchidos": [df[column].notna().sum() for column in df.columns],
            "valores_unicos": [df[column].nunique(dropna=True) for column in df.columns],
        }
    )


if __name__ == "__main__":
    original = load_data()
    wildlife = select_wildlife(original)

    print(f"Registros totais: {len(original):,}".replace(",", "."))
    print(f"Atributos: {original.shape[1]}")
    print(f"Registros silvestres: {len(wildlife):,}".replace(",", "."))
    print(f"Registros domésticos: {len(original) - len(wildlife):,}".replace(",", "."))
    print("\nInventário:")
    print(inventory(wildlife).to_string(index=False))

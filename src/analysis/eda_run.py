"""Generate the exploratory-analysis tables and figures."""

from pathlib import Path

import matplotlib.pyplot as plt

from src.analysis.data_quality import duplicate_count, quality_summary
from src.analysis.eda_distributions import count_by_column, monthly_counts, plot_counts
from src.analysis.eda_inventory import inventory
from src.analysis.eda_spatial import coordinate_summary, grid_counts, plot_spatial
from src.data.data_preparation import filter_wildlife, load_raw_data


ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "figures"
REPORTS_DIR = ROOT / "reports" / "generated"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_raw_data()
    wildlife = filter_wildlife(data)

    print(f"Registros totais: {len(data):,}".replace(",", "."))
    print(f"Registros silvestres: {len(wildlife):,}".replace(",", "."))
    print(f"Registros duplicados: {duplicate_count(wildlife)}")
    print("\nInventário:")
    print(inventory(wildlife).to_string(index=False))
    print("\nQualidade:")
    print(quality_summary(wildlife).to_string(index=False))
    print("\nTop 20 células espaciais:")
    print(grid_counts(wildlife, cell_size=500).head(20).to_string(index=False))
    print("\nResumo das coordenadas:")
    print(coordinate_summary(wildlife).to_string(index=False))

    _save_count_plot(
        count_by_column(wildlife, "classe"),
        "classe",
        "Ocorrências por classe",
        "ocorrencias_por_classe.png",
    )
    _save_count_plot(
        monthly_counts(wildlife),
        "mes",
        "Ocorrências por mês",
        "ocorrencias_por_mes.png",
    )
    _save_count_plot(
        count_by_column(wildlife, "uc"),
        "uc",
        "Ocorrências por unidade de conservação",
        "ocorrencias_por_uc.png",
    )
    spatial_figure, _ = plot_spatial(wildlife)
    spatial_figure.savefig(FIGURES_DIR / "distribuicao_espacial.png", dpi=160)
    plt.close(spatial_figure)
    print(f"\nGráficos salvos em: {FIGURES_DIR}")


def _save_count_plot(summary, category, title, filename):
    figure, _ = plot_counts(summary, category, title)
    figure.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()

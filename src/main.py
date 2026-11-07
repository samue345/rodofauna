from src.analysis.eda_inventory import inventory
from src.data.data_preparation import filter_wildlife, load_raw_data
from src.models.classification.classification_run import run_classification_phase
from src.models.regression.regression_run import main as run_regression
from src.models.optimization.optimization_run import main as run_optimization
from src.models.regularization.regularization_run import main as run_regularization


def run_inventory() -> None:
    original = load_raw_data()
    wildlife = filter_wildlife(original)

    print(f"Registros totais: {len(original):,}".replace(",", "."))
    print(f"Atributos: {original.shape[1]}")
    print(f"Registros silvestres: {len(wildlife):,}".replace(",", "."))
    print(f"Registros domésticos: {len(original) - len(wildlife):,}".replace(",", "."))
    print("\nInventário:")
    print(inventory(wildlife).to_string(index=False))


def main() -> None:
    run_inventory()
    run_classification_phase()
    run_regression()
    optimization_result = run_optimization()
    run_regularization(best_params=optimization_result["best_params"])
    print("\nExecução concluída com sucesso.")


if __name__ == "__main__":
    main()

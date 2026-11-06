import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.models.optimization.classification_optuna import optimize_classification_mlp


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "reports" / "generated"


def main(n_trials=50):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = optimize_classification_mlp(n_trials=n_trials)
    save_results(result)
    save_history_plot(result["study"])
    print_results(result)


def save_results(result):
    output = {
        "best_params": result["best_params"],
        "best_value": result["best_value"],
        "validation": _json_metrics(result["validation"]),
        "test": _json_metrics(result["test"]),
    }
    with (OUTPUT_DIR / "classification_phase5_optuna.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(output, file, ensure_ascii=False, indent=2)


def save_history_plot(study):
    trials = [trial for trial in study.trials if trial.value is not None]
    values = [trial.value for trial in trials]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(range(1, len(values) + 1), values, marker="o")
    axis.set(
        title="Histórico da otimização Optuna",
        xlabel="Trial",
        ylabel="F1 Macro (validação)",
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "classification_phase5_optuna_history.png", dpi=160)


def print_results(result):
    print("optuna_best_params", result["best_params"])
    print("optuna_best_validation_f1", result["best_value"])
    print("optuna_validation", result["validation"])
    print("optuna_test", result["test"])


def _json_metrics(metrics):
    return {
        key: value.tolist() if hasattr(value, "tolist") else value
        for key, value in metrics.items()
    }

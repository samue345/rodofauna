import json
from pathlib import Path

from src.models.classification.classification_experiment import (
    run_baselines,
    run_mlp_experiment,
)
from src.models.classification.classification_evaluation import plot_mlp_history
from src.models.classification.classification_pipeline import prepare_classification_data


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "reports" / "generated"


def run_classification_phase():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare_classification_data()
    baseline_results = run_baselines(data)
    mlp_result = run_mlp_experiment(data)
    print_results(baseline_results, mlp_result)

    summary = build_summary(baseline_results, mlp_result)
    save_metrics(summary)
    save_mlp_curve(mlp_result)
    save_markdown_summary(summary)


def print_results(baseline_results, mlp_result):
    """Print the validation and test metrics for every model."""
    for name, metrics in baseline_results.items():
        print(name)
        print("validation", metrics["validation"])
        print("test", metrics["test"])

    print("mlp_classifier")
    print("validation", mlp_result["validation"])
    print("test", mlp_result["test"])
    print("epochs", len(mlp_result["history"]["epoch"]))


def build_summary(baseline_results, mlp_result):
    """Convert experiment results into the structure used by the reports."""
    summary = {
        name: {
            "validation": _json_metrics(metrics["validation"]),
            "test": _json_metrics(metrics["test"]),
        }
        for name, metrics in baseline_results.items()
    }
    summary["mlp_classifier"] = {
        "validation": _json_metrics(mlp_result["validation"]),
        "test": _json_metrics(mlp_result["test"]),
        "epochs": len(mlp_result["history"]["epoch"]),
    }
    return summary


def save_metrics(summary):
    """Save all model metrics as JSON."""
    output_file = OUTPUT_DIR / "classification_phase3_metrics.json"
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def save_mlp_curve(mlp_result):
    """Save the MLP training curves as a PNG image."""
    figure = plot_mlp_history(mlp_result["history"])
    figure.savefig(OUTPUT_DIR / "mlp_training_curves.png", dpi=160)


def save_markdown_summary(summary):
    """Save a compact Markdown comparison table."""
    output_file = OUTPUT_DIR / "classification_phase3_summary.md"
    with output_file.open("w", encoding="utf-8") as file:
        file.write("# Fase 3 - Classificação com MLP\n\n")
        file.write(
            "Os três modelos usam a mesma divisão estratificada de treino, "
            "validação e teste.\n\n"
        )
        file.write(
            "| Modelo | F1 Macro (validação) | F1 Macro (teste) | "
            "Acurácia (teste) |\n"
        )
        file.write("|---|---:|---:|---:|\n")
        for name, metrics in summary.items():
            file.write(format_summary_row(name, metrics))


def format_summary_row(name, metrics):
    """Format one model as a Markdown table row."""
    return (
        f"| {name} | {metrics['validation']['f1_macro']:.4f} | "
        f"{metrics['test']['f1_macro']:.4f} | "
        f"{metrics['test']['accuracy']:.4f} |\n"
    )


def _json_metrics(metrics):
    return {
        key: value.tolist() if hasattr(value, "tolist") else value
        for key, value in metrics.items()
    }



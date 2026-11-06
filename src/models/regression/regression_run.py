from src.models.regression.regression_evaluation import plot_mlp_history, plot_predictions
from src.models.regression.regression_experiment import (
    run_mlp_regression,
    run_regression_baselines,
)


def main():
    baseline_results = run_regression_baselines()
    mlp_result = run_mlp_regression()
    print_results(baseline_results, mlp_result)
    plot_mlp_history(mlp_result["history"]).savefig(
        "reports/generated/mlp_regression_training_curves.png", dpi=160
    )
    plot_predictions(mlp_result["y_test"], mlp_result["test_predictions"]).savefig(
        "reports/generated/mlp_regression_predictions.png", dpi=160
    )


def print_results(baseline_results, mlp_result):
    for name, result in baseline_results.items():
        print(name)
        print("validation", format_metrics(result["validation"]))
        print("test", format_metrics(result["test"]))
    print("mlp_regressor")
    print("validation", format_metrics(mlp_result["validation"]))
    print("test", format_metrics(mlp_result["test"]))


def format_metrics(metrics):
    return {
        "mae": round(metrics["mae"], 4),
        "rmse": round(metrics["rmse"], 4),
        "r2": round(metrics["r2"], 4),
    }

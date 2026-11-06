import json
from pathlib import Path

from sklearn.neural_network import MLPClassifier

from src.models.classification.classification_evaluation import (
    classification_metrics,
    plot_mlp_history,
)
from src.models.classification.classification_pipeline import (
    build_classification_pipeline,
    prepare_classification_data,
)
from src.models.classification.classification_training import (
    fit_mlp_classifier,
    predict_classifier,
)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "reports" / "generated"


def compare_regularization(epochs=100):
    data = prepare_classification_data()
    X_train, X_validation, X_test, y_train, y_validation, y_test = data
    configurations = {
        "without_regularization": 0.0,
        "with_l2_regularization": 0.001,
    }
    results = {}

    for name, alpha in configurations.items():
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=0.001,
            batch_size=32,
            max_iter=1,
            random_state=42,
        )
        pipeline = build_classification_pipeline(model)
        pipeline, history = fit_mlp_classifier(
            pipeline,
            X_train,
            y_train,
            X_validation,
            y_validation,
            epochs=epochs,
        )
        results[name] = {
            "alpha": alpha,
            "validation": classification_metrics(
                y_validation, predict_classifier(pipeline, X_validation)
            ),
            "test": classification_metrics(
                y_test, predict_classifier(pipeline, X_test)
            ),
            "history": history,
        }

    save_results(results)
    return results


def save_results(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, result in results.items():
        summary[name] = {
            "alpha": result["alpha"],
            "validation": _json_metrics(result["validation"]),
            "test": _json_metrics(result["test"]),
        }
        plot_mlp_history(result["history"]).savefig(
            OUTPUT_DIR / f"{name}_training_curves.png", dpi=160
        )

    with (OUTPUT_DIR / "classification_phase6_regularization.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def _json_metrics(metrics):
    return {
        key: value.tolist() if hasattr(value, "tolist") else value
        for key, value in metrics.items()
    }

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


def compare_regularization(epochs=100, best_params=None):
    """Compare one unregularized MLP with several L2 configurations."""
    data = prepare_classification_data()
    base_parameters = build_base_parameters(best_params)
    without_regularization = train_configuration(
        alpha=0.0,
        base_parameters=base_parameters,
        data=data,
        epochs=epochs,
    )
    l2_candidates = train_l2_candidates(
        best_params=best_params,
        base_parameters=base_parameters,
        data=data,
        epochs=epochs,
    )
    best_l2 = select_best_l2(l2_candidates)
    best_l2["alpha_candidates"] = validation_scores(l2_candidates)

    results = {
        "without_regularization": without_regularization,
        "with_l2_regularization": best_l2,
    }

    save_results(results)
    return results


def train_l2_candidates(best_params, base_parameters, data, epochs):
    """Train one MLP for each candidate L2 strength."""
    return {
        alpha: train_configuration(
            alpha=alpha,
            base_parameters=base_parameters,
            data=data,
            epochs=epochs,
        )
        for alpha in build_alpha_candidates(best_params)
    }


def select_best_l2(candidates):
    """Select the L2 result with the highest validation F1 Macro."""
    return max(
        candidates.values(),
        key=lambda result: result["validation"]["f1_macro"],
    )


def validation_scores(candidates):
    """Return the validation F1 for every tested alpha."""
    return {
        str(alpha): result["validation"]["f1_macro"]
        for alpha, result in candidates.items()
    }


def train_configuration(alpha, base_parameters, data, epochs):
    """Train one MLP configuration using the same partial-fit protocol."""
    X_train, X_validation, X_test, y_train, y_validation, y_test = data
    parameters = {**base_parameters, "alpha": alpha}
    model = MLPClassifier(
        hidden_layer_sizes=parameters["hidden_layer_sizes"],
        activation=parameters["activation"],
        solver="adam",
        alpha=alpha,
        learning_rate_init=parameters["learning_rate_init"],
        batch_size=parameters["batch_size"],
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
    return {
        "alpha": alpha,
        "model_parameters": parameters,
        "validation": classification_metrics(
            y_validation, predict_classifier(pipeline, X_validation)
        ),
        "test": classification_metrics(y_test, predict_classifier(pipeline, X_test)),
        "history": history,
    }


def build_base_parameters(best_params=None):
    """Return architecture and optimizer settings shared by both variants."""
    if best_params is None:
        return {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "learning_rate_init": 0.001,
            "batch_size": 32,
        }
    return {
        "hidden_layer_sizes": tuple(
            best_params[f"units_layer_{index}"]
            for index in range(best_params["n_layers"])
        ),
        "activation": best_params["activation"],
        "learning_rate_init": best_params["learning_rate_init"],
        "batch_size": best_params["batch_size"],
    }


def build_alpha_candidates(best_params=None):
    """Return L2 values to compare under the partial-fit training protocol."""
    optimized_alpha = best_params["alpha"] if best_params else 0.001
    candidates = {1e-5, 1e-4, 1e-3, 1e-2, optimized_alpha}
    return sorted(candidates)


def save_results(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, result in results.items():
        summary[name] = {
            "alpha": result["alpha"],
            "model_parameters": _json_metrics(result["model_parameters"]),
            "alpha_candidates": result.get("alpha_candidates", {}),
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
        key: (
            value.tolist()
            if hasattr(value, "tolist")
            else list(value)
            if isinstance(value, tuple)
            else value
        )
        for key, value in metrics.items()
    }

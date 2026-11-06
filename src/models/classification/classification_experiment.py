from src.models.classification.classification_evaluation import classification_metrics
from src.models.classification.classification_models import (
    build_logistic_regression,
    build_mlp_classifier,
    build_random_forest,
)
from src.models.classification.classification_pipeline import build_classification_pipeline
from src.models.classification.classification_training import (
    fit_classifier,
    fit_mlp_classifier,
    predict_classifier,
)


def run_baselines(data):
    X_train, X_validation, X_test, y_train, y_validation, y_test = data
    models = {
        "logistic_regression": build_logistic_regression(),
        "random_forest": build_random_forest(),
    }
    results = {}

    for name, model in models.items():
        pipeline = build_classification_pipeline(model)
        fit_classifier(pipeline, X_train, y_train)
        validation_predictions = predict_classifier(pipeline, X_validation)
        test_predictions = predict_classifier(pipeline, X_test)
        results[name] = {
            "validation": classification_metrics(y_validation, validation_predictions),
            "test": classification_metrics(y_test, test_predictions),
        }

    return results


def run_mlp_experiment(data, epochs=100):
    """Train and evaluate the phase-3 MLP on the shared data split."""
    X_train, X_validation, X_test, y_train, y_validation, y_test = data
    pipeline = build_classification_pipeline(build_mlp_classifier())
    pipeline, history = fit_mlp_classifier(
        pipeline,
        X_train,
        y_train,
        X_validation,
        y_validation,
        epochs=epochs,
    )
    return {
        "validation": classification_metrics(y_validation, predict_classifier(pipeline, X_validation)),
        "test": classification_metrics(y_test, predict_classifier(pipeline, X_test)),
        "history": history,
        "pipeline": pipeline,
    }

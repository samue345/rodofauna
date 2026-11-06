from src.models.regression.regression_evaluation import regression_metrics
from src.models.regression.regression_models import (
    build_linear_regression,
    build_mlp_regressor,
    build_random_forest_regressor,
)
from src.models.regression.regression_pipeline import (
    build_regression_pipeline,
    prepare_regression_data,
)
from src.models.regression.regression_training import (
    fit_mlp_regressor,
    fit_regressor,
    predict_regressor,
)


def run_regression_baselines():
    data = prepare_regression_data()
    X_train, X_validation, X_test, y_train, y_validation, y_test = data
    models = {
        "linear_regression": build_linear_regression(),
        "random_forest_regressor": build_random_forest_regressor(),
    }
    results = {}
    for name, model in models.items():
        pipeline = build_regression_pipeline(model)
        fit_regressor(pipeline, X_train, y_train)
        results[name] = {
            "validation": regression_metrics(y_validation, predict_regressor(pipeline, X_validation)),
            "test": regression_metrics(y_test, predict_regressor(pipeline, X_test)),
        }
    return results


def run_mlp_regression(epochs=100):
    data = prepare_regression_data()
    X_train, X_validation, X_test, y_train, y_validation, y_test = data
    pipeline = build_regression_pipeline(build_mlp_regressor())
    pipeline, history = fit_mlp_regressor(
        pipeline, X_train, y_train, X_validation, y_validation, epochs=epochs
    )
    return {
        "validation": regression_metrics(y_validation, predict_regressor(pipeline, X_validation)),
        "test": regression_metrics(y_test, predict_regressor(pipeline, X_test)),
        "test_predictions": predict_regressor(pipeline, X_test),
        "y_test": y_test,
        "history": history,
    }

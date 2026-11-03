from src.models.classification_evaluation import classification_metrics
from src.models.classification_models import build_logistic_regression, build_random_forest
from src.models.classification_pipeline import build_classification_pipeline, prepare_classification_data
from src.models.classification_training import fit_classifier, predict_classifier


def run_baselines():
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_classification_data()
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

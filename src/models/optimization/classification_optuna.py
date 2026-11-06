import optuna
from sklearn.neural_network import MLPClassifier

from src.models.classification.classification_evaluation import classification_metrics
from src.models.classification.classification_pipeline import (
    build_classification_pipeline,
    prepare_classification_data,
)
from src.models.classification.classification_training import predict_classifier


def optimize_classification_mlp(n_trials=15):
    data = prepare_classification_data()
    X_train, X_validation, X_test, y_train, y_validation, y_test = data

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_layers = tuple(
            trial.suggest_int(f"units_layer_{index}", 32, 128, step=32)
            for index in range(n_layers)
        )
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=trial.suggest_categorical("activation", ["relu", "tanh"]),
            solver="adam",
            alpha=trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            learning_rate_init=trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-2, log=True
            ),
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=42,
        )
        pipeline = build_classification_pipeline(model)
        pipeline.fit(X_train, y_train)
        predictions = predict_classifier(pipeline, X_validation)
        return classification_metrics(y_validation, predictions)["f1_macro"]

    study = optuna.create_study(
        direction="maximize",
        study_name="classification_mlp",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_model = MLPClassifier(
        hidden_layer_sizes=tuple(
            study.best_params[f"units_layer_{index}"]
            for index in range(study.best_params["n_layers"])
        ),
        activation=study.best_params["activation"],
        solver="adam",
        alpha=study.best_params["alpha"],
        learning_rate_init=study.best_params["learning_rate_init"],
        batch_size=study.best_params["batch_size"],
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=42,
    )
    best_pipeline = build_classification_pipeline(best_model)
    best_pipeline.fit(X_train, y_train)

    return {
        "study": study,
        "validation": classification_metrics(
            y_validation, predict_classifier(best_pipeline, X_validation)
        ),
        "test": classification_metrics(y_test, predict_classifier(best_pipeline, X_test)),
        "best_params": study.best_params,
        "best_value": study.best_value,
    }

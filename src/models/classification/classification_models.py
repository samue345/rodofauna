from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def build_logistic_regression():
    return LogisticRegression(max_iter=1000)


def build_random_forest():
    return RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )


def build_mlp_classifier():
    """Build the baseline MLP used in phase 3.

    One epoch is executed at a time by ``fit_mlp_classifier`` so that the
    training and validation curves can be recorded without using the test set.
    """
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        batch_size=32,
        max_iter=1,
        shuffle=True,
        random_state=42,
    )

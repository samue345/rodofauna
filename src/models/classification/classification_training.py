from sklearn.metrics import accuracy_score, log_loss


def fit_classifier(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_classifier(pipeline, X):
    return pipeline.predict(X)


def fit_mlp_classifier(pipeline, X_train, y_train, X_validation, y_validation, epochs=100):
    """Train an MLP one epoch at a time and return its history.

    The validation set is used only to monitor the learning curves. It is
    never used to update the weights, and the test set remains untouched.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_validation_encoded = preprocessor.transform(X_validation)

    classes = sorted(y_train.unique())
    history = {
        "epoch": [],
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
    }

    for epoch in range(epochs):
        model.partial_fit(X_train_encoded, y_train, classes=classes)
        train_probabilities = model.predict_proba(X_train_encoded)
        validation_probabilities = model.predict_proba(X_validation_encoded)
        train_predictions = model.classes_[train_probabilities.argmax(axis=1)]
        validation_predictions = model.classes_[validation_probabilities.argmax(axis=1)]

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(log_loss(y_train, train_probabilities, labels=classes))
        history["validation_loss"].append(
            log_loss(y_validation, validation_probabilities, labels=classes)
        )
        history["train_accuracy"].append(accuracy_score(y_train, train_predictions))
        history["validation_accuracy"].append(
            accuracy_score(y_validation, validation_predictions)
        )

    return pipeline, history

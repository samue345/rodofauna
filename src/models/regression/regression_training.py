from sklearn.metrics import mean_absolute_error, mean_squared_error


def fit_regressor(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_regressor(pipeline, X):
    return pipeline.predict(X)


def fit_mlp_regressor(pipeline, X_train, y_train, X_validation, y_validation, epochs=100):
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_validation_encoded = preprocessor.transform(X_validation)
    history = {
        "epoch": [],
        "train_loss": [],
        "validation_loss": [],
        "train_mae": [],
        "validation_mae": [],
    }

    for epoch in range(epochs):
        model.partial_fit(X_train_encoded, y_train)
        train_predictions = model.predict(X_train_encoded)
        validation_predictions = model.predict(X_validation_encoded)
        train_mse = mean_squared_error(y_train, train_predictions)
        validation_mse = mean_squared_error(y_validation, validation_predictions)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_mse)
        history["validation_loss"].append(validation_mse)
        history["train_mae"].append(mean_absolute_error(y_train, train_predictions))
        history["validation_mae"].append(mean_absolute_error(y_validation, validation_predictions))

    return pipeline, history

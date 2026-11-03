def fit_classifier(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_classifier(pipeline, X):
    return pipeline.predict(X)

from sklearn.model_selection import train_test_split


def split_regression_data(X, y, test_size=0.15, validation_size=0.15, random_state=42):
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    validation_relative_size = validation_size / (1 - test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=validation_relative_size,
        random_state=random_state,
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test

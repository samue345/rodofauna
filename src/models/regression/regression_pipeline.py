from sklearn.pipeline import Pipeline

from src.models.regression.preprocessing import build_preprocessor
from src.models.regression.regression_dataset import regression_dataset
from src.models.regression.regression_split import split_regression_data


def prepare_regression_data():
    X, y = regression_dataset()
    return split_regression_data(X, y)


def build_regression_pipeline(model):
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )

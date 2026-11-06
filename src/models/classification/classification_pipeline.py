from sklearn.pipeline import Pipeline

from src.data.data_preparation import filter_wildlife, load_raw_data
from src.data.data_split import split_data
from src.data.model_dataset import classification_dataset
from src.models.classification.preprocessing import build_preprocessor


def load_classification_data():
    wildlife = filter_wildlife(load_raw_data())
    return classification_dataset(wildlife)


def prepare_classification_data():
    X, y = load_classification_data()
    return split_data(X, y)


def build_classification_pipeline(model):
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )

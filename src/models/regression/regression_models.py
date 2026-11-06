from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


def build_linear_regression():
    return LinearRegression()


def build_random_forest_regressor():
    return RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )


def build_mlp_regressor():
    return MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        batch_size=32,
        max_iter=1,
        random_state=42,
    )

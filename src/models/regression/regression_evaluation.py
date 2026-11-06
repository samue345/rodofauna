import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": mse**0.5,
        "r2": r2_score(y_true, y_pred),
    }


def plot_mlp_history(history):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = history["epoch"]
    axes[0].plot(epochs, history["train_loss"], label="treino")
    axes[0].plot(epochs, history["validation_loss"], label="validação")
    axes[0].set(title="Loss por época", xlabel="Época", ylabel="MSE")
    axes[0].legend()
    axes[1].plot(epochs, history["train_mae"], label="treino")
    axes[1].plot(epochs, history["validation_mae"], label="validação")
    axes[1].set(title="MAE por época", xlabel="Época", ylabel="MAE")
    axes[1].legend()
    figure.tight_layout()
    return figure


def plot_predictions(y_true, y_pred):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_true, y_pred, alpha=0.6)
    axes[0].set(title="Valores reais x preditos", xlabel="Real", ylabel="Predito")
    axes[1].scatter(y_pred, y_true - y_pred, alpha=0.6)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(title="Resíduos", xlabel="Predito", ylabel="Real - predito")
    figure.tight_layout()
    return figure

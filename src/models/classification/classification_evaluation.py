from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def plot_mlp_history(history):
    """Return the phase-3 training curves as a matplotlib figure."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = history["epoch"]

    axes[0].plot(epochs, history["train_loss"], label="treino")
    axes[0].plot(epochs, history["validation_loss"], label="validação")
    axes[0].set(title="Loss por época", xlabel="Época", ylabel="Log loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="treino")
    axes[1].plot(epochs, history["validation_accuracy"], label="validação")
    axes[1].set(title="Acurácia por época", xlabel="Época", ylabel="Acurácia")
    axes[1].legend()

    figure.tight_layout()
    return figure

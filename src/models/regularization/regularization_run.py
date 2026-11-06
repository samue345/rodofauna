from src.models.regularization.classification_regularization import compare_regularization


def main():
    results = compare_regularization()
    for name, result in results.items():
        print(name)
        print("validation", format_metrics(result["validation"]))
        print("test", format_metrics(result["test"]))


def format_metrics(metrics):
    return {
        "f1_macro": round(metrics["f1_macro"], 4),
        "accuracy": round(metrics["accuracy"], 4),
    }

from src.models.regularization.classification_regularization import compare_regularization


def main():
    results = compare_regularization()
    for name, result in results.items():
        print(name)
        print("validation", result["validation"])
        print("test", result["test"])

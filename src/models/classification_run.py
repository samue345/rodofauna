from src.models.classification_experiment import run_baselines


def main():
    results = run_baselines()
    for name, metrics in results.items():
        print(name)
        print("validation", metrics["validation"])
        print("test", metrics["test"])


if __name__ == "__main__":
    main()

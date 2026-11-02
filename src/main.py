from data_preparation import filter_wildlife, load_raw_data
from eda_inventory import inventory


def main() -> None:
    original = load_raw_data()
    wildlife = filter_wildlife(original)

    print(f"Registros totais: {len(original):,}".replace(",", "."))
    print(f"Atributos: {original.shape[1]}")
    print(f"Registros silvestres: {len(wildlife):,}".replace(",", "."))
    print(f"Registros domésticos: {len(original) - len(wildlife):,}".replace(",", "."))
    print("\nInventário:")
    print(inventory(wildlife).to_string(index=False))


if __name__ == "__main__":
    main()

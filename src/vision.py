from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("NDVI scaffold ready.")
    print("Next step: authenticate Earth Engine and export annual county NDVI values.")


if __name__ == "__main__":
    main()

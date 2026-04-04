from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODEL_DIR = PROJECT_ROOT / "models"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("Model scaffold ready.")
    print(f"Expected feature store: {FEATURES_PATH}")
    print("Next step: train XGBoost, log MLflow metrics, and save SHAP output.")


if __name__ == "__main__":
    main()

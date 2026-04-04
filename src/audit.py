from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_REPORT_PATH = PROJECT_ROOT / "audit_report.md"


def main() -> None:
    print(f"Audit scaffold ready. Planned output: {AUDIT_REPORT_PATH}")
    print("Next step: compute split metrics and write the final markdown report.")


if __name__ == "__main__":
    main()

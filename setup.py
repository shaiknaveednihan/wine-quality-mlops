from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED_FILES = (
    "requirements.txt",
    "requirements-api.txt",
    "data/raw/winequality-red.csv",
    "src/preprocess.py",
    "src/train.py",
    "src/evaluate.py",
    "src/train_mlflow.py",
    "src/train_registry.py",
    "src/model_registry.py",
    "src/extended/data_validation.py",
    "src/extended/quality_gate.py",
    "src/extended/pipeline.py",
    "src/extended/failure_test.py",
    "src/extended/inference.py",
    "src/extended/monitor_drift.py",
    "pipelines/run_lab3_baseline.py",
    "pipelines/run_lab4_tracking.py",
    "pipelines/run_lab5_pipeline.py",
    "pipelines/run_lab6_registry.py",
    "api/app.py",
    "Dockerfile",
    "deployment/kubernetes.yaml",
    ".github/workflows/wine-quality-ci.yml",
    "tests/test_extended_labs.py",
)


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(f"Project files are missing:\n{formatted}")
    print(f"Project setup verified: {len(REQUIRED_FILES)} required files are present.")


if __name__ == "__main__":
    main()

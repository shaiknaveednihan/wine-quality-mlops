import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    ARTIFACTS,
    DATA,
    EXPERIMENT,
    FEATURES,
    MINIMUM_R2,
    MODELS,
    PRODUCTION_MODEL,
    PRODUCTION_PREPROCESSOR,
    PROCESSED,
    REGISTERED_MODEL,
    REPORTS,
    TARGET,
)
from data_validation import load_dataset
from model_registry import promote_latest_model, write_production_report
from quality_gate import passes_quality_gate


def run() -> dict[str, object]:
    for directory in (PROCESSED, MODELS, REPORTS, ARTIFACTS):
        directory.mkdir(parents=True, exist_ok=True)

    frame = load_dataset(DATA).drop_duplicates().reset_index(drop=True)
    if len(frame) < 5:
        raise ValueError("At least five unique dataset rows are required")

    features = frame[FEATURES]
    target = frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    joblib.dump(preprocessor, PRODUCTION_PREPROCESSOR)
    np.save(PROCESSED / "X_train_final.npy", X_train_processed)
    np.save(PROCESSED / "X_test_final.npy", X_test_processed)
    np.save(PROCESSED / "y_train.npy", y_train.to_numpy())
    np.save(PROCESSED / "y_test.npy", y_test.to_numpy())

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_processed, y_train)
    predictions = model.predict(X_test_processed)
    metrics = {
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }
    gate_passed = passes_quality_gate(metrics["r2"], MINIMUM_R2)
    report = {
        "rows": len(frame),
        "features": FEATURES,
        "metrics": metrics,
        "quality_gate_passed": gate_passed,
    }
    (REPORTS / "evaluation_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (REPORTS / "quality_gate_report.json").write_text(
        json.dumps(
            {
                "metric": "r2",
                "minimum": MINIMUM_R2,
                "actual": metrics["r2"],
                "passed": gate_passed,
            },
            indent=2,
        )
        + "\n"
    )
    joblib.dump(model, PRODUCTION_MODEL)
    print(json.dumps(report, indent=2))
    if not gate_passed:
        raise SystemExit("Quality gate failed; model registration and promotion blocked.")

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="WineQuality_Lab7_13") as active_run:
        mlflow.log_params(
            {
                "model": "RandomForestRegressor",
                "n_estimators": 200,
                "max_depth": 20,
                "random_state": 42,
                "minimum_r2": MINIMUM_R2,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_param("dataset_sha256", hashlib.sha256(DATA.read_bytes()).hexdigest())
        mlflow.log_artifact(str(PRODUCTION_PREPROCESSOR), artifact_path="preprocessing")
        mlflow.log_artifact(str(REPORTS / "evaluation_report.json"), artifact_path="reports")
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=REGISTERED_MODEL,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        run_id = active_run.info.run_id

    promoted = promote_latest_model(run_id)
    if promoted.run_id != run_id:
        raise RuntimeError(
            "The production model version does not match the quality-gated run"
        )
    production_report = write_production_report()
    if production_report["status"] != "READY":
        raise RuntimeError("The registered production model did not pass its quality gate")

    print("PIPELINE PASSED")
    return report


if __name__ == "__main__":
    run()

import json
from typing import Any

from mlflow.tracking import MlflowClient

from src.extended.config import ARTIFACTS, MINIMUM_R2, REGISTERED_MODEL

PRODUCTION_REPORT = ARTIFACTS / "production_model_report.json"


def _client() -> MlflowClient:
    return MlflowClient()


def promote_latest_model(run_id: str | None = None) -> Any:
    client = _client()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    if not versions:
        raise RuntimeError(f"No registered versions found for {REGISTERED_MODEL}")

    candidates = versions
    if run_id is not None:
        candidates = [version for version in versions if version.run_id == run_id]
        if not candidates:
            raise RuntimeError(f"No registered model version found for run {run_id}")
    selected = max(candidates, key=lambda version: int(version.version))
    if selected.current_stage != "Production":
        client.transition_model_version_stage(
            name=REGISTERED_MODEL,
            version=selected.version,
            stage="Production",
            archive_existing_versions=True,
        )

    promoted = client.get_model_version(REGISTERED_MODEL, selected.version)
    if promoted.current_stage != "Production":
        raise RuntimeError(
            f"Model version {promoted.version} was not promoted to Production"
        )
    return promoted


def production_model_report() -> dict[str, Any]:
    client = _client()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    production_versions = [
        version for version in versions if version.current_stage == "Production"
    ]
    if not production_versions:
        return {
            "model": REGISTERED_MODEL,
            "stage": None,
            "status": "NOT_READY",
            "quality_gate_passed": False,
        }

    version = max(production_versions, key=lambda item: int(item.version))
    run = client.get_run(version.run_id)
    metrics = run.data.metrics
    r2 = metrics.get("r2", metrics.get("r2_score"))
    passed = r2 is not None and r2 >= MINIMUM_R2

    return {
        "model": REGISTERED_MODEL,
        "stage": version.current_stage,
        "status": "READY" if passed else "QUALITY_GATE_FAILED",
        "version": version.version,
        "run_id": version.run_id,
        "metrics": metrics,
        "quality_gate": {"metric": "r2", "minimum": MINIMUM_R2, "passed": passed},
        "quality_gate_passed": passed,
    }


def write_production_report() -> dict[str, Any]:
    report = production_model_report()
    PRODUCTION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    PRODUCTION_REPORT.write_text(json.dumps(report, indent=2) + "\n")
    return report

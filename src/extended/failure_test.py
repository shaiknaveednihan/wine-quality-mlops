import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA, REPORTS
from data_validation import load_dataset
from quality_gate import passes_quality_gate


def run() -> dict[str, object]:
    checks: dict[str, bool] = {}
    errors: dict[str, str] = {}

    try:
        dataset = load_dataset(DATA)
        checks["dataset_valid"] = not dataset.empty
    except (FileNotFoundError, ValueError, OSError) as exc:
        checks["dataset_valid"] = False
        errors["dataset_valid"] = str(exc)

    checks["quality_gate_accepts_threshold"] = passes_quality_gate(0.30, 0.30)
    checks["quality_gate_rejects_below_threshold"] = not passes_quality_gate(0.29, 0.30)
    checks["quality_gate_rejects_non_finite"] = not passes_quality_gate(float("nan"), 0.30)

    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {"checks": checks, "errors": errors, "passed": all(checks.values())}
    (REPORTS / "failure_test_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Failure checks did not pass")
    return report


if __name__ == "__main__":
    run()

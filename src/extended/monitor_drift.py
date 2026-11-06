import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scipy.stats import ks_2samp

from config import DATA, FEATURES, REPORTS
from data_validation import load_dataset

P_VALUE_THRESHOLD = 0.05


def run() -> dict[str, object]:
    frame = load_dataset(DATA)
    baseline = frame.sample(frac=0.7, random_state=42)
    current = frame.drop(index=baseline.index)
    if baseline.empty or current.empty:
        raise ValueError("Drift monitoring requires non-empty baseline and current data")

    results = {}
    for column in FEATURES:
        statistic, p_value = ks_2samp(baseline[column], current[column])
        results[column] = {
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": bool(p_value < P_VALUE_THRESHOLD),
        }

    report = {
        "method": "two-sample Kolmogorov-Smirnov test",
        "p_value_threshold": P_VALUE_THRESHOLD,
        "comparison": {
            "baseline_rows": len(baseline),
            "current_rows": len(current),
            "source": str(DATA.relative_to(ROOT)),
        },
        "features": results,
        "drift_detected": any(result["drift_detected"] for result in results.values()),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS / "drift_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Drift report: {report_path}")
    return report


if __name__ == "__main__":
    run()

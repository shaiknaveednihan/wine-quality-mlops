import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from model_registry import write_production_report


def run() -> dict[str, object]:
    report = write_production_report()
    print(f"[REGISTRY REPORT] {report}")
    if report["status"] != "READY":
        raise SystemExit("No quality-gated production model is available")
    return report


if __name__ == "__main__":
    run()

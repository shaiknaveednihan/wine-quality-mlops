import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from model_registry import promote_latest_model


def run() -> None:
    promoted = promote_latest_model()
    print(
        f"[SUCCESS] Model {promoted.name} version {promoted.version} "
        "is in Production"
    )


if __name__ == "__main__":
    run()

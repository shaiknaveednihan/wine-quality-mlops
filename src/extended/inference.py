import sys
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd

from config import FEATURES, PRODUCTION_MODEL, PRODUCTION_PREPROCESSOR


def predict(payload: Mapping[str, float]) -> float:
    if set(payload) != set(FEATURES):
        missing = sorted(set(FEATURES) - set(payload))
        unexpected = sorted(set(payload) - set(FEATURES))
        raise ValueError(
            f"Feature mismatch; missing: {missing}; unexpected: {unexpected}"
        )

    values = np.asarray([payload[name] for name in FEATURES], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Feature values must be finite numbers")

    if not PRODUCTION_PREPROCESSOR.is_file() or not PRODUCTION_MODEL.is_file():
        raise FileNotFoundError("Run the training pipeline to create model artifacts")

    preprocessor = joblib.load(PRODUCTION_PREPROCESSOR)
    model = joblib.load(PRODUCTION_MODEL)
    frame = pd.DataFrame([payload], columns=FEATURES)
    result = float(model.predict(preprocessor.transform(frame))[0])
    if not np.isfinite(result):
        raise ValueError("Model produced a non-finite prediction")
    return result


if __name__ == "__main__":
    sample = {
        "fixed acidity": 7.4,
        "volatile acidity": 0.7,
        "citric acid": 0.0,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11,
        "total sulfur dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }
    print({"predicted_quality": predict(sample)})

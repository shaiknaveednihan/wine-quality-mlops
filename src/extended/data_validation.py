from pathlib import Path

import numpy as np
import pandas as pd

from config import FEATURES, TARGET


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Dataset is empty: {path}")

    frame = pd.read_csv(path, sep=None, engine="python")
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names")

    expected = set(FEATURES) | {TARGET}
    actual = set(frame.columns)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"Dataset schema mismatch; missing columns: {missing}; "
            f"unexpected columns: {unexpected}"
        )
    if frame.empty:
        raise ValueError("Dataset contains no rows")

    frame = frame[[*FEATURES, TARGET]]
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Dataset contains missing or non-finite values")

    return frame

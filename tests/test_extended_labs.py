import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/extended"))
sys.path.insert(0, str(ROOT / "api"))

from config import DATA, FEATURES, TARGET
from data_validation import load_dataset
from quality_gate import passes_quality_gate
from app import Wine


def test_included_dataset_matches_training_schema() -> None:
    frame = load_dataset(DATA)

    assert list(frame.columns) == [*FEATURES, TARGET]
    assert not frame.empty


def test_data_validation_rejects_missing_columns(tmp_path: Path) -> None:
    dataset = tmp_path / "incomplete.csv"
    pd.DataFrame({"quality": [5], "alcohol": [9.4]}).to_csv(dataset, index=False)

    with pytest.raises(ValueError, match="schema mismatch"):
        load_dataset(dataset)


def test_data_validation_rejects_non_finite_values(tmp_path: Path) -> None:
    dataset = tmp_path / "non-finite.csv"
    pd.DataFrame(
        [{**{feature: 1.0 for feature in FEATURES}, TARGET: float("inf")}]
    ).to_csv(dataset, index=False)

    with pytest.raises(ValueError, match="non-finite"):
        load_dataset(dataset)


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.30, True), (0.31, True), (0.29, False), (float("nan"), False)],
)
def test_quality_gate(score: float, expected: bool) -> None:
    assert passes_quality_gate(score, 0.30) is expected


def test_api_accepts_dataset_names_and_python_field_names() -> None:
    features = {feature: 1.0 for feature in FEATURES}

    from_dataset = Wine.model_validate(features)
    from_python_names = Wine.model_validate(
        {
            "fixed_acidity": 7.4,
            "volatile_acidity": 0.7,
            "citric_acid": 0.0,
            "residual_sugar": 1.9,
            "chlorides": 0.076,
            "free_sulfur_dioxide": 11,
            "total_sulfur_dioxide": 34,
            "density": 0.9978,
            "pH": 3.51,
            "sulphates": 0.56,
            "alcohol": 9.4,
        }
    )

    assert from_dataset.as_features() == features
    assert list(from_python_names.as_features()) == FEATURES

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw/winequality-red.csv"
PROCESSED = ROOT / "data/processed"
MODELS = ROOT / "models"
PRODUCTION_PREPROCESSOR = MODELS / "production_preprocessor.pkl"
PRODUCTION_MODEL = MODELS / "production_model.pkl"
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"
ARTIFACTS = ROOT / "artifacts"
EXPERIMENT = "Wine_Quality_Prediction"
REGISTERED_MODEL = "Wine_Quality_Production_Model"
TARGET = "quality"
MINIMUM_R2 = 0.30
FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

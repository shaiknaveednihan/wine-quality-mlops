import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/extended"))

from config import FEATURES, PRODUCTION_MODEL, PRODUCTION_PREPROCESSOR
from inference import predict

app = FastAPI(title="Wine Quality Prediction API", version="1.0")


class Wine(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    fixed_acidity: float = Field(alias="fixed acidity")
    volatile_acidity: float = Field(alias="volatile acidity")
    citric_acid: float = Field(alias="citric acid")
    residual_sugar: float = Field(alias="residual sugar")
    chlorides: float
    free_sulfur_dioxide: float = Field(alias="free sulfur dioxide")
    total_sulfur_dioxide: float = Field(alias="total sulfur dioxide")
    density: float
    pH: float
    sulphates: float
    alcohol: float

    def as_features(self) -> dict[str, float]:
        return {
            "fixed acidity": self.fixed_acidity,
            "volatile acidity": self.volatile_acidity,
            "citric acid": self.citric_acid,
            "residual sugar": self.residual_sugar,
            "chlorides": self.chlorides,
            "free sulfur dioxide": self.free_sulfur_dioxide,
            "total sulfur dioxide": self.total_sulfur_dioxide,
            "density": self.density,
            "pH": self.pH,
            "sulphates": self.sulphates,
            "alcohol": self.alcohol,
        }


@app.get("/")
def root() -> dict[str, str]:
    return {"project": "Wine Quality Prediction", "status": "running"}


@app.get("/health")
def health() -> dict[str, object]:
    model_ready = all(
        path.is_file() for path in (PRODUCTION_PREPROCESSOR, PRODUCTION_MODEL)
    )
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model artifacts are unavailable")
    return {"status": "ok", "model_exists": model_ready, "features": FEATURES}


@app.post("/predict")
def predict_wine(wine: Wine) -> dict[str, float]:
    try:
        prediction = predict(wine.as_features())
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503, detail="Model artifacts are unavailable"
        ) from exc
    return {"predicted_quality": prediction}

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.extended.config import EXPERIMENT, MINIMUM_R2, REGISTERED_MODEL

def train_and_register():
    print("[INFO] Training & Registering Model (Lab 6)...")
    
    # 1. Load Processed Data
    X_train = np.load('data/processed/X_train_final.npy')
    X_test = np.load('data/processed/X_test_final.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    # 2. Train Challenger Model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=25,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))
    print(f"[SUCCESS] Model Trained | RMSE: {rmse:.4f} | R^2: {r2:.4f}")
    if not np.isfinite(r2) or r2 < MINIMUM_R2:
        raise SystemExit(
            f"Quality gate failed (r2={r2:.4f}, minimum={MINIMUM_R2}); "
            "model registration blocked."
        )
    
    # 3. Register in MLflow (with skops_trusted_types support)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="RandomForest_Registry_V2"):
        mlflow.log_params({
            "n_estimators": 200,
            "max_depth": 25,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42
        })
        mlflow.log_metrics({"rmse": rmse, "r2_score": r2})
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=REGISTERED_MODEL,
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        print("[SUCCESS] Model successfully registered in MLflow Model Registry.")
        
    # 4. Save local backup
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest_model_v2.pkl")
    print("[SUCCESS] Local model backup saved to models/random_forest_model_v2.pkl")

if __name__ == "__main__":
    train_and_register()
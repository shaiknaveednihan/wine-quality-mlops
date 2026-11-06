import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run():
    print("[INFO] Evaluating Baseline Model (Lab 2)...")
    X = np.load("data/processed/X_test_final.npy")
    y = np.load("data/processed/y_test.npy")
    m = joblib.load("models/random_forest_baseline.pkl")
    p = m.predict(X)
    print(f"\nTest RMSE: {np.sqrt(mean_squared_error(y, p)):.4f} | R2: {r2_score(y, p):.4f}\n")
    raw = pd.read_csv("data/raw/winequality-red.csv", sep=None, engine="python").drop_duplicates().reset_index(drop=True)
    _, X_raw, _, y_raw = train_test_split(raw.drop(columns=["quality"]), raw["quality"], test_size=0.2, random_state=42)
    df = X_raw.copy()
    df["actual"] = y_raw.values
    df["pred"] = p
    df["residual"] = df["actual"] - df["pred"]
    os.makedirs("outputs", exist_ok=True)
    df.sort_values(by="residual", ascending=True).head(20).to_csv("outputs/top_overpredicted.csv", index=False)
    df.sort_values(by="residual", ascending=False).head(20).to_csv("outputs/top_underpredicted.csv", index=False)
    print("[SUCCESS] Residuals saved to outputs/")

if __name__ == "__main__":
    run()

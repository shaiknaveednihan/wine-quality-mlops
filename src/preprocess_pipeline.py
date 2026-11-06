import os, joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def run():
    df = pd.read_csv("data/raw/winequality-red.csv", sep=None, engine="python").drop_duplicates().reset_index(drop=True)
    X, y = df.drop(columns=["quality"]), df["quality"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    pipe.fit(X_tr)
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/preprocessor.pkl")
    print("[SUCCESS] Pipeline Saved to models/preprocessor.pkl")

if __name__ == "__main__":
    run()

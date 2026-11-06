import os, json, joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def cap_outliers_iqr(df_tr, df_te, cols):
    dtr, dte = df_tr.copy(), df_te.copy()
    for c in cols:
        q1, q3 = df_tr[c].quantile(0.25), df_tr[c].quantile(0.75)
        iqr = q3 - q1
        dtr[c] = dtr[c].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)
        dte[c] = dte[c].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)
    return dtr, dte

def run():
    print("[INFO] Starting Wine Quality Preprocessing (Lab 1)...")
    df = pd.read_csv("data/raw/winequality-red.csv", sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.drop_duplicates().reset_index(drop=True)
    X, y = df.drop(columns=["quality"]), df["quality"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr, X_te = cap_outliers_iqr(X_tr, X_te, list(X.columns))
    X_tr["sulfur_dioxide_ratio"] = X_tr["free sulfur dioxide"] / X_tr["total sulfur dioxide"]
    X_te["sulfur_dioxide_ratio"] = X_te["free sulfur dioxide"] / X_te["total sulfur dioxide"]
    sc = StandardScaler()
    X_tr_sc, X_te_sc = sc.fit_transform(X_tr), sc.transform(X_te)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    np.save("data/processed/X_train_final.npy", X_tr_sc)
    np.save("data/processed/X_test_final.npy", X_te_sc)
    np.save("data/processed/y_train.npy", y_tr.to_numpy(dtype=np.float64))
    np.save("data/processed/y_test.npy", y_te.to_numpy(dtype=np.float64))
    joblib.dump(sc, "models/scaler.pkl")
    with open("data/processed/dataset_metadata.json", "w") as jf:
        json.dump({"dataset": "winequality-red", "train_shape": list(X_tr_sc.shape), "test_shape": list(X_te_sc.shape)}, jf, indent=4)
    print("[SUCCESS] Lab 1 Preprocessing Done!")

if __name__ == "__main__":
    run()

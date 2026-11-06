import pandas as pd

def run():
    df = pd.read_csv("data/raw/winequality-red.csv", sep=None, engine="python")
    assert df.isnull().sum().sum() == 0, "Null values detected"
    assert len(df.columns) >= 12, "Missing columns"
    print("[SUCCESS] Schema Validation PASSED")

if __name__ == "__main__":
    run()

from __future__ import annotations
import argparse
from pathlib import Path
import joblib

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from src.data import load_csv
from src.features import make_features
from src.train_model import time_split
from src.config import Config

MODEL_DIR = Path("models")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()

    artifact_path = MODEL_DIR / f"{args.ticker}_rf.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model not found: {artifact_path}. Run train_model first.")
    artifact = joblib.load(artifact_path)
    model = artifact["model"]
    cols = artifact["feature_cols"]

    raw = load_csv(args.ticker)
    df = make_features(raw)

    cfg = Config()
    _, _, test = time_split(df, cfg)

    X_test, y_test = test[cols], test["y_up"]
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print("=== Test Metrics ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion matrix [ [TN FP], [FN TP] ]:")
    print(confusion_matrix(y_test, pred))
    print("\nClassification report:")
    print(classification_report(y_test, pred, digits=4))

if __name__ == "__main__":
    main()

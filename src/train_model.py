from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.config import Config
from src.data import load_csv
from src.features import make_features, feature_columns
from src.utils import ensure_dir, set_seed

MODEL_DIR = Path("models")

def time_split(df: pd.DataFrame, cfg: Config):
    n = len(df)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    return train, val, test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    set_seed(cfg.seed)

    raw = load_csv(args.ticker)
    df = make_features(raw)

    cols = feature_columns()
    train, val, _ = time_split(df, cfg)

    X_train, y_train = train[cols], train["y_up"]
    X_val, y_val = val[cols], val["y_up"]

    if args.tune:
        param_grid = {
            "n_estimators": [100, 200, 300, 500],
            "min_samples_leaf": [1, 2, 5],
            "max_features": ["sqrt", "log2"],
        }
        base = RandomForestClassifier(
            random_state=cfg.seed,
            n_jobs=-1,
        )
        gs = GridSearchCV(
            base,
            param_grid,
            scoring="f1",
            cv=TimeSeriesSplit(n_splits=5),
            n_jobs=-1,
        )
        gs.fit(X_train, y_train)
        print(f"Best params: {gs.best_params_}")
        print(f"Best CV F1:  {gs.best_score_:.4f}")
        model = gs.best_estimator_
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=cfg.seed,
            min_samples_leaf=2,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

    pred_val = model.predict(X_val)
    acc = accuracy_score(y_val, pred_val)
    f1 = f1_score(y_val, pred_val)

    print("=== Validation Metrics ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion matrix [ [TN FP], [FN TP] ]:")
    print(confusion_matrix(y_val, pred_val))
    print("\nClassification report:")
    print(classification_report(y_val, pred_val, digits=4))

    ensure_dir(MODEL_DIR)
    artifact = {
        "model": model,
        "feature_cols": cols,
        "config": cfg,
        "ticker": args.ticker,
    }
    out_path = MODEL_DIR / f"{args.ticker}_rf.joblib"
    joblib.dump(artifact, out_path)
    print(f"\nSaved model artifact: {out_path}")

if __name__ == "__main__":
    main()

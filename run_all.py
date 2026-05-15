from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data import load_csv
from src.features import make_features
from src.train_model import time_split
from src.utils import ensure_dir

TICKERS = ["AAPL", "TSLA", "AMD", "NFLX", "F"]
START = "2018-01-01"
END = "2024-01-01"

BACKTEST_RE = re.compile(
    r"(\w[\w&]*):\s*return=\s*([-+]?\d+\.\d+)%\s*\|\s*maxDD=\s*(\d+\.\d+)%\s*\|\s*trades=(\d+)"
)


def run(cmd: list[str]) -> tuple[int, str]:
    import subprocess
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, combined


def classification_metrics(ticker: str) -> dict:
    artifact_path = ROOT / "models" / f"{ticker}_rf.joblib"
    artifact = joblib.load(artifact_path)
    model = artifact["model"]
    cols = artifact["feature_cols"]
    cfg = artifact["config"]

    raw = load_csv(ticker)
    df = make_features(raw)
    _, _, test = time_split(df, cfg)

    X_test, y_test = test[cols], test["y_up"]
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


def parse_backtest(output: str) -> dict:
    result = {}
    for name, ret, mdd, trades in BACKTEST_RE.findall(output):
        result[name] = {
            "return": float(ret),
            "maxDD": float(mdd),
            "trades": int(trades),
        }
    return result


def build_row(ticker: str, clf: dict, fin: dict) -> dict:
    row: dict = {"ticker": ticker}
    row.update({f"clf_{k}": v for k, v in clf.items()})
    for strategy in ("Agent", "Buy&Hold", "Random"):
        prefix = strategy.replace("&", "")
        data = fin.get(strategy, {"return": float("nan"), "maxDD": float("nan"), "trades": float("nan")})
        row[f"{prefix}_return%"] = data["return"]
        row[f"{prefix}_maxDD%"] = data["maxDD"]
        row[f"{prefix}_trades"] = data["trades"]
    return row


def pretty_table(df: pd.DataFrame) -> str:
    clf_cols = ["clf_accuracy", "clf_precision", "clf_recall", "clf_f1"]
    fin_cols = [c for c in df.columns if c not in ["ticker"] + clf_cols]

    clf_display = df[["ticker"] + clf_cols].copy()
    clf_display.columns = ["Ticker", "Accuracy", "Precision", "Recall", "F1"]

    fin_display = df[["ticker"] + fin_cols].copy()
    fin_display.columns = (
        ["Ticker"]
        + [c.replace("_", " ") for c in fin_cols]
    )

    lines = [
        "=== Classification Metrics (Test Set) ===",
        clf_display.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "=== Financial Metrics (Backtest) ===",
        fin_display.to_string(index=False, float_format=lambda x: f"{x:.2f}"),
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    ensure_dir(ROOT / "outputs")

    rows = []

    for ticker in TICKERS:
        print(f"\n{'='*60}")
        print(f"  {ticker}")
        print(f"{'='*60}")

        # 1) Data
        csv_path = ROOT / "data" / "raw" / f"{ticker}.csv"
        if not args.skip_data and not csv_path.exists():
            print(f"[data] downloading {ticker}...")
            cmd = [sys.executable, "-m", "src.data", "--ticker", ticker, "--start", START, "--end", END]
            rc, out = run(cmd)
            print(out.strip())
            if rc != 0:
                print(f"[ERROR] data step failed for {ticker}, skipping.")
                continue
        else:
            print(f"[data] skipped (CSV exists or --skip-data)")

        # 2) Train
        if not args.skip_train:
            print(f"[train] training {ticker}{'  (--tune)' if args.tune else ''}...")
            cmd = [sys.executable, "-m", "src.train_model", "--ticker", ticker]
            if args.tune:
                cmd.append("--tune")
            rc, out = run(cmd)
            print(out.strip())
            if rc != 0:
                print(f"[ERROR] train step failed for {ticker}, skipping.")
                continue
        else:
            print(f"[train] skipped (--skip-train)")

        # 3) Classification metrics on test set
        try:
            clf = classification_metrics(ticker)
            print(f"[metrics] acc={clf['accuracy']:.4f}  prec={clf['precision']:.4f}  "
                  f"rec={clf['recall']:.4f}  f1={clf['f1']:.4f}")
        except Exception as exc:
            print(f"[ERROR] could not compute metrics for {ticker}: {exc}")
            clf = {"accuracy": float("nan"), "precision": float("nan"),
                   "recall": float("nan"), "f1": float("nan")}

        # 4) Backtest
        print(f"[backtest] running {ticker}...")
        cmd = [sys.executable, "-m", "src.backtest", "--ticker", ticker]
        rc, out = run(cmd)
        print(out.strip())
        fin = parse_backtest(out)
        if not fin:
            print(f"[WARN] no backtest metrics parsed for {ticker}")

        rows.append(build_row(ticker, clf, fin))

    if not rows:
        print("\nNo results to report.")
        return

    results_df = pd.DataFrame(rows)

    numeric_cols = [c for c in results_df.columns if c != "ticker"]
    avg_row = results_df[numeric_cols].mean().to_dict()
    avg_row["ticker"] = "AVG"
    results_df = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)

    out_csv = ROOT / "outputs" / "all_results.csv"
    out_txt = ROOT / "outputs" / "all_results.txt"

    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    txt = pretty_table(results_df)
    out_txt.write_text(txt, encoding="utf-8")
    print(f"Saved: {out_txt}")

    print("\n" + txt)


if __name__ == "__main__":
    main()

import sys
import subprocess
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Intelligent Trading Agent Demo", layout="wide")

ROOT = Path(__file__).parent
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"
DATA = ROOT / "data" / "raw"

def run_cmd(args):
    """Run a command and return combined stdout/stderr as text."""
    proc = subprocess.run(
        args,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        shell=False
    )
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, out

st.title("Intelligent Trading Agent")
st.caption("Machine Learning (Random Forest) + Rule-based Policy — Demo UI (Streamlit)")

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Controls")

    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    start = st.text_input("Start date (YYYY-MM-DD)", value="2018-01-01").strip()
    end = st.text_input("End date (YYYY-MM-DD)", value="2025-01-01").strip()

    st.markdown("---")
    st.subheader("Run Pipeline")

    btn_data = st.button("1) Download / Load Data", use_container_width=True)
    btn_train = st.button("2) Train Model", use_container_width=True)
    btn_eval = st.button("3) Evaluate (Test)", use_container_width=True)
    btn_backtest = st.button("4) Backtest + Plot", use_container_width=True)

# ---- Main layout ----
colA, colB = st.columns([1, 1], gap="large")

with colA:
    st.subheader("Logs / Output")

    if "log" not in st.session_state:
        st.session_state.log = ""

    def append_log(title, text):
        st.session_state.log += f"\n\n===== {title} =====\n{text}"

    # 1) Data
    if btn_data:
        # This assumes you have src.data supporting these args.
        cmd = [sys.executable, "-m", "src.data", "--ticker", ticker, "--start", start, "--end", end]
        rc, out = run_cmd(cmd)
        append_log("Download/Load Data", out)
        st.success("Data step finished." if rc == 0 else "Data step finished with errors (see logs).")

    # 2) Train
    if btn_train:
        cmd = [sys.executable, "-m", "src.train_model", "--ticker", ticker]
        rc, out = run_cmd(cmd)
        append_log("Train Model", out)
        st.success("Training finished." if rc == 0 else "Training finished with errors (see logs).")

    # 3) Evaluate
    if btn_eval:
        cmd = [sys.executable, "-m", "src.evaluate", "--ticker", ticker]
        rc, out = run_cmd(cmd)
        append_log("Evaluate (Test)", out)
        st.success("Evaluation finished." if rc == 0 else "Evaluation finished with errors (see logs).")

    # 4) Backtest
    if btn_backtest:
        cmd = [sys.executable, "-m", "src.backtest", "--ticker", ticker]
        rc, out = run_cmd(cmd)
        append_log("Backtest", out)
        st.success("Backtest finished." if rc == 0 else "Backtest finished with errors (see logs).")

    st.code(st.session_state.log.strip() or "Run steps using the buttons on the left…", language="text")

with colB:
    st.subheader("Artifacts")

    csv_path = DATA / f"{ticker}.csv"
    model_path = MODELS / f"{ticker}_rf.joblib"
    plot_path = OUTPUTS / f"{ticker}_portfolio_comparison.png"

    st.write("**Expected files:**")
    st.write(f"- CSV: `{csv_path}` {'✅' if csv_path.exists() else '❌'}")
    st.write(f"- Model: `{model_path}` {'✅' if model_path.exists() else '❌'}")
    st.write(f"- Plot: `{plot_path}` {'✅' if plot_path.exists() else '❌'}")

    st.markdown("---")
    if plot_path.exists():
        st.image(str(plot_path), caption=f"{ticker} Portfolio Comparison (Agent vs Baselines)", use_container_width=True)
    else:
        st.info("Run “Backtest + Plot” to generate the plot.")

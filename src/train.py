import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import joblib

def short_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def main(smoke: bool = False):
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]

    if smoke:
        # 取一小部分数据，保证 CI 快速
        X = X.sample(frac=0.2, random_state=42)
        y = y.loc[X.index]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = make_pipeline(StandardScaler(with_mean=True), LinearRegression())
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)
    rmse = float(mean_squared_error(y_val, y_pred, squared=False))

    out = Path("models")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "model.pkl")
    (out / "metrics.json").write_text(json.dumps({"rmse": rmse}, indent=2))
    (out / "model_version.txt").write_text(short_git_sha())

    print(f"[train] rmse={rmse:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="use small subset for quick run")
    args = parser.parse_args()
    # 固定随机性
    np.random.seed(42)
    main(smoke=args.smoke)

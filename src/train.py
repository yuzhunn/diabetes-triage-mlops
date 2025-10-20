import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge  
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import joblib

def short_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def main(smoke: bool = False, model: str = "linear", alpha: float = 1.0):  
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]

    if smoke:
        X = X.sample(frac=0.2, random_state=42)
        y = y.loc[X.index]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model == "ridge":
        est = Ridge(alpha=alpha, random_state=42)
    else:
        est = LinearRegression()

    pipe = make_pipeline(StandardScaler(with_mean=True), est)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)
    rmse = float(mean_squared_error(y_val, y_pred, squared=False))

    out = Path("models")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "model.pkl")
    (out / "metrics.json").write_text(json.dumps({"rmse": rmse}, indent=2))

    version = "v0.2" if model == "ridge" else "v0.1"
    (out / "model_version.txt").write_text(version)

    print(f"[train] model={model}{'' if model!='ridge' else f' alpha={alpha}'} rmse={rmse:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="use small subset for quick run")
    parser.add_argument("--model", choices=["linear", "ridge"], default="linear", help="choose model type")  
    parser.add_argument("--alpha", type=float, default=1.0, help="ridge alpha (used when --model ridge)")  
    args = parser.parse_args()
    np.random.seed(42)
    main(smoke=args.smoke, model=args.model, alpha=args.alpha)
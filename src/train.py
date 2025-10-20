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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import time
import joblib

try:
    from sklearn.metrics import root_mean_squared_error as rmse_fn

    def RMSE(y_true, y_pred) -> float:
        return float(rmse_fn(y_true, y_pred))
except ImportError:
    from sklearn.metrics import mean_squared_error as mse

    def RMSE(y_true, y_pred) -> float:
        return float(mse(y_true, y_pred, squared=False))

def short_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"
    
def build_model(kind: str, alpha: float = 1.0, rf_estimators: int = 200, rf_max_depth=None):
    """Build model pipeline depending on type."""
    if kind == "linear":
        return make_pipeline(StandardScaler(with_mean=True), LinearRegression())
    if kind == "ridge":
        return make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=alpha, random_state=42))
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=rf_estimators,
            max_depth=rf_max_depth,
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model type: {kind}")

def main(
    smoke: bool = False,
    model: str = "linear",
    alpha: float = 1.0,
    search: bool = False,
    rf_estimators: int = 200,
    rf_max_depth=None,
):
    # ---- 数据加载 ----
    Xy = load_diabetes(as_frame=True)  # 需要 pandas（requirements 里已包含）
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]

    # 烟测：用少量数据以加速 CI
    if smoke:
        X = X.sample(frac=0.2, random_state=42)
        y = y.loc[X.index]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---- 训练（含 Ridge 的小网格搜索）----
    chosen_alpha = alpha
    start_train = time.perf_counter()

    if model == "ridge" and search:
        candidates = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
        best_rmse, best_alpha, best_model = None, None, None
        for a in candidates:
            m = build_model("ridge", alpha=a)
            m.fit(X_train, y_train)
            r = RMSE(y_val, m.predict(X_val))
            if best_rmse is None or r < best_rmse:
                best_rmse, best_alpha, best_model = r, a, m
        pipe = best_model
        chosen_alpha = best_alpha
        score_rmse = best_rmse
    else:
        pipe = build_model(model, alpha=alpha, rf_estimators=rf_estimators, rf_max_depth=rf_max_depth)
        pipe.fit(X_train, y_train)
        score_rmse = RMSE(y_val, pipe.predict(X_val))

    train_time_s = time.perf_counter() - start_train

    start_infer = time.perf_counter()
    _ = pipe.predict(X_val)
    total_infer_s = time.perf_counter() - start_infer
    infer_ms_per_row = (total_infer_s / len(X_val)) * 1000.0

    out = Path("models")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "model.pkl")
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "rmse": score_rmse,
                "model": model,
                "alpha": chosen_alpha if model == "ridge" else None,
                "rf_estimators": rf_estimators if model == "rf" else None,
                "rf_max_depth": rf_max_depth if model == "rf" else None,
                "train_size": int(len(X_train)),
                "val_size": int(len(X_val)),
                "train_time_s": round(train_time_s, 6),
                "infer_ms_per_row": round(infer_ms_per_row, 6),
            },
            indent=2,
        )
    )
    (out / "model_version.txt").write_text(short_git_sha())

    print(
        f"[train] model={model}"
        f"{'' if model!='ridge' else f' alpha={chosen_alpha}'} "
        f"rmse={score_rmse:.4f} "
        f"(train_time={train_time_s:.3f}s, infer≈{infer_ms_per_row:.4f} ms/row)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run a quick small training")
    parser.add_argument("--model", choices=["linear", "ridge", "rf"], default="linear", help="choose model type")
    parser.add_argument("--alpha", type=float, default=1.0, help="ridge alpha (used when --model ridge)")
    parser.add_argument("--search", action="store_true", help="grid search ridge alpha over a small candidate set")
    parser.add_argument("--rf-estimators", type=int, default=200, help="random forest n_estimators")
    parser.add_argument("--rf-max-depth", type=int, default=None, help="random forest max_depth")
    args = parser.parse_args()

    np.random.seed(42)

    main(
        smoke=args.smoke,
        model=args.model,
        alpha=args.alpha,
        search=args.search,
        rf_estimators=args.rf_estimators,
        rf_max_depth=args.rf_max_depth,
    )

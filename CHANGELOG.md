## [v0.2] - 2025-10-20
### Summary
- **Model upgraded to Ridge Regression (α=30)** with `StandardScaler`.
- Uses the same dataset, split (80/20), and random seed (=42) for fair comparison with v0.1.
- Updated CI/CD pipeline to train in GitHub Actions, and refactored Dockerfile (model copied, not retrained during build).
- Deployment verified:  
  `/health` → `{"status":"ok","model_version":"v0.2"}`  
  `/predict` → returns reasonable prediction (~198.7).

### Metrics (side-by-side)
| Metric               | v0.1 Linear | v0.2 Ridge(α=30) | Δ (Ridge - Linear) |
|:---------------------|------------:|-----------------:|-------------------:|
| RMSE                 | 53.8534     | **53.5004**      | **↓ 0.3530 (~0.65%)** |
| Infer time (ms/row)  | 0.0070      | **0.0063**       | ↓ 0.0007           |
| Train time (s)       | 0.004       | 0.017            | ↑ +0.013           |

### Notes
- Ridge model yields slightly better generalization with smaller coefficient magnitudes.
- Dockerfile now excludes in-image training to ensure consistent versioning.
- Confirmed version separation (`v0.1` → Linear, `v0.2` → Ridge) works correctly.

---

## [v0.1] - 2025-10-18
### Summary
- Baseline model using **StandardScaler + LinearRegression**.
- Fixed random seed = 42 for reproducibility.
- Split: 80% train / 20% test.
- Metrics logged to `models/metrics.json`.

### Notes
- Initial working pipeline (training → API → Docker → GHCR).  
- Established CI/CD and deterministic build workflow.  
- Endpoint `/predict` functional with correct JSON I/O.
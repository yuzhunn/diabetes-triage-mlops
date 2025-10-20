## [v0.2] - 2025-10-21
### Summary
- Model: **Ridge (alpha=30)** with StandardScaler.
- Same split (80/20) and seed=42 for fair comparison.

### Metrics (side-by-side)
| Metric               | v0.1 Linear | v0.2 Ridge(α=30) | Δ (Ridge - Linear) |
|:---------------------|------------:|-----------------:|-------------------:|
| RMSE                 | 53.8534     | **53.5004**      | **↓ 0.3530 (~0.65%)** |
| Infer time (ms/row)  | 0.0070      | **0.0063**       | ↓ 0.0007           |
| Train time (s)       | 0.004       | 0.017            | ↑ +0.013           |

## [v0.1]
### Summary
- Baseline model using **StandardScaler + LinearRegression**.
- Fixed random seed = 42 for reproducibility.
- Split: 80% train / 20% test.
- Metrics logged to `models/metrics.json`.

### Metrics
| Metric | Value |
|:-------|------:|
| RMSE   | 53.8534 |
| Train time (s) | 0.0070 |


### Notes
- Initial working pipeline (training → API → Docker → GHCR).  
- Established CI/CD and deterministic build workflow.  
- Endpoint `/predict` functional with correct JSON I/O.

---
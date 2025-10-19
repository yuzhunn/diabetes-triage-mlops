## [v0.1]
### Summary
- Baseline model using **StandardScaler + LinearRegression**.
- Fixed random seed = 42 for reproducibility.
- Split: 80% train / 20% test.
- Metrics logged to `models/metrics.json`.

### Metrics
| Metric | Value |
|:-------|------:|
| RMSE   | 58.41 |
| Train time (s) | 1.2 |
| Model size (KB) | 14 |

### Notes
- Initial working pipeline (training → API → Docker → GHCR).  
- Established CI/CD and deterministic build workflow.  
- Endpoint `/predict` functional with correct JSON I/O.

---
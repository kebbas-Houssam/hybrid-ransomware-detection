# Trained models

The deployed model set, used by `predict.py` and the FastAPI app. Files are
matched by role (`training/persistence.py::find_model_files`), so the folder
must hold exactly one file of each kind.

| File | Role | Saved with |
|---|---|---|
| `best_pso_ransomware_model.h5` | Stage 1: CNN-LSTM-Attention (input `50 × 13`, 515,329 params) | Keras 2.10.0 |
| `pso_ransomware_model_scaler_20250825_020931.pkl` | `RobustScaler` for the 13 window features | scikit-learn 1.7.0 |
| `pso_ransomware_model_xgboost_20250811_154446.pkl` | Stage 2: PSO-tuned `XGBClassifier` (72 hybrid features, 300 trees, depth 10) | XGBoost 3.0.2 |
| `pso_ransomware_model_xgb_scaler_20250825_020931.pkl` | `RobustScaler` for the 72 hybrid features | scikit-learn 1.7.0 |

Load them with:

```python
from training.persistence import load_inference_artifacts
model_system = load_inference_artifacts("models")
```

> **Note on provenance.** These are the files from the "new models" folder of
> `rsw-detection-app`. The `.h5` and both scalers come from the 2025-08-25
> training run, but the XGBoost model is from the 2025-08-11 run (no 08-25
> XGBoost file was present). XGBoost was fit on hybrid features scaled by the
> 08-11 `xgb_scaler`, whose statistics differ from the 08-25 one, so stage-2
> scores from this combination may not match the reported results. Replace
> the XGBoost file with the one from the 08-25 run (same timestamp as the scalers)
> when it is available.

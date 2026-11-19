# Trained models

The deployed model set, used by `predict.py` and the FastAPI app. All four files
come from **one training run (2025-08-11, timestamp `20250811_154446`)**. Files are
matched by role (`training/persistence.py::find_model_files`), so the folder must
hold exactly one file of each kind.

| File | Role | Saved with |
|---|---|---|
| `best_pso_ransomware_model.h5` | Stage 1: CNN-LSTM-Attention (input `50 × 13`, 515,329 params) | Keras 2.10.0 |
| `pso_ransomware_model_scaler_20250811_154446.pkl` | `RobustScaler` for the 13 window features | scikit-learn 1.7.0 |
| `pso_ransomware_model_xgboost_20250811_154446.pkl` | Stage 2: PSO-tuned `XGBClassifier` (72 hybrid features, 300 trees, depth 10) | XGBoost 3.0.2 |
| `pso_ransomware_model_xgb_scaler_20250811_154446.pkl` | `RobustScaler` for the 72 hybrid features | scikit-learn 1.7.0 |

`best_pso_ransomware_model.h5` has no timestamp because training's `ModelCheckpoint`
always writes that fixed name. In the source repo it sits in the same folder as the
other three files, and `platform_config.ini` there lists all four as one set.

Load them with:

```python
from training.persistence import load_inference_artifacts
model_system = load_inference_artifacts("models")
```

Source: `models/` folder of `rsw-detection-app`.

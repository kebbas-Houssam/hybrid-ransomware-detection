"""Loads the deployed hybrid model and runs inference on one 50 x 13 sequence."""

import logging
import os
from datetime import datetime
from typing import Optional, Tuple

import joblib
import numpy as np
import tensorflow as tf
import xgboost as xgb

from training.data import extract_hybrid_features
from training.layers import CUSTOM_OBJECTS
from training.persistence import find_model_files

from app.config import config

logger = logging.getLogger(__name__)


def default_model_files(model_dir: Optional[str] = None) -> dict:
    """Paths of the four artifacts in ``model_dir`` (defaults to models/)."""
    return find_model_files(model_dir or config.MODEL_DIR)


class ModelManager:
    """Manages the PSO-optimized hybrid model"""

    def __init__(self):
        self.primary_model = None
        self.xgb_model = None
        self.scaler = None
        self.xgb_scaler = None
        self.model_loaded = False
        self.model_info = {}

    def load_model_compatible(self, model_path):
        """Load the CNN-LSTM-Attention .h5 (custom layers from training/layers.py)."""
        logger.info(f"Loading model from {model_path}")
        return tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS, compile=False)

    def load_models(self, h5_path: Optional[str] = None, xgb_path: Optional[str] = None,
                    scaler_path: Optional[str] = None, xgb_scaler_path: Optional[str] = None):
        """Load the PSO-optimized hybrid models. Missing paths default to models/."""
        try:
            logger.info("=" * 60)
            logger.info("STARTING MODEL LOADING PROCESS")
            logger.info("=" * 60)

            if not all([h5_path, xgb_path, scaler_path, xgb_scaler_path]):
                defaults = default_model_files()
                h5_path = h5_path or defaults['primary']
                xgb_path = xgb_path or defaults['xgboost']
                scaler_path = scaler_path or defaults['scaler']
                xgb_scaler_path = xgb_scaler_path or defaults['xgb_scaler']

            for label, path in [("H5 model", h5_path), ("XGBoost model", xgb_path),
                                ("Feature scaler", scaler_path), ("XGBoost scaler", xgb_scaler_path)]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{label} file not found: {path}")

            logger.info("Loading primary CNN-LSTM-Attention model...")
            self.primary_model = self.load_model_compatible(h5_path)
            logger.info(f"Primary model loaded - Input shape: {self.primary_model.input_shape}")

            logger.info("Loading PSO-optimized XGBoost model...")
            self.xgb_model = joblib.load(xgb_path)

            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
            self.xgb_scaler = joblib.load(xgb_scaler_path)
            logger.info(f"XGBoost scaler loaded from {xgb_scaler_path}")

            self.model_loaded = True
            self.model_info = {
                'primary_model_path': h5_path,
                'xgb_model_path': xgb_path,
                'scaler_path': scaler_path,
                'xgb_scaler_path': xgb_scaler_path,
                'loaded_at': datetime.now().isoformat(),
                'input_shape': self.primary_model.input_shape,
                'tensorflow_version': tf.__version__,
                'xgboost_version': xgb.__version__
            }

            logger.info("MODEL LOADING COMPLETED SUCCESSFULLY!")
            return True

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            self.model_loaded = False
            return False

    def predict(self, sequence: np.ndarray) -> Tuple[float, float]:
        """Predict on one raw (unscaled) ``(50, 13)`` sequence.

        Returns ``(cnn_lstm_score, hybrid_score)``.
        """
        try:
            if not self.model_loaded:
                raise Exception("Models not loaded")

            sequence = np.asarray(sequence, dtype=np.float64)
            if sequence.ndim == 2:
                sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])

            # Same per-feature RobustScaler as in training
            scaled = self.scaler.transform(sequence.reshape(-1, sequence.shape[-1])).reshape(sequence.shape)

            # Primary model prediction
            y_pred_primary = self.primary_model.predict(scaled, verbose=0)

            # Hybrid features -> scaled -> XGBoost
            hybrid_features = extract_hybrid_features(scaled, y_pred_primary, verbose=False)
            hybrid_features_scaled = self.xgb_scaler.transform(hybrid_features)
            y_pred_hybrid = self.xgb_model.predict_proba(hybrid_features_scaled)[0][1]

            return float(y_pred_primary[0][0]), float(y_pred_hybrid)

        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return 0.0, 0.0

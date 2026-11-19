"""Data preparation: I/O-trace windows -> sequences, scaling, splits and
the hand-crafted statistical features fed to XGBoost.

This module has no TensorFlow dependency so it can be reused by the API and
``predict.py`` for inference.
"""

import traceback
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

# The 13 per-window features (order matters: it is the model's input order).
FEATURE_COLUMNS = ['read_write_ratio', 'war_ratio', 'wss', 'entropy',
                   'read_pct', 'write_pct', 'repeat_ratio',
                   'read_entropy', 'write_entropy',
                   'total_ops', 'write_to_unique_ratio',
                   'avg_offset_gap', 'burstiness']


def prepare_data(df, sequence_length=50, overlap=0.5):
    """Prepare time-series sequences (same as before)"""
    try:
        feature_columns = FEATURE_COLUMNS

        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        X_sequences = []
        y_sequences = []
        file_sequences = []

        for file_name in df['file'].unique():
            file_data = df[df['file'] == file_name].sort_values('start_time')

            if len(file_data) < sequence_length:
                print(f"Skipping {file_name}: insufficient data ({len(file_data)} < {sequence_length})")
                continue

            features = file_data[feature_columns].values

            if np.isnan(features).any():
                print(f"Warning: NaN values found in {file_name}, filling with 0")
                features = np.nan_to_num(features, nan=0.0)

            features = np.clip(features, -1e10, 1e10)

            label = file_data['label'].iloc[0]
            step_size = max(1, int(sequence_length * (1 - overlap)))

            for i in range(0, len(features) - sequence_length + 1, step_size):
                sequence = features[i:i + sequence_length]

                if not np.isfinite(sequence).all():
                    print(f"Warning: Non-finite values in sequence from {file_name}")
                    continue

                X_sequences.append(sequence)
                y_sequences.append(label)
                file_sequences.append(file_name)

        if len(X_sequences) == 0:
            raise ValueError("No valid sequences could be created from the data")

        X = np.array(X_sequences, dtype=np.float32)
        y = np.array(y_sequences, dtype=np.int32)

        print(f"Created {len(X)} sequences of length {sequence_length}")
        print(f"Feature shape: {X.shape}")
        print(f"Label distribution: {Counter(y)}")

        return X, y, file_sequences

    except Exception as e:
        print(f"Error in prepare_data: {e}")
        traceback.print_exc()
        raise


def scale_sequences(X):
    """Fit a RobustScaler on all time steps and scale the sequences.

    Returns ``(X_scaled, scaler)``.
    """
    scaler = RobustScaler()
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled_reshaped = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled_reshaped.reshape(X.shape)

    # Data validation
    if not np.isfinite(X_scaled).all():
        print("Warning: Non-finite values detected, cleaning...")
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    return X_scaled, scaler


def split_data(X, y, test_size=0.2, val_size=0.25, random_state=42):
    """Stratified train / validation / test split (60 / 20 / 20 by default)."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def handle_class_imbalance(y_train):
    """Calculate class weights for imbalanced dataset"""
    try:
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))

        print(f"Class distribution: {Counter(y_train)}")
        print(f"Class weights: {class_weight_dict}")

        return class_weight_dict
    except Exception as e:
        print(f"Error calculating class weights: {e}")
        return None


def extract_hybrid_features(X_sequences, y_pred_proba_primary, verbose=True):
    """Extract features for XGBoost second-level classifier.

    Output: 2 primary-model features + 13*5 per-feature stats + 5 global stats
    = 72 columns.
    """
    try:
        if verbose:
            print("Extracting hybrid features for PSO-optimized XGBoost...")

        # Primary model predictions
        primary_features = []
        primary_features.append(y_pred_proba_primary.flatten())
        primary_features.append((y_pred_proba_primary > 0.5).astype(int).flatten())

        # Statistical features from sequences
        sequence_stats = []

        for i, seq in enumerate(X_sequences):
            stats = []

            # Per-feature statistics across time steps
            stats.extend([
                np.mean(seq, axis=0),
                np.std(seq, axis=0),
                np.min(seq, axis=0),
                np.max(seq, axis=0),
                np.median(seq, axis=0),
            ])

            # Global sequence statistics
            stats.extend([
                np.mean(seq),
                np.std(seq),
                np.var(seq),
                np.ptp(seq),
                np.mean(np.diff(seq, axis=0)**2)
            ])

            flattened_stats = np.concatenate([np.array(s).flatten() for s in stats])
            sequence_stats.append(flattened_stats)

        sequence_stats = np.array(sequence_stats)

        # Combine all features
        hybrid_features = np.column_stack([
            np.array(primary_features).T,
            sequence_stats
        ])

        if verbose:
            print(f"Hybrid features shape: {hybrid_features.shape}")

        return hybrid_features

    except Exception as e:
        print(f"Error extracting hybrid features: {e}")
        traceback.print_exc()
        raise

"""Predict ransomware vs. benign from a CSV of 5-second I/O windows.

The CSV needs the 13 feature columns listed in data/README.md. Optional columns:
``file`` (one sequence stream per file/process) and ``start_time`` (row order).
Each stream is cut into sequences of 50 windows (stride 15, i.e. 70 % overlap
as in training); streams shorter than 50 windows are front-padded with their
first window.

    python predict.py path/to/windows.csv [--output predictions.csv]
"""

import argparse
import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pandas as pd

from training.data import FEATURE_COLUMNS, clean_windows, pad_sequence, scale_with
from training.persistence import DEFAULT_MODEL_DIR, load_inference_artifacts


def build_sequences(df, sequence_length, stride):
    """Return ``(X, stream_ids, start_rows)`` with X shaped (n, sequence_length, 13)."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    groups = df.groupby('file', sort=False) if 'file' in df.columns else [('all', df)]
    X, ids, starts = [], [], []
    for stream_id, g in groups:
        if 'start_time' in g.columns:
            g = g.sort_values('start_time')
        windows = clean_windows(g[FEATURE_COLUMNS].values)
        if len(windows) < sequence_length:
            X.append(pad_sequence(windows, sequence_length))
            ids.append(stream_id)
            starts.append(0)
            continue
        for i in range(0, len(windows) - sequence_length + 1, stride):
            X.append(windows[i:i + sequence_length])
            ids.append(stream_id)
            starts.append(i)
    return np.array(X, dtype=np.float32), ids, starts


def main(argv=None):
    parser = argparse.ArgumentParser(description="Hybrid CNN-LSTM-Attention + XGBoost ransomware prediction")
    parser.add_argument('csv', help='CSV with the 13 window features')
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR, help='folder with the .h5/.pkl artifacts')
    parser.add_argument('--stride', type=int, default=15, help='step between sequences (default 15 = 70%% overlap)')
    parser.add_argument('--threshold', type=float, default=0.5, help='decision threshold on the hybrid score')
    parser.add_argument('--output', help='optional CSV to write predictions to')
    args = parser.parse_args(argv)

    model_system = load_inference_artifacts(args.model_dir)
    seq_len = model_system.time_steps

    df = pd.read_csv(args.csv)
    X, ids, starts = build_sequences(df, seq_len, args.stride)
    X_scaled = scale_with(model_system.scaler, X)

    p_primary, p_hybrid = model_system.predict_pso_hybrid(X_scaled, verbose=False)
    if p_primary is None:
        print("Prediction failed", file=sys.stderr)
        return 1

    out = pd.DataFrame({
        'stream': ids,
        'start_row': starts,
        'score_cnn_lstm': p_primary.flatten(),
        'score_hybrid': p_hybrid.flatten(),
    })
    out['prediction'] = np.where(out['score_hybrid'] > args.threshold, 'ransomware', 'benign')

    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.output:
        out.to_csv(args.output, index=False)
        print(f"\nSaved {len(out)} predictions to {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

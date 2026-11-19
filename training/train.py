"""Train the PSO-optimized CNN-LSTM-Attention + XGBoost ransomware detector.

Usage (from the repository root):

    python -m training.train --data data/full_dataset_5s.csv --config training/config.yaml
"""

import argparse
import os
import sys
import time
import traceback
import warnings

if __package__ in (None, ''):  # allow `python training/train.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import tensorflow as tf
import yaml

from training.data import scale_sequences, split_data
from training.evaluate import evaluate_pso_model
from training.model import XGBOOST_AVAILABLE, PSOHybridRansomwareDetectionModel
from training.persistence import generate_pso_report, save_pso_model

warnings.filterwarnings('ignore')

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')


def main_pso_training_pipeline(cfg):
    """
    Main training pipeline with PSO optimization
    """
    csv_file_path = cfg['data']['csv_path']
    sequence_length = cfg['data']['sequence_length']
    pso_particles = cfg['pso']['particles']
    pso_iterations = cfg['pso']['iterations']
    output_dir = cfg['output']['dir']
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("PSO-OPTIMIZED HYBRID RANSOMWARE DETECTION MODEL TRAINING")
    print("CNN-LSTM-Attention + PSO-Optimized XGBoost")
    print("="*80)

    # Print system information
    print(f"TensorFlow version: {tf.__version__}")
    print(f"CUDA available: {tf.test.is_built_with_cuda()}")
    print(f"GPU available: {tf.test.is_gpu_available()}")
    print(f"XGBoost available: {XGBOOST_AVAILABLE}")
    print(f"PSO Configuration: {pso_particles} particles, {pso_iterations} iterations")

    # Check file existence
    if not os.path.exists(csv_file_path):
        print(f"Error: File not found at {csv_file_path}")
        return None, None

    try:
        # Load and validate data
        print("\nLoading and validating data...")
        df = pd.read_csv(csv_file_path)
        print(f"Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print(f"Unique files: {df['file'].nunique()}")
        print(f"Label distribution: {df['label'].value_counts().to_dict()}")

        # Initialize PSO-optimized model
        m = cfg['model']
        model_system = PSOHybridRansomwareDetectionModel(
            time_steps=sequence_length,
            n_features=m['n_features'],
            lstm_units=m['lstm_units'],
            cnn_filters=m['cnn_filters'],
            attention_dim=m['attention_dim'],
            dropout_rate=m['dropout_rate'],
            output_dir=output_dir
        )

        # Prepare sequences
        print(f"\nPreparing sequences (length: {sequence_length})...")
        X, y, file_names = model_system.prepare_data(
            df, sequence_length=sequence_length, overlap=cfg['data']['overlap'])

        if len(X) == 0:
            print("Error: No sequences could be created")
            return None, None

        # Scale features
        print("Scaling features...")
        X_scaled, scaler = scale_sequences(X)
        model_system.scaler = scaler

        # Split data
        print("Splitting data...")
        s = cfg['split']
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X_scaled, y, test_size=s['test_size'], val_size=s['val_size'],
            random_state=s['random_state'])

        print(f"Data split:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")

        # Build and compile model
        print("\nBuilding CNN-LSTM-Attention model...")
        model_system.build_model()
        model_system.compile_model(learning_rate=cfg['training']['learning_rate'])

        print("\nModel Architecture:")
        model_system.model.summary()

        # Train PSO-optimized hybrid model
        print("\n" + "="*80)
        print("STARTING PSO-OPTIMIZED HYBRID TRAINING")
        print("="*80)

        start_time = time.time()

        history = model_system.train_model(
            X_train, y_train, X_val, y_val,
            epochs=cfg['training']['epochs'],
            batch_size=cfg['training']['batch_size'],
            verbose=1,
            pso_particles=pso_particles,
            pso_iterations=pso_iterations
        )

        training_time = time.time() - start_time

        # Evaluate PSO-optimized model
        print("\n" + "="*80)
        print("EVALUATING PSO-OPTIMIZED MODEL")
        print("="*80)

        results = evaluate_pso_model(model_system, X_test, y_test, plot_results=True,
                                     output_dir=output_dir)

        # Save PSO model
        base_path = os.path.join(output_dir, cfg['output']['base_name'])
        timestamp = save_pso_model(model_system, base_path)

        # Generate comprehensive report
        if results:
            generate_pso_report(model_system, results, training_time,
                                os.path.join(output_dir, f'pso_model_report_{timestamp}.txt'))

        print(f"\n" + "="*80)
        print("PSO-OPTIMIZED TRAINING COMPLETED!")
        print("="*80)
        print(f"Total training time: {training_time:.1f} seconds")

        if results and 'pso_hybrid' in results:
            print(f"\nFINAL RESULTS COMPARISON:")
            print(f"Primary Model:")
            for key, value in results['primary'].items():
                if key != 'loss':
                    print(f"  {key.capitalize()}: {value:.4f}")

            print(f"\nPSO-Optimized Hybrid Model:")
            for key, value in results['pso_hybrid'].items():
                print(f"  {key.capitalize()}: {value:.4f}")

            print(f"\nPSO Optimization Results:")
            pso_results = results.get('pso_results', {})
            if pso_results.get('best_fitness'):
                print(f"  Best PSO Fitness: {pso_results['best_fitness']:.4f}")
            if pso_results.get('convergence_iterations'):
                print(f"  Convergence Iterations: {pso_results['convergence_iterations']}")

            # Highlight key improvements
            improvements = results.get('improvement', {})
            precision_gain = improvements.get('precision_gain', 0)
            f2_gain = improvements.get('f2_gain', 0)

            print(f"\nKEY IMPROVEMENTS:")
            print(f"  Precision Gain: {precision_gain:+.4f}")
            print(f"  F2 Score Gain: {f2_gain:+.4f}")

            if precision_gain > 0.01:
                print(f"\n SUCCESS: PSO optimization significantly improved precision!")
                print("   Recommended for production deployment.")
            elif precision_gain > 0:
                print(f"\n GOOD: PSO optimization improved precision modestly.")
                print("   Consider production deployment with monitoring.")
            else:
                print(f"\n NOTE: PSO optimization didn't improve precision.")
                print("   Consider adjusting PSO parameters or feature engineering.")

        print(f"\nArtifacts written to: {os.path.abspath(output_dir)}")
        return model_system, results

    except Exception as e:
        print(f"Error in PSO training pipeline: {e}")
        traceback.print_exc()
        return None, None


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data', help='CSV of 5 s I/O windows (overrides data.csv_path)')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='YAML config (default: training/config.yaml)')
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.data:
        cfg['data']['csv_path'] = args.data

    model_system, results = main_pso_training_pipeline(cfg)
    if model_system is None or results is None:
        print("\nPSO-OPTIMIZED TRAINING FAILED - check error messages above")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
